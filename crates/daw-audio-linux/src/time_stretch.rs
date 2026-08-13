//! Real-time, pitch-preserving time-stretch for the playback mix (WSOLA).
//!
//! Changing a song's tempo by resampling also shifts its pitch, like a
//! turntable. WSOLA — Waveform Similarity Overlap-Add — instead advances through
//! the source at a different rate while overlap-adding short windows, choosing
//! each window by waveform similarity so the pitch is preserved and the joins
//! stay phase-coherent.
//!
//! The stretcher pulls source (mix) frames strictly forward through a supplied
//! closure, so the caller's stateful voices are never rewound. It owns fixed
//! ring buffers and allocates nothing while running.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)]

/// Analysis/synthesis window length in frames (~11 ms at 48 kHz).
const FRAME: usize = 512;
/// Synthesis hop: half the window, so Hann windows overlap-add to unity gain.
const HOP: usize = FRAME / 2;
/// How far, in frames, the similarity search may nudge the next window.
const SEARCH: usize = 128;
/// Correlation length for the similarity search.
const CORR: usize = HOP;
/// Source ring capacity. Must exceed the furthest read ahead of the analysis
/// point: `HOP * MAX_RATIO + FRAME + SEARCH`, with generous headroom.
const RING: usize = 4096;

/// Time-domain time-stretcher for one stereo stream.
pub struct TimeStretcher {
    window: [f32; FRAME],
    ring_left: [f32; RING],
    ring_right: [f32; RING],
    /// Absolute count of source frames pulled into the ring so far.
    filled: u64,
    /// Absolute source frame where the current analysis window starts.
    analysis: u64,
    /// The second half of the previous windowed frame, awaiting overlap-add.
    tail_left: [f32; HOP],
    tail_right: [f32; HOP],
    /// Finished output frames not yet handed back, as a small ring FIFO.
    fifo_left: [f32; FRAME],
    fifo_right: [f32; FRAME],
    fifo_head: usize,
    fifo_len: usize,
}

impl Default for TimeStretcher {
    fn default() -> Self {
        Self::new()
    }
}

impl TimeStretcher {
    #[must_use]
    pub fn new() -> Self {
        let mut window = [0.0_f32; FRAME];
        for (index, value) in window.iter_mut().enumerate() {
            // Periodic Hann; two of them, hopped by HOP, sum to 1.
            let phase = std::f32::consts::TAU * index as f32 / FRAME as f32;
            *value = 0.5 - 0.5 * phase.cos();
        }
        Self {
            window,
            ring_left: [0.0; RING],
            ring_right: [0.0; RING],
            filled: 0,
            analysis: 0,
            tail_left: [0.0; HOP],
            tail_right: [0.0; HOP],
            fifo_left: [0.0; FRAME],
            fifo_right: [0.0; FRAME],
            fifo_head: 0,
            fifo_len: 0,
        }
    }

    /// Drops all buffered state, for a stop or seek.
    pub fn reset(&mut self) {
        self.filled = 0;
        self.analysis = 0;
        self.tail_left = [0.0; HOP];
        self.tail_right = [0.0; HOP];
        self.fifo_head = 0;
        self.fifo_len = 0;
    }

    fn source(&self, index: u64) -> (f32, f32) {
        let slot = (index % RING as u64) as usize;
        (self.ring_left[slot], self.ring_right[slot])
    }

    /// Fills the ring forward until at least `until` absolute frames exist,
    /// pulling from `render`, which writes `n` fresh source frames into the two
    /// slices it is given.
    fn ensure_filled(
        &mut self,
        until: u64,
        render: &mut impl FnMut(usize, &mut [f32], &mut [f32]),
    ) {
        let mut scratch_left = [0.0_f32; HOP];
        let mut scratch_right = [0.0_f32; HOP];
        while self.filled < until {
            let want = ((until - self.filled) as usize).min(HOP);
            scratch_left[..want].fill(0.0);
            scratch_right[..want].fill(0.0);
            render(want, &mut scratch_left[..want], &mut scratch_right[..want]);
            for offset in 0..want {
                let slot = ((self.filled + offset as u64) % RING as u64) as usize;
                self.ring_left[slot] = scratch_left[offset];
                self.ring_right[slot] = scratch_right[offset];
            }
            self.filled += want as u64;
        }
    }

    /// Finds the offset in `[-SEARCH, SEARCH]` around `nominal` whose source best
    /// matches the natural continuation `target` of the last window, by
    /// cross-correlation of the two channels summed.
    fn best_offset(&self, nominal: u64, target: u64) -> i64 {
        let mut best_delta = 0_i64;
        let mut best_score = f32::NEG_INFINITY;
        for delta in -(SEARCH as i64)..=(SEARCH as i64) {
            let candidate = nominal as i64 + delta;
            if candidate < 0 {
                continue;
            }
            let candidate = candidate as u64;
            let mut score = 0.0_f32;
            for i in 0..CORR as u64 {
                let (cl, cr) = self.source(candidate + i);
                let (tl, tr) = self.source(target + i);
                score += (cl + cr) * (tl + tr);
            }
            if score > best_score {
                best_score = score;
                best_delta = delta;
            }
        }
        best_delta
    }

    fn push_output(&mut self, left: f32, right: f32) {
        let slot = (self.fifo_head + self.fifo_len) % FRAME;
        self.fifo_left[slot] = left;
        self.fifo_right[slot] = right;
        self.fifo_len += 1;
    }

    /// Produces one synthesis hop (`HOP` output frames) into the FIFO.
    fn synthesis_step(
        &mut self,
        ratio: f64,
        render: &mut impl FnMut(usize, &mut [f32], &mut [f32]),
    ) {
        let analysis_hop = (HOP as f64 * ratio).round().max(0.0) as u64;
        // Everything this step may read: the window, the continuation target one
        // hop ahead, and the search around the nominal next window.
        let reach = self.analysis + analysis_hop + (FRAME + SEARCH + 1) as u64;
        self.ensure_filled(reach, render);

        // Window the current frame and overlap-add its first half with the tail.
        for i in 0..HOP {
            let (l0, r0) = self.source(self.analysis + i as u64);
            let (l1, r1) = self.source(self.analysis + (i + HOP) as u64);
            let out_left = self.tail_left[i] + l0 * self.window[i];
            let out_right = self.tail_right[i] + r0 * self.window[i];
            self.push_output(out_left, out_right);
            // Keep the windowed second half as the next tail.
            self.tail_left[i] = l1 * self.window[i + HOP];
            self.tail_right[i] = r1 * self.window[i + HOP];
        }

        // Choose the next window: near the nominal analysis hop, nudged to best
        // continue the samples that naturally followed this one.
        let nominal = self.analysis + analysis_hop;
        let target = self.analysis + HOP as u64;
        let delta = self.best_offset(nominal, target);
        self.analysis = (nominal as i64 + delta).max(0) as u64;
    }

    /// Fills `out_left`/`out_right` with time-stretched output at `ratio`
    /// (output advances through the source `ratio`× as fast, pitch preserved),
    /// pulling source through `render`.
    pub fn process(
        &mut self,
        out_left: &mut [f32],
        out_right: &mut [f32],
        ratio: f64,
        mut render: impl FnMut(usize, &mut [f32], &mut [f32]),
    ) {
        let frames = out_left.len().min(out_right.len());
        let mut produced = 0;
        while produced < frames {
            if self.fifo_len == 0 {
                self.synthesis_step(ratio, &mut render);
            }
            let take = (frames - produced).min(self.fifo_len);
            for offset in 0..take {
                let slot = (self.fifo_head + offset) % FRAME;
                out_left[produced + offset] = self.fifo_left[slot];
                out_right[produced + offset] = self.fifo_right[slot];
            }
            self.fifo_head = (self.fifo_head + take) % FRAME;
            self.fifo_len -= take;
            produced += take;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Renders a mono sine (same on both channels) as the source, advancing a
    /// phase so the pulled stream is a continuous tone.
    fn sine_source(frequency: f32, rate: f32) -> impl FnMut(usize, &mut [f32], &mut [f32]) {
        let mut phase = 0.0_f32;
        move |n, left, right| {
            for i in 0..n {
                let value = (phase).sin();
                left[i] = value;
                right[i] = value;
                phase += std::f32::consts::TAU * frequency / rate;
            }
        }
    }

    fn zero_crossings(signal: &[f32]) -> usize {
        signal
            .windows(2)
            .filter(|pair| pair[0] <= 0.0 && pair[1] > 0.0)
            .count()
    }

    #[test]
    fn stretching_preserves_pitch() {
        // A 440 Hz tone stretched to any tempo must still be ~440 Hz: the count
        // of zero crossings per output second is unchanged, unlike resampling.
        let rate = 48_000.0;
        for ratio in [0.75_f64, 1.5, 2.0] {
            let mut stretcher = TimeStretcher::new();
            let mut out_left = vec![0.0_f32; 48_000];
            let mut out_right = vec![0.0_f32; 48_000];
            stretcher.process(
                &mut out_left,
                &mut out_right,
                ratio,
                sine_source(440.0, rate),
            );
            // Skip the first window where the tail primes from silence.
            let crossings = zero_crossings(&out_left[FRAME..]);
            let seconds = (out_left.len() - FRAME) as f32 / rate;
            let detected = crossings as f32 / seconds;
            assert!(
                (detected - 440.0).abs() < 30.0,
                "ratio {ratio} shifted pitch to ~{detected:.0} Hz",
            );
        }
    }

    #[test]
    fn faster_ratios_consume_more_source() {
        // At ratio 2 the stretcher must read about twice as far through the
        // source as at ratio 1 for the same amount of output.
        let rate = 48_000.0;
        let buffer_left = vec![0.0_f32; 24_000];
        let buffer_right = vec![0.0_f32; 24_000];
        let consume = |ratio: f64| -> usize {
            let mut stretcher = TimeStretcher::new();
            let mut total = 0usize;
            let mut source = sine_source(220.0, rate);
            let mut left = buffer_left.clone();
            let mut right = buffer_right.clone();
            stretcher.process(
                &mut left,
                &mut right,
                ratio,
                |count, dst_left, dst_right| {
                    total += count;
                    source(count, dst_left, dst_right);
                },
            );
            total
        };
        let slow = consume(1.0);
        let fast = consume(2.0);
        assert!(
            fast as f64 > slow as f64 * 1.7,
            "ratio 2 consumed {fast} vs ratio 1 {slow}"
        );
    }

    #[test]
    fn output_stays_finite_and_bounded() {
        let rate = 48_000.0;
        let mut stretcher = TimeStretcher::new();
        let mut out_left = vec![0.0_f32; 10_000];
        let mut out_right = vec![0.0_f32; 10_000];
        stretcher.process(
            &mut out_left,
            &mut out_right,
            1.3,
            sine_source(1000.0, rate),
        );
        assert!(
            out_left
                .iter()
                .all(|value| value.is_finite() && value.abs() <= 2.0)
        );
    }
}
