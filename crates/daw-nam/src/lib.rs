//! A small, safe ownership boundary around NeuralAmpModelerCore, and the
//! library of captures on disk that feeds it.

pub mod library;

pub use library::{AMP_DIR_ENV, AmpModel, amp_dir, discover, search_paths};

use std::ffi::{CStr, CString, c_char, c_double, c_int, c_void};
use std::path::Path;
use std::ptr::NonNull;

unsafe extern "C" {
    fn rustdaw_nam_load(
        path: *const c_char,
        sample_rate: c_double,
        max_block: c_int,
    ) -> *mut c_void;
    fn rustdaw_nam_free(model: *mut c_void);
    fn rustdaw_nam_loudness(model: *mut c_void, out: *mut c_double) -> bool;
    fn rustdaw_nam_process(model: *mut c_void, samples: *mut f32, frames: c_int) -> bool;
    fn rustdaw_nam_last_error() -> *const c_char;
}

/// A loaded mono NAM model. Loading and dropping must happen off the audio callback.
pub struct NamProcessor {
    model: NonNull<c_void>,
}

// NAM processors own their state and are only accessed through `&mut self`.
unsafe impl Send for NamProcessor {}

impl NamProcessor {
    /// Loads and prewarms a model for the given stream format.
    pub fn load(path: &Path, sample_rate: u32, max_block: usize) -> Result<Self, String> {
        let path = CString::new(path.to_string_lossy().as_bytes())
            .map_err(|_| "NAM model path contains a NUL byte".to_owned())?;
        let block =
            c_int::try_from(max_block).map_err(|_| "audio block is too large".to_owned())?;
        // SAFETY: the string lives through the call and the bridge owns the returned allocation.
        let model = unsafe { rustdaw_nam_load(path.as_ptr(), f64::from(sample_rate), block) };
        NonNull::new(model)
            .map(|model| Self { model })
            .ok_or_else(last_error)
    }

    /// The loudness the model was measured at, in dB, when it knows it.
    ///
    /// Captures vary enormously in level — a survey of the reference models
    /// spanned a factor of six hundred — so levelling them against each other
    /// is the difference between swapping amps and re-dialling the gain
    /// staging every time. Models trained without a loudness return `None`.
    #[must_use]
    pub fn loudness(&self) -> Option<f64> {
        let mut value = 0.0_f64;
        // SAFETY: the model is exclusively borrowed and `value` is writable.
        let known = unsafe { rustdaw_nam_loudness(self.model.as_ptr(), &raw mut value) };
        known.then_some(value)
    }

    /// Processes one mono block in place without allocating.
    pub fn process(&mut self, samples: &mut [f32]) -> Result<(), String> {
        let frames =
            c_int::try_from(samples.len()).map_err(|_| "audio block is too large".to_owned())?;
        // SAFETY: `samples` is writable for `frames` elements and the model is exclusively borrowed.
        if unsafe { rustdaw_nam_process(self.model.as_ptr(), samples.as_mut_ptr(), frames) } {
            Ok(())
        } else {
            Err(last_error())
        }
    }
}

impl Drop for NamProcessor {
    fn drop(&mut self) {
        // SAFETY: this pointer was returned by `rustdaw_nam_load` and is freed exactly once.
        unsafe { rustdaw_nam_free(self.model.as_ptr()) };
    }
}

fn last_error() -> String {
    // SAFETY: the bridge returns a thread-local, NUL-terminated string.
    let pointer = unsafe { rustdaw_nam_last_error() };
    if pointer.is_null() {
        return "Neural Amp Modeler failed without an error message".to_owned();
    }
    // SAFETY: checked non-null and owned by the bridge for the duration of this call.
    unsafe { CStr::from_ptr(pointer) }
        .to_string_lossy()
        .into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_and_processes_reference_model() {
        let model = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../third_party/NeuralAmpModelerCore/example_models/lstm.nam");
        let mut processor = NamProcessor::load(&model, 48_000, 256).unwrap();
        let mut audio = vec![0.01_f32; 256];
        processor.process(&mut audio).unwrap();
        assert!(audio.iter().all(|sample| sample.is_finite()));
        assert!(
            audio
                .iter()
                .any(|sample| (*sample - 0.01).abs() > f32::EPSILON)
        );
    }
}
