# RustDAW

An Ubuntu-first audio recorder and digital audio workstation written in Rust.

The first release targets mono/stereo tracks, recording through a Focusrite
Scarlett Solo 4th Gen (or a USB pedal), playback, and a sample-accurate click.

## Current development commands

```bash
cargo test --workspace
cargo run -p hardware-probe
cargo run -p rustdaw
```

The hardware probe is read-only: it lists audio hosts, devices, default stream
formats, and supported channel/sample-rate ranges. It does not open a stream.

The desktop application opens the Scarlett through PipeWire's PulseAudio
compatibility service. In the initial edit window you can add mono/stereo
tracks, select one of four capture channels, arm a track, enable the click, and
record 24-bit WAV takes into `Recordings/`. Recorded clips play from their
sample-accurate timeline positions. `Ctrl+S` saves the current versioned session
to `Sessions/Current.rustdaw.json`; it is restored on the next launch. Select a
clip and press `Delete` to remove it non-destructively (the WAV is preserved).

Use **Audio Settings** in the bottom bar to select PipeWire input/output
devices, change the buffer size, test outputs 1–2, and identify the Scarlett's
backend capture channels with live meters. Device choices and custom channel
labels are saved in `~/.config/rustdaw/audio.json`.

Each track has an **FX** insert button for the built-in EQ III, Dyn3-style
compressor, and expander/noise gate. Effects are non-destructive: they process
software monitoring, timeline playback, and stereo exports while recorded WAV
files remain dry. Insert settings are stored in the session document.
The channel-strip window uses console-style rotary controls, illuminated
module switches, a live segmented input meter, an EQ response graph, and
dynamics activity indicators.
EQ, compressor, and gate parameters are applied block-by-block in the real-time
audio engine. Turning a control sends a bounded parameter command and never
reloads or reprocesses an entire WAV file.

Drag one or more 48 kHz mono/stereo WAV files onto the edit window, or use
**IMPORT AUDIO** to choose them in Ubuntu's file explorer. Each imported file
creates a track at the current playhead. **SAVE AS** writes a named `.rustdaw`
session, while **OPEN SESSION** switches projects with unsaved-change
protection. Shortcuts: `Ctrl+O` opens a session and `Ctrl+Shift+S` saves as.

Track **M** and **S** buttons update mute/solo audibility immediately during
playback without reloading imported stems. Multiple tracks may be soloed, mute
overrides solo, and the same mixer state is saved in sessions and honored by
stereo export.

Imported and recorded media is preloaded whenever a session changes, so the
transport starts immediately when **Play** is pressed. Drag a clip horizontally
to change its timeline position, or vertically onto another audio track to move
it between tracks. Moving uses a real-time engine command and does not reload or
reprocess the WAV. The red **×** in a track header removes that track after a
confirmation; source WAV files are always preserved.

Tracks and clips have persistent UUIDs, including automatic migration when an
older session is opened. Decoded WAV data is retained in a shared immutable
cache: on the eight-stem reference session, the first release-mode load takes
about 3 seconds and subsequent playback schedule rebuilds take approximately
0.005 ms. Clip moves and keyboard nudges use UUID-based edit commands and can
be reversed with **UNDO** / `Ctrl+Z`, then restored with **REDO** /
`Ctrl+Shift+Z`.

Open the floating **MIX** window from the bottom bar or with `Ctrl+M`. The
first mixer release provides a horizontally scrolling console strip for every
track: insert access, send placeholders, I/O labels, real-time pan, Mute/Solo,
vertical gain fader, and independent playback meter. Mixer controls update the
audio thread immediately and pan is persisted and included in stereo export.
Mixer channels and plug-in modules use explicit vertical layouts and fixed
control widths so labels, buttons, knobs, faders, and meters remain separated
at 4K resolution and when the floating windows are resized.
The native window embeds the RustDAW icon and the desktop launcher declares
`StartupWMClass=rustdaw`, allowing GNOME Shell to associate running windows
with the installed launcher instead of showing a generic application icon.

## MIDI, the piano roll, and tempo

Instrument tracks hold notes rather than audio and are played by a built-in
polyphonic synth. Open the **PIANO ROLL** from the bottom bar, or double-click a
MIDI clip in the timeline: double-click adds a note, drag moves it, its right
edge changes the length, `Del` removes it, and the grid snaps to anything from
whole notes to 1/32. Velocity sets each note's brightness as well as its level.
Notes are stored in ticks, not seconds, so changing the tempo moves the notes
and leaves recorded audio where it is.

Sessions carry a **tempo map**. A song at one steady tempo stores one entry; a
song that really moves stores the changes and the bar lines follow. Session
files are now version 2 — version 1 sessions open unchanged and gain a constant
tempo map at their stored tempo.

**Tempo is detected natively**, from the audio, in Rust: spectral-flux onsets
over an in-house FFT, a global tempo chosen by autocorrelation with harmonic
summation and a perceptual prior, then beats fitted by dynamic programming.
Deciding one tempo first and fitting beats to it is what keeps a steady song at
one tempo instead of recording the tracker's own error as tempo change.

```bash
cargo run --release -p daw-analysis --example detect-tempo -- drums.wav
cargo run -p daw-midi --example dump-midi -- song.mid
```

Measured against the kick-drum onsets of real songs, where 25% of a beat is
what random alignment scores:

| Song | RustDAW | DSPRO Studio | librosa |
|---|---|---|---|
| King Von — Armed & Dangerous | **15.6%** | 24.3% | 23.6% |
| Bruno & Marrone — Bijuteria | **12.5%** | 24.1% | 16.4% |
| Zezé Di Camargo — No Dia em Que Eu Saí de Casa | **5.8%** | 24.7% | 24.5% |

## Instruments and chords

Instrument tracks play a **General MIDI bank** synthesised in Rust: all 128
programs, described by a harmonic recipe, an envelope, a brightness and a
little noise, plus the channel-10 drum kit as pitch-swept tones and filtered
noise. A piano decays on its own, an organ holds, a flute is mostly breath, and
a kick is not a low beep. Nothing is downloaded and nothing is bundled — a
sampled bank would sound better and cost a 140 MB dependency that sessions
would break without. Imported MIDI keeps the program each track asks for, and
drum tracks land on the kit.

**Chords are detected natively too.** A chromagram with the harmonic series
discounted, averaged over half-beats from the detected beat grid, matched
against ten chord qualities and decoded by Viterbi so the chart holds still
instead of flickering between relative keys. The key comes from
Krumhansl–Schmuckler profiles and the bass register is read separately, which
is what names inversions like `D/F#`. Drums and vocals are excluded from the
input: percussion has no pitch, and a singer's passing notes are not the chord.

Scored against each song's independently transcribed MIDI — what fraction of
sounding note-time the chart actually explains:

| Song | RustDAW | DSPRO Studio |
|---|---|---|
| Zezé Di Camargo — No Dia em Que Eu Saí de Casa | **66.1%** | 66.5% |
| Andre Renner — Será Que Está Pensando Em Mim | 72.1% | **74.5%** |
| King Von — Armed & Dangerous | **63.4%** | 54.5% |
| Leonardo — Pot Pourri | 39.3% | **40.0%** |

Ahead on average, and much further ahead on the trap track, where the song is
one sustained harmony and a chart with 155 chord changes is describing noise.

```bash
cargo run --release -p daw-analysis --example detect-chords -- <project-dir>
```

## Song import

**IMPORT SONG** in the bottom bar turns a song into instrument tracks you can
play along with. Paste a link and the song is downloaded and separated on the
GPU into drums, bass, guitar, piano, other and vocals; each stem becomes a
stereo track, and the session tempo and meter come from the detected beat grid.
Songs that have already been processed are listed in the same window and import
in about a second, because only format conversion is left to do.

The transcription is imported too: each pitched MIDI track becomes an instrument
track you can edit in the piano roll and hear against the stems. Drum MIDI is
left out — the synth is pitched and cannot play a kit, and the drum stem already
covers it. Tempo comes from the detector above rather than from the pipeline's
beat grid, and the notes are rebased through seconds so a transcription written
at 120 BPM still lines up when the song turns out to be 94.

The separation itself is Demucs, running in the DSPRO Studio Python worker
already installed in `~/.local/share/chords-extraction`. RustDAW talks to it
over loopback HTTP and starts it if it is not answering; nothing is uploaded
anywhere. Set `CHORDS_STUDIO_LAUNCHER` if `bin/chords-studio-servers` lives
somewhere unusual.

Stems are produced at 44.1 kHz, so they are converted once at import to the
session's rate with ffmpeg and written into `Songs/<name>/Audio/` as 24-bit
WAV — the engine refuses mismatched media rather than resampling during
playback. Expect roughly 200–300 MB per song. Silent stems are skipped, and by
default the song is delayed by less than a bar so its first downbeat lands on
bar 1 of the click.

Import a song without the desktop app:

```bash
cargo run -p daw-songimport --example import-song              # list processed songs
cargo run -p daw-songimport --example import-song -- <id>      # import one
cargo run -p daw-songimport --example import-song -- <url>     # separate, then import
```

For a two-second command-line recording check using Scarlett Input 2:

```bash
cargo run -p daw-audio-linux --example recording-smoke
```

Run the hardware soak test for a chosen number of seconds (60 by default):

```bash
cargo run -p daw-audio-linux --example recording-soak -- 60
```

## Ubuntu package

Build the optimized `.deb` package with:

```bash
./packaging/build-deb.sh
```

The package is written to `dist/rustdaw_0.6.1_amd64.deb` and installs the
`rustdaw` executable, desktop launcher, and application icon. Installation is
explicit and remains under the user's control:

```bash
sudo apt install ./dist/rustdaw_0.6.1_amd64.deb
```

See [MVP_RECORDING_PLAN.md](MVP_RECORDING_PLAN.md) for the scoped roadmap.
