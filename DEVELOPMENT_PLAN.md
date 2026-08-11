# RustDAW Development Plan

## 1. Product definition

RustDAW is an Ubuntu-first professional desktop digital audio workstation written primarily in Rust. Its first marketable identity should be a reliable multitrack audio recorder, editor, and mixer for musicians and small studios—not a full Pro Tools clone on day one.

### Product principles

- Never lose or corrupt a recording.
- The audio callback must be deterministic: no allocation, locks, file access, logging, or blocking system calls.
- Editing is non-destructive and undoable.
- Sessions are recoverable after a crash or power loss.
- The engine is independent of the UI, operating system backend, and plug-in format.
- Every milestone ends in a usable, testable application.

### Initial platform scope

- Ubuntu LTS on x86-64
- PipeWire/JACK audio and ALSA MIDI
- WAV/BWF import, recording, and export
- CLAP plug-in hosting first; VST3 after legal and technical review
- English UI initially

Windows, macOS, AAX, video post-production, cloud collaboration, and control-surface depth are later projects. AAX compatibility in particular requires a commercial relationship and must not be treated as an early dependency.

## 2. Release targets

### Prototype — engine proof (8–12 weeks)

Goal: prove that the real-time architecture is viable on supported Ubuntu hardware.

- Enumerate audio devices and configure sample rate/buffer size
- Duplex low-latency audio through PipeWire/JACK
- Lock-free command and metering channels
- Transport, sample clock, click, and one stereo playback track
- Stream audio from disk with read-ahead buffering
- Record one or more inputs to temporary WAV files
- XRUN, callback-time, and disk-throughput telemetry
- Offline engine tests and a 30-minute recording soak test

Exit criteria: glitch-free playback and recording at the chosen baseline buffer on reference hardware, bounded callback time, and recovery of a deliberately interrupted recording.

### Milestone 1 — recorder/editor alpha (4–6 months)

- Multi-track recording and playback
- Track arm, mute, solo, gain, pan, input monitoring, and meters
- Timeline navigation, snapping, selection, split, trim, move, fades, and crossfades
- Non-destructive clips referencing immutable source media
- Waveform peak-cache generation
- Undo/redo command model
- Autosave journal, crash recovery, and missing-media relinking
- WAV/BWF import and stereo/multichannel export
- Basic keyboard-driven editing

Exit criteria: complete a real 16-track recording/edit/export session without destructive data loss.

### Milestone 2 — mixer beta (4–6 months)

- Bus graph, sends, returns, groups, and master output
- Sample-accurate automation for volume, pan, mute, and plug-in parameters
- Delay-compensation model and latency reporting
- Freeze/bounce-in-place and offline render
- CLAP plug-in discovery, hosting, state persistence, and automation
- Plug-in scanner and host in separate processes
- Session templates and routing presets
- Performance profiler visible to users

Exit criteria: repeatable offline/real-time renders, correct compensation tests, and survival/recovery when a test plug-in hangs or crashes.

### Milestone 3 — MIDI and instruments (4–6 months)

- Timestamped MIDI input/output
- MIDI clips, piano roll, quantize, velocity, and controller lanes
- Instrument tracks and virtual-instrument hosting
- Tempo and meter maps; musical-time and absolute-time clip modes
- Loop recording, takes, comping, punch, pre-roll, and count-in
- External synchronization investigation and prototype

Exit criteria: record, edit, and render a song containing audio, MIDI, tempo changes, and virtual instruments.

### Version 1.0 — production hardening (6–9 months)

- Accessibility pass and scalable UI
- Configurable shortcuts and localization foundation
- Session interchange: prioritize AAF investigation; document compatibility limits
- Control-surface foundation and MIDI learn
- Packaging, signed releases, migration tests, and support diagnostics
- Large-session optimization and extensive hardware/plug-in compatibility tests
- User documentation, tutorials, and issue-report bundles

Exit criteria: beta users can depend on the application for real projects, old sessions migrate safely, and release-blocking performance/reliability budgets pass.

For a focused team of 4–6 experienced engineers, a credible 1.0 target is roughly 24–36 months. One developer can build a valuable DAW, but Pro Tools breadth and reliability is more realistically a multi-year effort with deliberately narrower early scope.

## 3. Proposed architecture

```text
UI / application shell
        |
commands, snapshots, events
        v
session model + undo journal ---- project persistence
        |
compiled immutable graph
        v
real-time engine ---- lock-free queues ---- disk I/O workers
        |                                     plug-in workers
        v
audio/MIDI backend (PipeWire/JACK; later platform adapters)
```

### Workspace boundaries

```text
crates/
  daw-core/          IDs, time, channels, parameters, shared domain types
  daw-session/       editable session model and commands
  daw-engine/        render graph, scheduling, transport, automation
  daw-dsp/           mixer primitives, meters, resampling helpers
  daw-audio-linux/   PipeWire/JACK and MIDI adapters
  daw-media/         audio decoding, recording, peak caches, disk streaming
  daw-plugin-api/    format-neutral plug-in abstraction
  daw-plugin-host/   scanner and isolated worker protocol
  daw-project/       manifest, journal, migrations, media integrity
  daw-ui/            desktop application and custom timeline/mixer widgets
  daw-render/        offline bounce/export
  daw-testkit/       deterministic engine, stress, and fixture utilities
apps/
  rustdaw/           desktop binary
  plugin-scanner/    disposable discovery process
  plugin-worker/     isolated hosting process
docs/
```

Keep framework-specific types out of `daw-core`, `daw-session`, and `daw-engine`. For the UI, prototype the timeline and mixer with `winit` plus a GPU renderer such as `wgpu`; select a higher-level Rust UI toolkit only after testing text input, accessibility, docking, drag behavior, and rendering thousands of clips. Professional DAW widgets usually need substantial custom rendering.

### Real-time contract

The audio thread owns only preallocated buffers and an immutable render plan. The control side builds a replacement plan, then atomically publishes it at a block boundary. Commands and meter data use bounded single-producer/single-consumer queues. Destruction and memory reclamation happen away from the callback.

Enforce the contract through code review and tests:

- no mutex/RwLock acquisition in processing code
- no heap allocation or deallocation in the callback
- no filesystem, network, console, or unbounded operation
- bounded work per block, including automation and graph traversal
- denormal handling and explicit floating-point behavior
- stable monotonic sample position as the timing authority

## 4. Core data and persistence decisions

- Use stable UUID-like IDs for project objects; never persist memory addresses or array positions as identity.
- Store a versioned, human-inspectable project manifest plus immutable media files and generated caches.
- Append edits to a checksummed journal; periodically write a new atomic snapshot.
- Record first to a recoverable temporary media file, updating durable headers/side metadata periodically, then adopt it into the project.
- Represent timeline locations with integer sample positions; add rational musical time for tempo-aware content.
- Make schema migration one-way per release but always retain backups and fixtures for every historical version.
- Treat peak files, thumbnails, and analysis as disposable caches.

## 5. Quality strategy

### Automated tests

- Unit tests for time conversion, fades, pan laws, automation interpolation, and serialization
- Property tests for edit invariants, graph topology, and project round trips
- Golden-sample DSP tests with tolerances and denormal/NaN cases
- Deterministic offline renders compared by hash or signal metrics
- Fuzzing for project parsers, audio importers, and plug-in IPC messages
- Fault injection for short writes, full disks, disconnected devices, and crashed workers
- Migration fixtures from every released project version
- UI interaction and screenshot tests for critical workflows

### Performance gates

Define budgets before optimization:

- callback maximum and percentile time relative to buffer deadline
- allocations per callback: zero after initialization
- maximum UI frame time during playback
- waveform-cache generation throughput
- project open/save time at reference sizes
- track/plug-in capacity on named reference machines

Run continuous benchmarks on a stable Ubuntu reference machine; virtual CI is insufficient for low-latency claims.

## 6. Engineering process

- Use short architecture decision records for irreversible choices.
- Keep `main` releasable and gate changes with formatting, linting, tests, sanitizer-compatible native checks, and benchmarks.
- Ship an internal build every week and a user-test build at each milestone.
- Maintain a small hardware matrix: integrated audio, two USB interfaces, multiple channel counts, 44.1/48/96 kHz, and several buffer sizes.
- Recruit recording engineers early. Observe complete sessions rather than validating isolated screens.
- Track reliability metrics: recording failures, recovery success, XRUNs, crashes, and project migration failures.

## 7. Team shape

An effective initial team has these responsibilities (people may cover more than one):

- real-time/audio engine and DSP
- Linux audio/MIDI and hardware integration
- session model, storage, and recovery
- desktop UI/interaction design
- plug-in hosting and process isolation
- QA, performance lab, and release engineering

The first specialist hire should usually be an engineer with shipped real-time audio software experience. Visual similarity to a professional DAW is far easier than matching its recording safety and timing behavior.

## 8. Major risks and mitigations

| Risk | Mitigation |
|---|---|
| Audio glitches from accidental blocking/allocation | Narrow RT-safe API, preallocation, callback instrumentation, stress tests |
| Plug-in crashes or hangs | Separate scanner/worker processes, timeouts, state snapshots, quarantine list |
| Session corruption | Append-only journal, atomic snapshots, checksums, migration fixtures, recovery drills |
| UI framework limitations | Build a timeline/mixer spike before committing; isolate UI behind application commands |
| Linux hardware variability | Reference hardware matrix, backend diagnostics, reproducible support bundle |
| Scope explosion | Recorder-first product definition and milestone exit criteria |
| Licensing surprises | Review every SDK and dependency before integration; keep format adapters isolated |

## 9. First six two-week iterations

1. Establish the Cargo workspace, CI, coding rules, ADR template, and deterministic offline engine harness.
2. Build a synthetic render graph with gain/mix nodes, transport, sample clock, and golden-output tests.
3. Add the Linux backend spike, device selection, duplex pass-through, XRUN counters, and timing telemetry.
4. Add disk read-ahead and safe recording writers; test interruption and disk-full recovery.
5. Build a minimal desktop shell showing transport, one track, waveform, meters, and backend diagnostics.
6. Integrate multitrack record/playback and run the first long-form reference-hardware test.

At the end of iteration 6, review measured latency, callback headroom, disk behavior, and UI feasibility. Only then lock the main UI toolkit and proceed to editor-alpha work.

## 10. Decisions to make before coding beyond the prototype

- Open-source, commercial, or dual-license product
- Exact minimum Ubuntu release and packaging method
- Primary users: music recording, mixing, podcasting, or post-production
- Maximum reference session for 1.0 (tracks, sample rate, plug-in count)
- Whether CLAP-only is acceptable for the first public alpha
- Whether the first UI should emulate established DAW workflows or establish a distinct interaction model
- Available team, monthly budget, and desired first-public-build date

