//! HTTP client for the local DSPRO Studio worker.
//!
//! The worker binds to loopback only, so no TLS is involved and no audio ever
//! leaves the machine. Job control goes over HTTP; the produced audio is read
//! straight off the filesystem by [`crate::ingest`], because both applications
//! share one project store and copying 300 MB of stems through localhost to
//! land in the same place would be pure waste.

use anyhow::{Context, Result, bail};
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use ureq::Agent;

use crate::supervisor;

/// Generous enough for a cold worker that is importing torch, short enough
/// that a wedged service surfaces as an error instead of a frozen import.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

/// An upload is one request carrying a whole song, so it gets its own, longer
/// budget than the job-control calls [`REQUEST_TIMEOUT`] covers.
const UPLOAD_TIMEOUT: Duration = Duration::from_secs(300);

#[derive(Clone, Debug, Deserialize)]
pub struct Health {
    #[serde(default)]
    pub ok: bool,
    #[serde(default)]
    pub cuda: bool,
    #[serde(default)]
    pub models: HealthModels,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct HealthModels {
    #[serde(default)]
    pub demucs: Option<String>,
    #[serde(default)]
    pub drumsep: bool,
    /// The `MuScriptor` variant the worker would transcribe with, when that
    /// backend is installed; `None` leaves the stems to basic-pitch.
    #[serde(default)]
    pub muscriptor: Option<String>,
    /// The ACE-Step install the worker would generate with, when there is
    /// one. `None` means neither songs nor renders are available.
    #[serde(default)]
    pub acestep: Option<AceStepHealth>,
}

/// What the worker knows about its ACE-Step install without starting it.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AceStepHealth {
    #[serde(default)]
    pub home: String,
    /// Whether the ACE-Step server is already answering. The first job after
    /// a cold start pays for loading — and on a fresh install, downloading —
    /// the models; the UI says so when this is false.
    #[serde(default)]
    pub running: bool,
    /// `DiT` checkpoints already on disk.
    #[serde(default)]
    pub models: Vec<String>,
    #[serde(default)]
    pub song_model: Option<String>,
    #[serde(default)]
    pub stem_model: Option<String>,
    /// The instruments the stem tasks can name, in ACE-Step's vocabulary.
    #[serde(default)]
    pub tracks: Vec<String>,
}

/// What to ask ACE-Step for when writing a song.
///
/// Serialised as the JSON body of `/api/jobs/generate`, so the field names
/// are the worker's. Anything left `None` is decided by the model and comes
/// back in the manifest.
#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerateRequest {
    /// The music, in words: genre, instruments, mood, era.
    pub prompt: String,
    /// Empty means an instrumental.
    pub lyrics: String,
    /// Seconds, 10–600.
    pub duration: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bpm: Option<u16>,
    /// "A Minor", "F# Major" — ACE-Step's own spelling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub key: Option<String>,
    /// Beats per bar: 2, 3, 4 or 6.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub time_signature: Option<u16>,
    /// ISO language of the lyrics ("en", "pt").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    /// What to file the song under.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
}

/// The three things ACE-Step can do to one track of a loaded session.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RenderMode {
    /// Re-render the track's own audio in the sound the prompt asks for. The
    /// part is kept; the timbre, and the separation artefacts, are replaced.
    Rerender,
    /// Pull the named instrument out of the whole mix afresh.
    Extract,
    /// Write a new part for the named instrument that fits the mix.
    Add,
}

impl RenderMode {
    pub const ALL: [Self; 3] = [Self::Rerender, Self::Extract, Self::Add];

    /// The name the worker's `/api/jobs/render` reads.
    #[must_use]
    pub fn wire_name(self) -> &'static str {
        match self {
            Self::Rerender => "rerender",
            Self::Extract => "extract",
            Self::Add => "add",
        }
    }

    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Rerender => "Re-render this track",
            Self::Extract => "Extract from the song",
            Self::Add => "Add a new part",
        }
    }

    #[must_use]
    pub fn description(self) -> &'static str {
        match self {
            Self::Rerender => {
                "Plays the track's own audio back through ACE-Step in the sound described. \
                 The part stays; the sound and the separation artefacts are replaced."
            }
            Self::Extract => {
                "Asks ACE-Step for this instrument from the whole song, written afresh rather \
                 than filtered out. Cleaner than a separated stem, but a performance that \
                 resembles the record, not the record itself."
            }
            Self::Add => {
                "Writes a new part on this instrument that fits the rest of the song, on a \
                 new track."
            }
        }
    }

    /// Whether the mode works on the whole mix rather than on one track.
    #[must_use]
    pub fn takes_mix(self) -> bool {
        !matches!(self, Self::Rerender)
    }
}

/// What to ask ACE-Step to do with the audio [`WorkerClient::submit_render`]
/// uploads alongside it.
#[derive(Clone, Debug)]
pub struct RenderRequest {
    pub mode: RenderMode,
    /// The instrument, in ACE-Step's vocabulary ([`AceStepHealth::tracks`]).
    /// Ignored by [`RenderMode::Rerender`], which describes the sound in the
    /// prompt instead.
    pub track: String,
    pub prompt: String,
    /// How closely a re-render keeps the part, 0.0–1.0. Only
    /// [`RenderMode::Rerender`] reads it.
    pub strength: f32,
}

/// One pipeline job. `percent` is 0–100 as sent by the worker.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Job {
    pub id: String,
    #[serde(default)]
    pub status: String,
    #[serde(default)]
    pub stage: String,
    #[serde(default)]
    pub percent: f32,
    #[serde(default)]
    pub message: String,
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default)]
    pub project_id: Option<String>,
    /// Where a render job left its WAV. Only `/api/jobs/render` sets it.
    #[serde(default)]
    pub result_path: Option<PathBuf>,
}

impl Job {
    #[must_use]
    pub fn is_finished(&self) -> bool {
        matches!(self.status.as_str(), "done" | "error")
    }

    #[must_use]
    pub fn is_failed(&self) -> bool {
        self.status == "error"
    }
}

/// Summary of an already-processed project, for the picker.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProjectSummary {
    pub id: String,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub artist: Option<String>,
    #[serde(default)]
    pub style: Option<String>,
    #[serde(default)]
    pub duration: Option<f64>,
    #[serde(default)]
    pub has_stems: bool,
}

impl ProjectSummary {
    #[must_use]
    pub fn label(&self) -> String {
        let title = self.title.as_deref().unwrap_or("Untitled");
        match self.artist.as_deref().filter(|value| !value.is_empty()) {
            Some(artist) => format!("{title} — {artist}"),
            None => title.to_owned(),
        }
    }
}

/// A `multipart/form-data` body: any plain fields, then one part named `file`,
/// which is what the worker's upload endpoints read. Written by hand because
/// these are the only multipart requests `RustDAW` makes, and they are not
/// worth a dependency.
fn multipart_body(boundary: &str, fields: &[(&str, &str)], name: &str, bytes: &[u8]) -> Vec<u8> {
    // A quote or a newline in a filename would close the header early and split
    // the request into something the worker reads as a different upload.
    let name = name.replace(['"', '\r', '\n'], "_");
    let mut body = Vec::with_capacity(bytes.len() + 512);
    for (field, value) in fields {
        body.extend_from_slice(
            format!("--{boundary}\r\nContent-Disposition: form-data; name=\"{field}\"\r\n\r\n")
                .as_bytes(),
        );
        body.extend_from_slice(value.as_bytes());
        body.extend_from_slice(b"\r\n");
    }
    let header = format!(
        "--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{name}\"\r\n\
         Content-Type: application/octet-stream\r\n\r\n"
    );
    body.extend_from_slice(header.as_bytes());
    body.extend_from_slice(bytes);
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
    body
}

pub struct WorkerClient {
    agent: Agent,
    base_url: String,
}

impl Default for WorkerClient {
    fn default() -> Self {
        Self::new()
    }
}

impl WorkerClient {
    #[must_use]
    pub fn new() -> Self {
        let agent: Agent = Agent::config_builder()
            .timeout_global(Some(REQUEST_TIMEOUT))
            .build()
            .into();
        Self {
            agent,
            base_url: supervisor::base_url(),
        }
    }

    fn url(&self, path: &str) -> String {
        format!("{}{path}", self.base_url)
    }

    /// Queries the worker's health endpoint.
    ///
    /// # Errors
    ///
    /// Returns an error if the worker is unreachable or answers unexpectedly.
    pub fn health(&self) -> Result<Health> {
        let mut response = self
            .agent
            .get(self.url("/api/health"))
            .call()
            .context("the DSPRO Studio worker is not answering")?;
        response
            .body_mut()
            .read_json::<Health>()
            .context("unexpected response from /api/health")
    }

    /// Cheap liveness probe used to decide whether to start the worker.
    #[must_use]
    pub fn is_healthy(&self) -> bool {
        self.health().is_ok_and(|health| health.ok)
    }

    /// Lists already-processed projects, newest first.
    ///
    /// # Errors
    ///
    /// Returns an error if the request or its decoding fails.
    pub fn projects(&self) -> Result<Vec<ProjectSummary>> {
        let mut response = self
            .agent
            .get(self.url("/api/projects"))
            .call()
            .context("failed to list DSPRO Studio projects")?;
        response
            .body_mut()
            .read_json::<Vec<ProjectSummary>>()
            .context("unexpected response from /api/projects")
    }

    /// Starts a pipeline run for a URL.
    ///
    /// # Errors
    ///
    /// Returns an error if the URL is not http(s) or the worker rejects it.
    pub fn submit_url(&self, url: &str) -> Result<Job> {
        let url = url.trim();
        if !(url.starts_with("http://") || url.starts_with("https://")) {
            bail!("only http(s) links are accepted");
        }
        let mut response = self
            .agent
            .post(self.url("/api/jobs"))
            .send_json(serde_json::json!({ "url": url }))
            .context("failed to start the import job")?;
        response
            .body_mut()
            .read_json::<Job>()
            .context("unexpected response from /api/jobs")
    }

    /// Starts a pipeline run for an audio file on this machine.
    ///
    /// The worker copies the bytes into its own uploads folder before the job
    /// starts, so the original is read exactly once: moving, renaming or
    /// deleting it afterwards cannot disturb a run in flight.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or the worker rejects it.
    pub fn submit_file(&self, path: &Path) -> Result<Job> {
        self.upload("/api/jobs/upload", path, &[])
    }

    /// Starts a song being written by ACE-Step and run through the pipeline.
    ///
    /// The job ends in a project like any download does: the generated mix is
    /// separated, transcribed and beat-tracked, so it imports as stems.
    ///
    /// # Errors
    ///
    /// Returns an error if the description is empty, ACE-Step is not
    /// installed on the worker, or the request fails.
    pub fn submit_generate(&self, request: &GenerateRequest) -> Result<Job> {
        if request.prompt.trim().is_empty() {
            bail!("describe the song first");
        }
        let mut response = self
            .agent
            .post(self.url("/api/jobs/generate"))
            .send_json(request)
            .context("failed to start the generation job")?;
        response
            .body_mut()
            .read_json::<Job>()
            .context("unexpected response from /api/jobs/generate")
    }

    /// Uploads one track (or a mix) for ACE-Step to re-render, extract from,
    /// or write against. The finished job's `result_path` is the WAV.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or the worker rejects it.
    pub fn submit_render(&self, path: &Path, request: &RenderRequest) -> Result<Job> {
        let strength = format!("{:.2}", request.strength);
        let fields = [
            ("mode", request.mode.wire_name()),
            ("track", request.track.as_str()),
            ("prompt", request.prompt.as_str()),
            ("strength", strength.as_str()),
        ];
        self.upload("/api/jobs/render", path, &fields)
    }

    /// One multipart request carrying `path` and `fields` to `endpoint`.
    fn upload(&self, endpoint: &str, path: &Path, fields: &[(&str, &str)]) -> Result<Job> {
        let bytes =
            std::fs::read(path).with_context(|| format!("could not read {}", path.display()))?;
        if bytes.is_empty() {
            bail!("{} is empty", path.display());
        }
        // The worker takes the extension from the name, and ffmpeg picks the
        // decoder from it, so a name is always sent even for an unnamed path.
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("upload.bin");
        // A boundary must not occur anywhere in the payload. Random beats any
        // fixed string here because the payload is arbitrary audio bytes.
        let boundary = format!("----RustDAW{}", uuid::Uuid::new_v4().simple());
        let body = multipart_body(&boundary, fields, name, &bytes);

        let agent: Agent = Agent::config_builder()
            .timeout_global(Some(UPLOAD_TIMEOUT))
            .build()
            .into();
        let mut response = agent
            .post(self.url(endpoint))
            .header(
                "Content-Type",
                &format!("multipart/form-data; boundary={boundary}"),
            )
            .send(&body[..])
            .with_context(|| format!("failed to upload the file to the worker at {endpoint}"))?;
        response
            .body_mut()
            .read_json::<Job>()
            .with_context(|| format!("unexpected response from {endpoint}"))
    }

    /// Polls a job.
    ///
    /// # Errors
    ///
    /// Returns an error if the job is unknown or the request fails.
    pub fn job(&self, job_id: &str) -> Result<Job> {
        let mut response = self
            .agent
            .get(self.url(&format!("/api/jobs/{job_id}")))
            .call()
            .with_context(|| format!("failed to read job {job_id}"))?;
        response
            .body_mut()
            .read_json::<Job>()
            .context("unexpected response from /api/jobs/{id}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn job_status_transitions_are_recognised() {
        let running = Job {
            status: "running".to_owned(),
            ..Job::default()
        };
        assert!(!running.is_finished());
        let done = Job {
            status: "done".to_owned(),
            ..Job::default()
        };
        assert!(done.is_finished() && !done.is_failed());
        let failed = Job {
            status: "error".to_owned(),
            ..Job::default()
        };
        assert!(failed.is_finished() && failed.is_failed());
    }

    #[test]
    fn worker_job_json_parses() {
        let json = br#"{"id":"abc123","status":"running","stage":"separate","stages":["download"],
            "percent":42.5,"message":"htdemucs_6s","error":null,"projectId":"20260810-214917-untitled",
            "createdAt":1.0,"updatedAt":2.0}"#;
        let job: Job = serde_json::from_slice(json).unwrap();
        assert_eq!(job.stage, "separate");
        assert!((job.percent - 42.5).abs() < f32::EPSILON);
        assert_eq!(job.project_id.as_deref(), Some("20260810-214917-untitled"));
    }

    #[test]
    fn project_summary_json_parses() {
        let json = br#"[{"id":"20260810-214917-untitled","title":"Armed & Dangerous",
            "artist":"King Von","style":"Hip Hop","sourceUrl":"https://example.invalid",
            "duration":122.8,"stages":{},"updatedAt":1.0,"hasSession":true,"hasStems":true,
            "takeCount":1}]"#;
        let projects: Vec<ProjectSummary> = serde_json::from_slice(json).unwrap();
        assert_eq!(projects[0].label(), "Armed & Dangerous — King Von");
        assert!(projects[0].has_stems);
    }

    #[test]
    fn multipart_fields_precede_the_file() {
        let body = multipart_body(
            "BOUND",
            &[("mode", "extract"), ("track", "guitar")],
            "mix.wav",
            b"RIFF",
        );
        let text = String::from_utf8_lossy(&body);
        let mode = text
            .find("name=\"mode\"\r\n\r\nextract\r\n")
            .expect("mode field");
        let track = text
            .find("name=\"track\"\r\n\r\nguitar\r\n")
            .expect("track field");
        let file = text
            .find("name=\"file\"; filename=\"mix.wav\"")
            .expect("file part");
        assert!(mode < track && track < file);
        assert!(text.ends_with("RIFF\r\n--BOUND--\r\n"));
    }

    #[test]
    fn generate_request_leaves_unset_choices_to_the_model() {
        let request = GenerateRequest {
            prompt: "slow bossa nova, nylon guitar".to_owned(),
            lyrics: String::new(),
            duration: 90.0,
            bpm: None,
            key: Some("A Minor".to_owned()),
            time_signature: None,
            language: None,
            title: None,
        };
        let json = serde_json::to_value(&request).unwrap();
        assert_eq!(json["key"], "A Minor");
        assert!(json.get("bpm").is_none());
        assert!(json.get("timeSignature").is_none());
        assert_eq!(json["duration"], 90.0);
    }

    #[test]
    fn render_job_carries_its_result_path() {
        let json = br#"{"id":"r1","status":"done","stage":"render","percent":100.0,
            "message":"done","error":null,"projectId":null,"resultPath":"/tmp/renders/x.wav"}"#;
        let job: Job = serde_json::from_slice(json).unwrap();
        assert_eq!(
            job.result_path.as_deref(),
            Some(Path::new("/tmp/renders/x.wav"))
        );
        assert!(job.is_finished() && !job.is_failed());
    }

    #[test]
    fn multipart_body_frames_the_bytes_untouched() {
        // 0x0d 0x0a is a boundary in text and ordinary data in audio; the part
        // has to survive it byte for byte.
        let audio = [0xff_u8, 0xfb, 0x00, 0x0d, 0x0a, 0x2d, 0x2d];
        let body = multipart_body("BOUND", &[], "song.mp3", &audio);
        let text = String::from_utf8_lossy(&body);
        assert!(text.starts_with("--BOUND\r\nContent-Disposition: form-data; name=\"file\";"));
        assert!(text.contains("filename=\"song.mp3\""));
        assert!(body.ends_with(b"\r\n--BOUND--\r\n"));
        let start = body.len() - audio.len() - b"\r\n--BOUND--\r\n".len();
        assert_eq!(&body[start..start + audio.len()], &audio);
    }

    #[test]
    fn a_filename_cannot_break_out_of_its_header() {
        let body = multipart_body("BOUND", &[], "ev\"il\r\nX: y.mp3", b"a");
        let header = String::from_utf8_lossy(&body[..body.len() - 1]);
        assert!(header.contains("filename=\"ev_il__X: y.mp3\""));
        // One header block only: the part's blank line is the sole \r\n\r\n.
        assert_eq!(header.matches("\r\n\r\n").count(), 1);
    }

    #[test]
    fn non_http_sources_are_refused_before_any_request() {
        let client = WorkerClient::new();
        for hostile in ["file:///etc/passwd", "ftp://example.invalid", "not a url"] {
            assert!(client.submit_url(hostile).is_err());
        }
    }
}
