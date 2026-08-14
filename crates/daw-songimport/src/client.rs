//! HTTP client for the local DSPRO Studio worker.
//!
//! The worker binds to loopback only, so no TLS is involved and no audio ever
//! leaves the machine. Job control goes over HTTP; the produced audio is read
//! straight off the filesystem by [`crate::ingest`], because both applications
//! share one project store and copying 300 MB of stems through localhost to
//! land in the same place would be pure waste.

use anyhow::{Context, Result, bail};
use std::path::Path;
use std::time::Duration;

use serde::Deserialize;
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

/// One `multipart/form-data` part named `file`, which is what the worker's
/// upload endpoint reads. Written by hand because this is the only multipart
/// request `RustDAW` makes, and it is not worth a dependency.
fn multipart_file_body(boundary: &str, name: &str, bytes: &[u8]) -> Vec<u8> {
    // A quote or a newline in a filename would close the header early and split
    // the request into something the worker reads as a different upload.
    let name = name.replace(['"', '\r', '\n'], "_");
    let header = format!(
        "--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{name}\"\r\n\
         Content-Type: application/octet-stream\r\n\r\n"
    );
    let trailer = format!("\r\n--{boundary}--\r\n");
    let mut body = Vec::with_capacity(header.len() + bytes.len() + trailer.len());
    body.extend_from_slice(header.as_bytes());
    body.extend_from_slice(bytes);
    body.extend_from_slice(trailer.as_bytes());
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
        let bytes = std::fs::read(path)
            .with_context(|| format!("could not read {}", path.display()))?;
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
        let body = multipart_file_body(&boundary, name, &bytes);

        let agent: Agent = Agent::config_builder()
            .timeout_global(Some(UPLOAD_TIMEOUT))
            .build()
            .into();
        let mut response = agent
            .post(self.url("/api/jobs/upload"))
            .header(
                "Content-Type",
                &format!("multipart/form-data; boundary={boundary}"),
            )
            .send(&body[..])
            .context("failed to upload the file to the import worker")?;
        response
            .body_mut()
            .read_json::<Job>()
            .context("unexpected response from /api/jobs/upload")
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
    fn multipart_body_frames_the_bytes_untouched() {
        // 0x0d 0x0a is a boundary in text and ordinary data in audio; the part
        // has to survive it byte for byte.
        let audio = [0xff_u8, 0xfb, 0x00, 0x0d, 0x0a, 0x2d, 0x2d];
        let body = multipart_file_body("BOUND", "song.mp3", &audio);
        let text = String::from_utf8_lossy(&body);
        assert!(text.starts_with("--BOUND\r\nContent-Disposition: form-data; name=\"file\";"));
        assert!(text.contains("filename=\"song.mp3\""));
        assert!(body.ends_with(b"\r\n--BOUND--\r\n"));
        let start = body.len() - audio.len() - b"\r\n--BOUND--\r\n".len();
        assert_eq!(&body[start..start + audio.len()], &audio);
    }

    #[test]
    fn a_filename_cannot_break_out_of_its_header() {
        let body = multipart_file_body("BOUND", "ev\"il\r\nX: y.mp3", b"a");
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
