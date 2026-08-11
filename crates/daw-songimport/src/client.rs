//! HTTP client for the local DSPRO Studio worker.
//!
//! The worker binds to loopback only, so no TLS is involved and no audio ever
//! leaves the machine. Job control goes over HTTP; the produced audio is read
//! straight off the filesystem by [`crate::ingest`], because both applications
//! share one project store and copying 300 MB of stems through localhost to
//! land in the same place would be pure waste.

use anyhow::{Context, Result, bail};
use std::time::Duration;

use serde::Deserialize;
use ureq::Agent;

use crate::supervisor;

/// Generous enough for a cold worker that is importing torch, short enough
/// that a wedged service surfaces as an error instead of a frozen import.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

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
    fn non_http_sources_are_refused_before_any_request() {
        let client = WorkerClient::new();
        for hostile in ["file:///etc/passwd", "ftp://example.invalid", "not a url"] {
            assert!(client.submit_url(hostile).is_err());
        }
    }
}
