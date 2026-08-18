//! Importing a song into `RustDAW` as separated instrument tracks.
//!
//! A `YouTube` link, an audio file on this machine, or an already-processed
//! project becomes a session with one stereo track per instrument, at the
//! session's sample rate, with the tempo and meter the pipeline detected. You
//! can then arm a track and play along.
//!
//! The separation itself is not done here and cannot be: it is Demucs on the
//! GPU, driven by the DSPRO Studio worker that is already installed on this
//! machine. `RustDAW` drives that worker over loopback HTTP and takes ownership
//! of the audio afterwards. Nothing is uploaded anywhere.

pub mod client;
pub mod ingest;
pub mod manifest;
pub mod supervisor;

use anyhow::{Context, Result, bail};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

pub use client::{Job, ProjectSummary, WorkerClient};
pub use ingest::{IngestOptions, Ingested, MAX_TRANSPOSE_SEMITONES};
pub use manifest::SongManifest;

/// How long to wait for a worker we started to come up. Importing torch and
/// opening CUDA on a cold start is slow; this is deliberately patient.
const WORKER_START_TIMEOUT: Duration = Duration::from_secs(180);
const POLL_INTERVAL: Duration = Duration::from_secs(1);

#[derive(Clone, Debug)]
pub enum ImportSource {
    /// Run the full pipeline on a link.
    Url(String),
    /// Run the full pipeline on an audio file already on this machine. Same
    /// pipeline as [`Self::Url`] with the download replaced by a copy, so any
    /// format ffmpeg decodes works.
    LocalFile(PathBuf),
    /// Reuse a project the pipeline has already finished.
    ExistingProject(String),
}

/// Progress updates for the UI. Deliberately coarse: the caller polls these
/// from an `egui` frame, and the pipeline's own stages are the only timing
/// information worth showing.
#[derive(Clone, Debug)]
pub enum ImportProgress {
    /// A step with no measurable fraction, such as starting the worker.
    Status(String),
    /// Pipeline progress. `percent` is 0–100.
    Pipeline {
        stage: String,
        percent: f32,
        message: String,
    },
    /// Local conversion progress. `fraction` is 0.0–1.0.
    Converting { fraction: f32, stem: String },
}

/// Stops an import from waiting any longer.
///
/// Cancelling stops `RustDAW` waiting; it does not stop the worker, which owns
/// the GPU job and finishes it regardless. The project simply appears in the
/// existing-projects list when it is done.
#[derive(Clone, Debug, Default)]
pub struct CancelFlag(Arc<AtomicBool>);

impl CancelFlag {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }

    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

/// Runs a complete import and returns the saved session.
///
/// Blocking, and slow by nature — a full pipeline run is several minutes on an
/// RTX 3060. Call it from a background thread.
///
/// # Errors
///
/// Returns an error if the worker is missing or unreachable, the pipeline
/// fails, the import is cancelled, or conversion cannot complete.
pub fn run_import(
    source: &ImportSource,
    options: &IngestOptions,
    target_rate: u32,
    cancel: &CancelFlag,
    mut on_progress: impl FnMut(ImportProgress),
) -> Result<Ingested> {
    let client = WorkerClient::new();
    on_progress(ImportProgress::Status(
        "Contacting the song pipeline…".to_owned(),
    ));
    ensure_worker(&client, &mut on_progress)?;

    let project_id = match source {
        ImportSource::ExistingProject(id) => id.clone(),
        ImportSource::Url(url) => {
            let job = client.submit_url(url)?;
            run_pipeline(&client, &job.id, cancel, &mut on_progress)?
        }
        ImportSource::LocalFile(path) => {
            if !path.is_file() {
                bail!("{} is not a file", path.display());
            }
            on_progress(ImportProgress::Status(
                "Sending the file to the song pipeline…".to_owned(),
            ));
            let job = client.submit_file(path)?;
            run_pipeline(&client, &job.id, cancel, &mut on_progress)?
        }
    };

    let project_dir = supervisor::project_dir(&project_id)?;
    if !project_dir.is_dir() {
        bail!("the pipeline reported project {project_id}, but its folder is missing");
    }

    on_progress(ImportProgress::Status(
        "Converting stems to the session's sample rate…".to_owned(),
    ));
    ingest::ingest_project(&project_dir, options, target_rate, |fraction, stem| {
        on_progress(ImportProgress::Converting {
            fraction,
            stem: stem.to_owned(),
        });
    })
}

/// Lists projects the pipeline has already finished, for the import picker.
///
/// # Errors
///
/// Returns an error if the worker cannot be reached or started.
pub fn list_ready_projects() -> Result<Vec<ProjectSummary>> {
    let client = WorkerClient::new();
    ensure_worker(&client, &mut |_| {})?;
    Ok(client
        .projects()
        .unwrap_or_default()
        .into_iter()
        .filter(|project| project.has_stems)
        .collect())
}

fn ensure_worker(
    client: &WorkerClient,
    on_progress: &mut impl FnMut(ImportProgress),
) -> Result<()> {
    if client.is_healthy() {
        return Ok(());
    }
    if !supervisor::is_installed() {
        bail!(
            "The song-import worker is not installed. Install it once with \
             `crates/daw-songimport/worker/install.sh` (macOS and Ubuntu); it sets up a Python \
             environment with Demucs and basic-pitch under {}.",
            supervisor::data_dir().map_or_else(
                || "the worker data directory".to_owned(),
                |dir| dir.display().to_string()
            )
        );
    }
    on_progress(ImportProgress::Status(
        "Starting the song pipeline (loading models, this can take a minute)…".to_owned(),
    ));
    supervisor::start_worker(WORKER_START_TIMEOUT, || client.is_healthy())
}

/// Waits for an already-submitted pipeline job and returns its project id.
fn run_pipeline(
    client: &WorkerClient,
    job_id: &str,
    cancel: &CancelFlag,
    on_progress: &mut impl FnMut(ImportProgress),
) -> Result<String> {
    loop {
        if cancel.is_cancelled() {
            bail!(
                "import cancelled; the pipeline keeps running and the song will appear under \
                 existing projects when it finishes"
            );
        }
        let job = client.job(job_id)?;
        on_progress(ImportProgress::Pipeline {
            stage: job.stage.clone(),
            percent: job.percent,
            message: job.message.clone(),
        });
        if job.is_failed() {
            bail!(
                "the pipeline failed: {}",
                job.error.unwrap_or_else(|| "unknown error".to_owned())
            );
        }
        if job.is_finished() {
            return job
                .project_id
                .context("the pipeline finished without producing a project");
        }
        std::thread::sleep(POLL_INTERVAL);
    }
}

/// Default location for imported songs: a `Songs` folder beside the
/// application's `Recordings` and `Sessions` directories, under the writable
/// media root (which falls back to `~/RustDAW` when the working directory is not
/// writable, as it is for a Finder-launched app).
#[must_use]
pub fn default_song_root() -> PathBuf {
    daw_core::media_dir("Songs")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cancelling_is_visible_to_the_waiting_thread() {
        let flag = CancelFlag::new();
        let clone = flag.clone();
        assert!(!clone.is_cancelled());
        flag.cancel();
        assert!(clone.is_cancelled());
    }

    #[test]
    fn song_root_sits_beside_the_other_media_folders() {
        assert!(default_song_root().ends_with("Songs"));
    }
}
