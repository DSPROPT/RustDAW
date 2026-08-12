//! Locating the DSPRO Studio worker and starting it on demand.
//!
//! `RustDAW` does not own the pipeline: the Python worker, its virtual
//! environment and its ~3 GB of model checkpoints already exist under
//! `~/.local/share/chords-extraction`. This module finds that installation and
//! makes sure the service is answering, without ever duplicating it.

use anyhow::{Context, Result, bail};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// Environment overrides, matching the names the worker itself honours so a
/// relocated installation stays consistent between the two applications.
const DATA_DIR_VARIABLE: &str = "CHORDS_STUDIO_DATA";
const PORT_VARIABLE: &str = "CHORDS_STUDIO_PORT";
/// `RustDAW`-specific: points straight at the `chords-studio-servers` script.
const LAUNCHER_VARIABLE: &str = "CHORDS_STUDIO_LAUNCHER";

const DEFAULT_PORT: u16 = 8765;

/// The folder name the worker's data lives under, inside whichever per-user data
/// directory the platform uses.
const DATA_DIR_NAME: &str = "chords-extraction";

/// Candidate locations for the launcher script, relative to `$HOME`, tried in
/// order. `.local/bin` is where the install script drops it on every platform;
/// the rest cover manual and macOS-conventional installs.
const LAUNCHER_CANDIDATES: [&str; 5] = [
    ".local/bin/chords-studio-servers",
    "Library/Application Support/chords-extraction/bin/chords-studio-servers",
    ".local/share/chords-extraction/bin/chords-studio-servers",
    "Documents/chords-extraction/bin/chords-studio-servers",
    "chords-extraction/bin/chords-studio-servers",
];

#[must_use]
pub fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

/// The per-user data directories the worker may live under, most preferred
/// first. macOS uses `~/Library/Application Support`; every platform also
/// accepts the XDG `~/.local/share` location so a single install layout works
/// on both, and an existing installation is always found wherever it sits.
fn data_dir_candidates(home: &Path) -> Vec<PathBuf> {
    // The XDG location is always accepted; macOS prefers Application Support and
    // so puts it first.
    #[allow(unused_mut)]
    let mut candidates = vec![home.join(".local/share").join(DATA_DIR_NAME)];
    #[cfg(target_os = "macos")]
    candidates.insert(0, home.join("Library/Application Support").join(DATA_DIR_NAME));
    candidates
}

/// The worker's data directory: projects, models, logs and the virtualenv.
///
/// An explicit `CHORDS_STUDIO_DATA` wins. Otherwise an existing installation is
/// preferred wherever it is found, falling back to the platform's conventional
/// location when nothing is installed yet.
#[must_use]
pub fn data_dir() -> Option<PathBuf> {
    if let Some(value) = std::env::var_os(DATA_DIR_VARIABLE).filter(|value| !value.is_empty()) {
        return Some(PathBuf::from(value));
    }
    let home = home_dir()?;
    let candidates = data_dir_candidates(&home);
    candidates
        .iter()
        .find(|path| path.is_dir())
        .cloned()
        .or_else(|| candidates.into_iter().next())
}

/// Where finished song projects live.
#[must_use]
pub fn projects_dir() -> Option<PathBuf> {
    data_dir().map(|dir| dir.join("projects"))
}

/// Resolves one project directory, rejecting anything that is not a plain
/// child name. Project identifiers arrive from HTTP responses, so they are
/// treated as untrusted input and are never allowed to escape the store.
///
/// # Errors
///
/// Returns an error if the data directory cannot be resolved or the
/// identifier is not a single safe path component.
pub fn project_dir(project_id: &str) -> Result<PathBuf> {
    let root = projects_dir().context("cannot resolve the DSPRO Studio projects directory")?;
    let mut components = Path::new(project_id).components();
    let (Some(std::path::Component::Normal(name)), None) = (components.next(), components.next())
    else {
        bail!("{project_id:?} is not a valid project identifier");
    };
    if name != std::ffi::OsStr::new(project_id) {
        bail!("{project_id:?} is not a valid project identifier");
    }
    Ok(root.join(name))
}

#[must_use]
pub fn port() -> u16 {
    std::env::var(PORT_VARIABLE)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(DEFAULT_PORT)
}

/// The worker always binds to loopback; nothing is ever sent off the machine.
#[must_use]
pub fn base_url() -> String {
    format!("http://127.0.0.1:{}", port())
}

/// Finds the script that starts the worker.
#[must_use]
pub fn launcher_path() -> Option<PathBuf> {
    if let Some(value) = std::env::var_os(LAUNCHER_VARIABLE).filter(|value| !value.is_empty()) {
        let path = PathBuf::from(value);
        return path.is_file().then_some(path);
    }
    let home = home_dir()?;
    LAUNCHER_CANDIDATES
        .iter()
        .map(|candidate| home.join(candidate))
        .find(|path| path.is_file())
}

/// Whether a DSPRO Studio installation is present at all.
#[must_use]
pub fn is_installed() -> bool {
    data_dir().is_some_and(|dir| dir.join("venv").is_dir())
}

/// Starts the worker and waits for it to answer.
///
/// Only called after a health check has already failed. The launcher starts
/// the Next.js UI as well; that is harmless here and keeps a single supported
/// startup path rather than a `RustDAW`-only variant that could drift.
///
/// # Errors
///
/// Returns an error if no launcher can be found, it cannot be spawned, or the
/// worker does not answer within `timeout`.
pub fn start_worker(timeout: Duration, is_healthy: impl Fn() -> bool) -> Result<()> {
    let launcher = launcher_path().ok_or_else(|| {
        anyhow::anyhow!(
            "DSPRO Studio's launcher was not found. Set {LAUNCHER_VARIABLE} to the full path of \
             bin/chords-studio-servers, or start it yourself before importing."
        )
    })?;
    Command::new(&launcher)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .with_context(|| format!("failed to start {}", launcher.display()))?;

    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if is_healthy() {
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(500));
    }
    bail!(
        "started {} but the worker did not answer on {} within {} s",
        launcher.display(),
        base_url(),
        timeout.as_secs()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_url_is_always_loopback() {
        assert!(base_url().starts_with("http://127.0.0.1:"));
    }

    #[test]
    fn data_dir_lands_in_the_worker_folder_on_every_platform() {
        let home = Path::new("/home/tester");
        let candidates = data_dir_candidates(home);
        assert!(!candidates.is_empty());
        assert!(
            candidates.iter().all(|path| path.ends_with(DATA_DIR_NAME)),
            "every candidate must be the chords-extraction folder"
        );
        // macOS prefers Application Support; the XDG path is always accepted too.
        if cfg!(target_os = "macos") {
            assert!(candidates[0].starts_with("/home/tester/Library/Application Support"));
        }
        assert!(
            candidates
                .iter()
                .any(|path| path.starts_with("/home/tester/.local/share")),
            "the XDG location must always be a candidate for a shared layout"
        );
    }

    #[test]
    fn project_identifiers_cannot_escape_the_store() {
        for hostile in [
            "../../etc",
            "..",
            "a/b",
            "/etc/passwd",
            "",
            ".",
            "./x",
            "a/",
        ] {
            assert!(
                project_dir(hostile).is_err(),
                "{hostile:?} should have been rejected"
            );
        }
    }

    #[test]
    fn ordinary_project_identifiers_resolve_under_the_store() {
        let Some(root) = projects_dir() else {
            return;
        };
        let resolved = project_dir("20260810-214917-untitled").unwrap();
        assert_eq!(resolved, root.join("20260810-214917-untitled"));
    }
}
