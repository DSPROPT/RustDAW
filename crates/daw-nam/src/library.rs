//! Finding the amp captures a user already has.
//!
//! A `.nam` file is one capture of one amp at one setting, so anybody using
//! this seriously ends up with dozens of them. Picking through a file dialog
//! every time is the wrong shape for that: captures belong in a folder the app
//! knows about, listed in a menu.
//!
//! Captures are kept in `Amps/` beside `Recordings/` and `Sessions/`, which is
//! also where anything downloaded from within the app is written. Several other
//! conventional locations are searched too, so a collection that predates
//! RustDAW is found where it already lives rather than having to be moved.

use std::path::{Path, PathBuf};

/// Overrides where captures are kept, for a collection on another disk.
pub const AMP_DIR_ENV: &str = "RUSTDAW_AMP_MODELS";

/// How far into a search directory to look. Captures downloaded as packs
/// arrive in a folder per amp, sometimes with a folder per cabinet inside it;
/// past that it is somebody's whole drive and not an amp library.
const MAX_DEPTH: usize = 3;
/// A ceiling on how many captures one scan will return, so a directory that
/// turns out to be enormous cannot stall the interface.
const MAX_MODELS: usize = 1_024;

/// One capture found on disk.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AmpModel {
    pub path: PathBuf,
    /// The file's own name without its extension, for display.
    pub name: String,
}

impl AmpModel {
    fn from_path(path: PathBuf) -> Option<Self> {
        let name = path.file_stem()?.to_string_lossy().into_owned();
        if name.is_empty() {
            return None;
        }
        Some(Self { path, name })
    }
}

/// Where captures are kept, and where downloads are written.
#[must_use]
pub fn amp_dir() -> PathBuf {
    if let Some(override_dir) = std::env::var_os(AMP_DIR_ENV).filter(|value| !value.is_empty()) {
        return PathBuf::from(override_dir);
    }
    daw_core::media_dir("Amps")
}

/// Every directory searched for captures, most important first.
#[must_use]
pub fn search_paths() -> Vec<PathBuf> {
    let mut paths = vec![amp_dir()];
    if let Some(home) = std::env::var_os("HOME")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
    {
        // Where a collection gathered before RustDAW is likely to already be.
        paths.push(home.join("Documents/NAM"));
        paths.push(home.join("NAM"));
        paths.push(home.join(".nam"));
        paths.push(home.join(".local/share/rustdaw/amps"));
    }
    paths
}

/// Every capture in the usual places, sorted by name.
#[must_use]
pub fn discover() -> Vec<AmpModel> {
    collect(&search_paths())
}

/// Every capture under `roots`, sorted by name and free of duplicates.
///
/// A root that does not exist is skipped rather than reported: most of them
/// will not exist on any given machine, which is not a problem to solve.
#[must_use]
pub fn collect(roots: &[PathBuf]) -> Vec<AmpModel> {
    let mut found = Vec::new();
    for root in roots {
        scan(root, 0, &mut found);
    }
    // The same capture can sit under two roots when one is a symlink to the
    // other, or when a search path nests inside another.
    found.sort_by(|left, right| {
        left.name
            .to_lowercase()
            .cmp(&right.name.to_lowercase())
            .then_with(|| left.path.cmp(&right.path))
    });
    found.dedup_by(|left, right| left.path == right.path);
    found
}

fn scan(directory: &Path, depth: usize, found: &mut Vec<AmpModel>) {
    if depth > MAX_DEPTH || found.len() >= MAX_MODELS {
        return;
    }
    let Ok(entries) = std::fs::read_dir(directory) else {
        return;
    };
    for entry in entries.flatten() {
        if found.len() >= MAX_MODELS {
            return;
        }
        let path = entry.path();
        // `file_type` rather than `path.is_dir`, so a symlinked directory is
        // not followed and a loop cannot be walked forever.
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        if file_type.is_dir() {
            scan(&path, depth + 1, found);
        } else if file_type.is_file()
            && path
                .extension()
                .is_some_and(|extension| extension.eq_ignore_ascii_case("nam"))
        {
            if let Some(model) = AmpModel::from_path(path) {
                found.push(model);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A throwaway directory tree, removed when the test finishes.
    struct TempTree(PathBuf);

    impl TempTree {
        fn new(label: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "rustdaw-amps-{label}-{}-{:?}",
                std::process::id(),
                std::thread::current().id()
            ));
            let _ = std::fs::remove_dir_all(&path);
            std::fs::create_dir_all(&path).expect("create the tree");
            Self(path)
        }

        fn write(&self, relative: &str) -> PathBuf {
            let path = self.0.join(relative);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).expect("create the parent");
            }
            std::fs::write(&path, b"not really a model").expect("write the file");
            path
        }
    }

    impl Drop for TempTree {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn captures_are_found_and_named_by_their_file() {
        let tree = TempTree::new("basic");
        tree.write("Marshall JCM800.nam");
        tree.write("notes.txt");
        let found = collect(std::slice::from_ref(&tree.0));
        assert_eq!(found.len(), 1, "found {found:?}");
        assert_eq!(found[0].name, "Marshall JCM800");
    }

    #[test]
    fn a_pack_downloaded_as_folders_is_still_found() {
        // Captures arrive as a folder per amp, sometimes nested again per cab.
        let tree = TempTree::new("nested");
        tree.write("Fender/Twin Reverb.nam");
        tree.write("Vox/AC30/Top Boost.nam");
        let found = collect(std::slice::from_ref(&tree.0));
        let names: Vec<&str> = found.iter().map(|model| model.name.as_str()).collect();
        assert_eq!(names, ["Top Boost", "Twin Reverb"]);
    }

    #[test]
    fn the_scan_stops_before_it_walks_the_whole_disk() {
        let tree = TempTree::new("deep");
        // One level past the limit, which must not be reached.
        let deep = "a/b/c/d/e/Too Deep.nam";
        tree.write(deep);
        tree.write("a/b/c/Just Deep Enough.nam");
        let found = collect(std::slice::from_ref(&tree.0));
        let names: Vec<&str> = found.iter().map(|model| model.name.as_str()).collect();
        assert_eq!(names, ["Just Deep Enough"]);
    }

    #[test]
    fn the_extension_is_matched_whatever_its_case() {
        let tree = TempTree::new("case");
        tree.write("Shouty.NAM");
        tree.write("Quiet.nam");
        assert_eq!(collect(std::slice::from_ref(&tree.0)).len(), 2);
    }

    #[test]
    fn results_are_sorted_and_free_of_duplicates() {
        let tree = TempTree::new("sorted");
        tree.write("zeta.nam");
        tree.write("Alpha.nam");
        tree.write("middle.nam");
        // The same root twice, as a nested search path would produce.
        let found = collect(&[tree.0.clone(), tree.0.clone()]);
        let names: Vec<&str> = found.iter().map(|model| model.name.as_str()).collect();
        assert_eq!(names, ["Alpha", "middle", "zeta"]);
    }

    #[test]
    fn a_missing_directory_is_skipped_rather_than_failing() {
        // Most search paths will not exist on any given machine.
        let found = collect(&[PathBuf::from("/nonexistent/amps"), PathBuf::from("")]);
        assert!(found.is_empty());
    }

    #[test]
    fn every_search_path_is_absolute_and_distinct() {
        let paths = search_paths();
        assert!(!paths.is_empty());
        for path in &paths {
            assert!(path.is_absolute(), "{} is not absolute", path.display());
        }
        let mut sorted = paths.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), paths.len(), "a search path is listed twice");
    }

    #[test]
    fn captures_are_kept_beside_the_other_media() {
        // Downloads land here, so it has to be the same place the scan reads.
        assert!(search_paths().contains(&amp_dir()));
        assert!(amp_dir().ends_with("Amps"));
    }
}
