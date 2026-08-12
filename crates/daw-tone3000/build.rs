//! Bakes the TONE3000 publishable key into the build.
//!
//! The publishable key is the OAuth `client_id`. It is meant to be public —
//! it travels in every authorisation URL — so compiling it in is intended and
//! is what lets a packaged build work without the user configuring anything.
//!
//! The secret key is deliberately **not** read here, and must never be. It is
//! a server credential; anything compiled into a desktop binary can be read
//! straight back out of it, so a secret in this build would be a published
//! secret. PKCE is what removes the need for one.

use std::path::PathBuf;

/// The only variable this script will take from the environment file.
const PUBLISHABLE_KEY: &str = "TONE3000_PUBLISHABLE_KEY";

fn main() {
    println!("cargo:rerun-if-env-changed={PUBLISHABLE_KEY}");

    // An explicit environment variable wins over the file.
    if std::env::var(PUBLISHABLE_KEY).is_ok_and(|key| !key.trim().is_empty()) {
        return;
    }

    let env_file = PathBuf::from("../../.env");
    println!("cargo:rerun-if-changed={}", env_file.display());
    let Ok(contents) = std::fs::read_to_string(&env_file) else {
        return;
    };

    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((name, value)) = line.split_once('=') else {
            continue;
        };
        if name.trim() != PUBLISHABLE_KEY {
            continue;
        }
        let value = value.trim().trim_matches(['"', '\''].as_slice());
        if !value.is_empty() {
            println!("cargo:rustc-env={PUBLISHABLE_KEY}={value}");
        }
        return;
    }
}
