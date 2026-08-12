//! Proof Key for Code Exchange, which is what lets a desktop app use OAuth
//! without holding a client secret.
//!
//! A native application cannot keep a secret: it ships to the user's machine
//! and anything compiled into it can be read straight back out with `strings`.
//! PKCE replaces the secret with a value invented per authorisation — the
//! verifier — of which only a hash, the challenge, is sent up front. An
//! attacker who intercepts the redirect gets a code they cannot exchange,
//! because they do not have the verifier that hashes to the challenge.
//!
//! See RFC 7636. Only the `S256` method is implemented; `plain` exists in the
//! specification and defeats the point of it.

use base64::Engine as _;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use sha2::{Digest, Sha256};

/// Bytes of entropy behind a verifier and a state value.
///
/// RFC 7636 permits a verifier of 43 to 128 characters; 32 bytes encodes to
/// exactly 43, the shortest the specification allows and already far past
/// guessing.
const ENTROPY_BYTES: usize = 32;

/// One authorisation's secret and the challenge derived from it.
#[derive(Clone, Debug)]
pub struct Pkce {
    /// Kept locally and sent only when redeeming the code.
    pub verifier: String,
    /// Sent in the authorisation URL, where it is not a secret.
    pub challenge: String,
    /// Ties the redirect back to this request, so another tab's callback — or
    /// a forged one — is not accepted.
    pub state: String,
}

impl Pkce {
    /// Draws a fresh verifier, challenge and state.
    ///
    /// # Errors
    ///
    /// Returns an error when the system random source cannot be read, which is
    /// the only condition under which generating these would be unsafe.
    pub fn generate() -> Result<Self, std::io::Error> {
        let verifier = URL_SAFE_NO_PAD.encode(random_bytes()?);
        let state = URL_SAFE_NO_PAD.encode(random_bytes()?);
        let challenge = challenge_for(&verifier);
        Ok(Self {
            verifier,
            challenge,
            state,
        })
    }
}

/// The `S256` challenge for a verifier: base64url of its SHA-256, unpadded.
#[must_use]
pub fn challenge_for(verifier: &str) -> String {
    let digest = Sha256::digest(verifier.as_bytes());
    URL_SAFE_NO_PAD.encode(digest)
}

/// Reads entropy from the system.
///
/// `/dev/urandom` rather than a crate: this is Linux-first, the file is the
/// kernel's own CSPRNG, and it saves a dependency whose whole job would be to
/// read it.
fn random_bytes() -> Result<[u8; ENTROPY_BYTES], std::io::Error> {
    use std::io::Read as _;
    let mut bytes = [0_u8; ENTROPY_BYTES];
    std::fs::File::open("/dev/urandom")?.read_exact(&mut bytes)?;
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_challenge_matches_the_worked_example_in_rfc_7636() {
        // Appendix B of RFC 7636, which is the whole point of having a vector:
        // an implementation that agrees with it interoperates.
        let verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk";
        assert_eq!(
            challenge_for(verifier),
            "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"
        );
    }

    #[test]
    fn a_verifier_is_the_length_and_alphabet_the_specification_allows() {
        let pkce = Pkce::generate().expect("the system random source");
        assert_eq!(pkce.verifier.len(), 43, "{}", pkce.verifier);
        assert!((43..=128).contains(&pkce.verifier.len()));
        // The unreserved set: base64url produces a subset of it.
        assert!(
            pkce.verifier
                .chars()
                .all(|character| character.is_ascii_alphanumeric()
                    || matches!(character, '-' | '.' | '_' | '~')),
            "{} is not URL-safe",
            pkce.verifier
        );
        assert!(!pkce.verifier.contains('='), "padding must be stripped");
    }

    #[test]
    fn the_challenge_is_derived_from_the_verifier_it_was_made_with() {
        let pkce = Pkce::generate().expect("the system random source");
        assert_eq!(pkce.challenge, challenge_for(&pkce.verifier));
        assert_ne!(
            pkce.challenge, pkce.verifier,
            "the challenge must not be the verifier itself"
        );
    }

    #[test]
    fn every_authorisation_draws_fresh_values() {
        // Reusing a verifier across authorisations would let a code stolen from
        // one be redeemed against another.
        let first = Pkce::generate().expect("the system random source");
        let second = Pkce::generate().expect("the system random source");
        assert_ne!(first.verifier, second.verifier);
        assert_ne!(first.state, second.state);
        assert_ne!(first.challenge, second.challenge);
    }

    #[test]
    fn state_is_its_own_value_and_not_the_verifier() {
        // State travels in the URL; the verifier must never leave the process
        // until the code is redeemed.
        let pkce = Pkce::generate().expect("the system random source");
        assert_ne!(pkce.state, pkce.verifier);
        assert_ne!(pkce.state, pkce.challenge);
    }
}
