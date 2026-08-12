//! Loading amp captures from TONE3000, the community library behind
//! [tone3000.com](https://www.tone3000.com/).
//!
//! The catalogue is theirs and is browsed on their site: RustDAW opens the
//! user's browser at TONE3000's own picker, waits for them to choose, and
//! downloads the one model they chose. That is the `select_tone` flow, and it
//! is what their free tier permits — bulk downloading, mirroring or bundling
//! the catalogue is not, which is why no captures ship with RustDAW.
//!
//! # No client secret
//!
//! A desktop application cannot keep one: anything compiled in can be read out
//! of the binary. This uses OAuth with PKCE and the *publishable* key only, so
//! there is nothing in the build worth extracting. TONE3000's secret key is
//! for servers and must never reach here.
//!
//! # Shape of the flow
//!
//! 1. Bind a loopback listener and take its port as the redirect URI.
//! 2. Open the browser at TONE3000's authorisation URL, carrying the PKCE
//!    challenge and a state value.
//! 3. The user signs in, browses, and picks a tone; TONE3000 redirects to the
//!    listener with a code.
//! 4. Redeem the code with the verifier for an access token.
//! 5. Ask which models belong to that tone, and download the first.

// "TONE3000" and "RustDAW" are names, not items.
#![allow(clippy::doc_markdown)]

pub mod pkce;
pub mod redirect;

use std::io::Read as _;
use std::time::Duration;

use serde::Deserialize;

pub use pkce::Pkce;
pub use redirect::{Callback, RedirectError, RedirectServer};

/// Where the service lives. Overridable for development, as their own client
/// does, but the default is production.
pub const DEFAULT_API: &str = "https://www.tone3000.com";

/// How long to wait for the user to finish choosing before giving up. Long
/// enough to sign in from a password manager and browse; short enough that a
/// forgotten tab does not park a thread for the session.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(300);

/// How long any one HTTP call may take.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);

/// A capture downloaded from TONE3000, in memory, ready to be written into the
/// amp library.
#[derive(Clone, Debug)]
pub struct FetchedModel {
    /// The tone's name on TONE3000, for display.
    pub name: String,
    /// A file name safe to write, taken from the name and the stored file's
    /// own extension.
    pub file_name: String,
    pub bytes: Vec<u8>,
}

#[derive(Debug)]
pub enum Error {
    /// No publishable key was compiled in or configured.
    NotConfigured,
    Redirect(RedirectError),
    Random(std::io::Error),
    /// The service answered, but not with what was asked for.
    Api { status: u16, detail: String },
    Transport(String),
    /// The chosen tone has no model file behind it.
    NoModel,
}

impl std::fmt::Display for Error {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotConfigured => write!(
                formatter,
                "no TONE3000 publishable key is configured; set {PUBLISHABLE_KEY_ENV}"
            ),
            Self::Redirect(error) => write!(formatter, "{error}"),
            Self::Random(error) => write!(formatter, "could not read system entropy: {error}"),
            Self::Api { status, detail } => {
                write!(formatter, "TONE3000 replied {status}: {detail}")
            }
            Self::Transport(detail) => write!(formatter, "could not reach TONE3000: {detail}"),
            Self::NoModel => write!(formatter, "that tone has no downloadable model"),
        }
    }
}

impl std::error::Error for Error {}

impl From<RedirectError> for Error {
    fn from(error: RedirectError) -> Self {
        Self::Redirect(error)
    }
}

/// The environment variable holding the publishable key, read at build time so
/// a package carries it, and at run time so it can be overridden.
pub const PUBLISHABLE_KEY_ENV: &str = "TONE3000_PUBLISHABLE_KEY";
/// Overrides the loopback port, which must match the redirect URI registered
/// with TONE3000. Zero lets the operating system choose, which only works if a
/// wildcard loopback redirect is registered.
pub const REDIRECT_PORT_ENV: &str = "TONE3000_REDIRECT_PORT";
/// The port TONE3000's own example client registers.
pub const DEFAULT_REDIRECT_PORT: u16 = 3_001;

/// The publishable key this build will use, if any.
///
/// Compiled in when the variable is set at build time, overridden by the
/// environment at run time. The publishable key is designed to be public —
/// it is the OAuth `client_id` — so carrying it in the binary is intended.
#[must_use]
pub fn publishable_key() -> Option<String> {
    if let Ok(key) = std::env::var(PUBLISHABLE_KEY_ENV) {
        if !key.trim().is_empty() {
            return Some(key);
        }
    }
    option_env!("TONE3000_PUBLISHABLE_KEY")
        .map(str::trim)
        .filter(|key| !key.is_empty())
        .map(str::to_owned)
}

/// The loopback port to listen on for the redirect.
#[must_use]
pub fn redirect_port() -> u16 {
    std::env::var(REDIRECT_PORT_ENV)
        .ok()
        .and_then(|value| value.trim().parse().ok())
        .unwrap_or(DEFAULT_REDIRECT_PORT)
}

/// A client for one account's worth of browsing and downloading.
pub struct Client {
    api: String,
    publishable_key: String,
    port: u16,
    timeout: Duration,
}

impl Client {
    /// Builds a client from the configured key.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotConfigured`] when no publishable key is available.
    pub fn from_env() -> Result<Self, Error> {
        Ok(Self {
            api: std::env::var("TONE3000_API").unwrap_or_else(|_| DEFAULT_API.to_owned()),
            publishable_key: publishable_key().ok_or(Error::NotConfigured)?,
            port: redirect_port(),
            timeout: DEFAULT_TIMEOUT,
        })
    }

    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// The authorisation URL for one attempt, given where to be redirected.
    ///
    /// Separated from the flow so it can be checked without a browser.
    #[must_use]
    pub fn authorize_url(&self, redirect_uri: &str, pkce: &Pkce) -> String {
        let query = url::form_urlencoded::Serializer::new(String::new())
            .append_pair("client_id", &self.publishable_key)
            .append_pair("redirect_uri", redirect_uri)
            .append_pair("response_type", "code")
            .append_pair("code_challenge", &pkce.challenge)
            .append_pair("code_challenge_method", "S256")
            .append_pair("state", &pkce.state)
            // The free tier's flow: TONE3000 shows its own picker and returns
            // whichever tone the user chose.
            .append_pair("prompt", "select_tone")
            .finish();
        format!("{}/api/v1/oauth/authorize?{query}", self.api)
    }

    /// Runs the whole flow and returns the capture the user chose.
    ///
    /// `open` is handed the authorisation URL and is expected to put it in
    /// front of the user; it is a parameter so the caller decides how, and so
    /// this can be driven without a browser in a test.
    ///
    /// # Errors
    ///
    /// Returns an error when the port cannot be bound, the user does not
    /// finish in time, TONE3000 refuses, or the download fails.
    pub fn select_tone(&self, open: impl FnOnce(&str)) -> Result<FetchedModel, Error> {
        let server = RedirectServer::bind(self.port)?;
        let redirect_uri = server.redirect_uri();
        let pkce = Pkce::generate().map_err(Error::Random)?;

        open(&self.authorize_url(&redirect_uri, &pkce));

        let callback = server.wait(&pkce.state, self.timeout)?;
        let token = self.exchange(&callback.code, &pkce.verifier, &redirect_uri)?;
        let model = self.first_model(&token, callback.tone_id.as_deref())?;
        let bytes = self.download(&token, &model.model_url)?;
        let name = model.name.unwrap_or_else(|| "TONE3000 capture".to_owned());
        Ok(FetchedModel {
            file_name: file_name_for(&name, &model.model_url),
            name,
            bytes,
        })
    }

    /// Redeems the authorisation code, proving ownership with the verifier.
    fn exchange(
        &self,
        code: &str,
        verifier: &str,
        redirect_uri: &str,
    ) -> Result<String, Error> {
        let body = url::form_urlencoded::Serializer::new(String::new())
            .append_pair("grant_type", "authorization_code")
            .append_pair("code", code)
            .append_pair("code_verifier", verifier)
            .append_pair("redirect_uri", redirect_uri)
            .append_pair("client_id", &self.publishable_key)
            .finish();

        let response = ureq::post(format!("{}/api/v1/oauth/token", self.api))
            .config()
            .timeout_global(Some(REQUEST_TIMEOUT))
            .build()
            .header("Content-Type", "application/x-www-form-urlencoded")
            .send(&body);
        let token: TokenResponse = read_json(response)?;
        Ok(token.access_token)
    }

    /// The first model behind a tone, which is the one to download.
    fn first_model(&self, token: &str, tone_id: Option<&str>) -> Result<ModelRecord, Error> {
        let tone_id = tone_id.ok_or(Error::NoModel)?;
        let url = format!("{}/api/v1/models?tone_id={tone_id}", self.api);
        let response = ureq::get(&url)
            .config()
            .timeout_global(Some(REQUEST_TIMEOUT))
            .build()
            .header("Authorization", format!("Bearer {token}"))
            .call();
        let listing: ModelListing = read_json(response)?;
        listing.data.into_iter().next().ok_or(Error::NoModel)
    }

    /// Fetches the model file itself.
    ///
    /// The `model_url` is on TONE3000 and needs the same bearer token; their
    /// own client notes it cannot be fetched unauthenticated.
    fn download(&self, token: &str, model_url: &str) -> Result<Vec<u8>, Error> {
        let url = if model_url.starts_with("http") {
            model_url.to_owned()
        } else {
            format!("{}{model_url}", self.api)
        };
        let response = ureq::get(&url)
            .config()
            .timeout_global(Some(REQUEST_TIMEOUT))
            .build()
            .header("Authorization", format!("Bearer {token}"))
            .call();
        let mut response = match response {
            Ok(response) => response,
            Err(ureq::Error::StatusCode(status)) => {
                return Err(Error::Api {
                    status,
                    detail: "the model could not be downloaded".to_owned(),
                });
            }
            Err(error) => return Err(Error::Transport(error.to_string())),
        };
        let mut bytes = Vec::new();
        response
            .body_mut()
            .as_reader()
            .read_to_end(&mut bytes)
            .map_err(|error| Error::Transport(error.to_string()))?;
        Ok(bytes)
    }
}

#[derive(Deserialize)]
struct TokenResponse {
    access_token: String,
}

#[derive(Deserialize)]
struct ModelListing {
    data: Vec<ModelRecord>,
}

#[derive(Clone, Debug, Deserialize)]
struct ModelRecord {
    model_url: String,
    #[serde(default)]
    name: Option<String>,
}

/// Reads a JSON body, turning any failure into an [`Error`].
fn read_json<T: serde::de::DeserializeOwned>(
    response: Result<ureq::http::Response<ureq::Body>, ureq::Error>,
) -> Result<T, Error> {
    match response {
        Ok(mut response) => response
            .body_mut()
            .read_json::<T>()
            .map_err(|error| Error::Transport(error.to_string())),
        Err(ureq::Error::StatusCode(status)) => Err(Error::Api {
            status,
            detail: "the request was refused".to_owned(),
        }),
        Err(error) => Err(Error::Transport(error.to_string())),
    }
}

/// A file name for a downloaded capture: the tone's name, reduced to something
/// safe, with the extension the stored file actually has.
#[must_use]
pub fn file_name_for(name: &str, model_url: &str) -> String {
    let extension = model_url
        .rsplit('?')
        .next_back()
        .and_then(|path| path.rsplit('/').next())
        .and_then(|file| file.rsplit_once('.').map(|(_, extension)| extension))
        .filter(|extension| {
            !extension.is_empty()
                && extension.len() <= 8
                && extension.chars().all(|c| c.is_ascii_alphanumeric())
        })
        .unwrap_or("nam");

    let mut stem = String::new();
    let mut last_was_dash = true;
    for character in name.chars() {
        if character.is_ascii_alphanumeric() {
            stem.push(character.to_ascii_lowercase());
            last_was_dash = false;
        } else if !last_was_dash {
            stem.push('-');
            last_was_dash = true;
        }
    }
    let stem = stem.trim_matches('-');
    let stem = if stem.is_empty() { "tone3000" } else { stem };
    format!("{stem}.{extension}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn client() -> Client {
        Client {
            api: "https://example.test".to_owned(),
            publishable_key: "t3k_pk_example".to_owned(),
            port: 0,
            timeout: Duration::from_millis(200),
        }
    }

    #[test]
    fn the_authorisation_url_carries_everything_the_service_needs() {
        let pkce = Pkce::generate().expect("entropy");
        let url = client().authorize_url("http://localhost:3001", &pkce);
        assert!(url.starts_with("https://example.test/api/v1/oauth/authorize?"));
        for expected in [
            "client_id=t3k_pk_example",
            "response_type=code",
            "code_challenge_method=S256",
            "prompt=select_tone",
        ] {
            assert!(url.contains(expected), "{expected} missing from {url}");
        }
        assert!(url.contains(&format!("code_challenge={}", pkce.challenge)));
        assert!(url.contains(&format!("state={}", pkce.state)));
        // The redirect has to survive being put in a query string.
        assert!(url.contains("redirect_uri=http%3A%2F%2Flocalhost%3A3001"));
    }

    #[test]
    fn the_verifier_never_appears_in_the_authorisation_url() {
        // The whole point of PKCE: only the hash travels, and it travels alone.
        let pkce = Pkce::generate().expect("entropy");
        let url = client().authorize_url("http://localhost:3001", &pkce);
        assert!(
            !url.contains(&pkce.verifier),
            "the verifier leaked into the URL"
        );
    }

    #[test]
    fn a_missing_key_is_reported_rather_than_guessed_at() {
        // Only meaningful when the environment does not supply one; when a key
        // is configured for this machine the constructor should succeed.
        match Client::from_env() {
            Err(Error::NotConfigured) => {
                assert!(publishable_key().is_none());
            }
            Err(other) => panic!("unexpected error: {other}"),
            Ok(_) => assert!(publishable_key().is_some()),
        }
    }

    #[test]
    fn the_flow_gives_up_when_nobody_comes_back() {
        // `open` deliberately does nothing, standing in for a browser that was
        // never opened or a user who walked away.
        let opened = std::cell::Cell::new(false);
        let result = client().select_tone(|_| opened.set(true));
        assert!(opened.get(), "the URL should have been offered to the user");
        assert!(
            matches!(result, Err(Error::Redirect(RedirectError::TimedOut))),
            "{result:?}"
        );
    }

    #[test]
    fn downloaded_captures_are_named_after_the_tone() {
        assert_eq!(
            file_name_for("1966 Marshall 1962 Bluesbreaker", "https://x/y/abc123.nam"),
            "1966-marshall-1962-bluesbreaker.nam"
        );
        assert_eq!(file_name_for("Vox AC30/6", "https://x/y/z.nam"), "vox-ac30-6.nam");
    }

    #[test]
    fn a_file_name_is_always_usable() {
        // Names come from other people's uploads and can be anything at all.
        for (name, url) in [
            ("", "https://x/y/z.nam"),
            ("///", "https://x/y/z"),
            ("   ", "https://x/y/z.nam?token=abc"),
            ("Ünïcodé ✨", "https://x/y/z.NAM"),
        ] {
            let file = file_name_for(name, url);
            assert!(!file.starts_with('.'), "{file} has no stem");
            assert!(!file.contains('/'), "{file} contains a path separator");
            assert!(file.contains('.'), "{file} has no extension");
            assert!(
                file.chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '.'),
                "{file} is not a safe file name"
            );
        }
    }

    #[test]
    fn an_extension_is_taken_from_the_stored_file_when_it_has_one() {
        assert_eq!(file_name_for("Amp", "https://x/y/model.wavenet"), "amp.wavenet");
        // And falls back rather than inventing something unusable.
        assert_eq!(file_name_for("Amp", "https://x/y/model"), "amp.nam");
        assert_eq!(file_name_for("Amp", ""), "amp.nam");
    }
}
