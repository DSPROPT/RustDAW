//! A one-shot loopback server that catches the OAuth redirect.
//!
//! TONE3000 sends the user back to a `redirect_uri` after they pick a tone. A
//! web app receives that in the browser it already owns; a desktop app has no
//! browser, so it listens on `127.0.0.1` and registers that address as its
//! redirect. This is the loopback flow of RFC 8252 — the standard way native
//! applications do OAuth — and it is why no client secret is involved.
//!
//! Loopback only: binding the wildcard address would put the callback, which
//! carries a single-use authorisation code, on every interface on the machine.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read as _, Write};
use std::net::{Ipv4Addr, SocketAddr, TcpListener, TcpStream};
use std::time::{Duration, Instant};

/// What the browser is left showing once the code has been captured.
const SUCCESS_PAGE: &str = "<!doctype html><meta charset=utf-8>\
<title>RustDAW</title>\
<body style=\"font-family:system-ui;background:#111;color:#eee;text-align:center;padding:4rem\">\
<h1>Amp loaded</h1><p>You can close this tab and go back to RustDAW.</p>";

/// What it shows when the callback arrived without a code.
const FAILURE_PAGE: &str = "<!doctype html><meta charset=utf-8>\
<title>RustDAW</title>\
<body style=\"font-family:system-ui;background:#111;color:#eee;text-align:center;padding:4rem\">\
<h1>Nothing loaded</h1><p>RustDAW did not receive a tone. You can close this tab.</p>";

/// The longest a request line is read before giving up, so a client that opens
/// a connection and dribbles bytes cannot hold the listener open.
const MAX_REQUEST_BYTES: u64 = 8 * 1024;

#[derive(Debug)]
pub enum RedirectError {
    Bind {
        port: u16,
        source: std::io::Error,
    },
    /// Nobody arrived before the deadline — the user closed the tab, or never
    /// finished signing in.
    TimedOut,
    /// TONE3000 reported a problem instead of returning a code.
    Denied(String),
    /// The callback did not carry the state we sent, so it belongs to a
    /// different request and must not be trusted.
    StateMismatch,
    Missing(&'static str),
    Io(std::io::Error),
}

impl std::fmt::Display for RedirectError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bind { port, source } => write!(
                formatter,
                "could not listen on 127.0.0.1:{port} for the TONE3000 redirect: {source}"
            ),
            Self::TimedOut => write!(formatter, "timed out waiting for TONE3000"),
            Self::Denied(reason) => write!(formatter, "TONE3000 refused the request: {reason}"),
            Self::StateMismatch => {
                write!(
                    formatter,
                    "the reply did not match the request that was sent"
                )
            }
            Self::Missing(field) => write!(formatter, "the reply carried no {field}"),
            Self::Io(error) => write!(formatter, "{error}"),
        }
    }
}

impl std::error::Error for RedirectError {}

/// What the redirect carried back.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Callback {
    pub code: String,
    pub tone_id: Option<String>,
    pub model_id: Option<String>,
}

/// A bound loopback listener, waiting for one callback.
///
/// Bound before the browser is opened, so the port in the redirect URI is
/// known to be listening by the time the user finishes signing in.
pub struct RedirectServer {
    listener: TcpListener,
    address: SocketAddr,
}

impl RedirectServer {
    /// Binds `127.0.0.1:port`. Port `0` lets the operating system choose.
    ///
    /// # Errors
    ///
    /// Returns an error when the port is already in use.
    pub fn bind(port: u16) -> Result<Self, RedirectError> {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, port))
            .map_err(|source| RedirectError::Bind { port, source })?;
        let address = listener
            .local_addr()
            .map_err(|source| RedirectError::Bind { port, source })?;
        Ok(Self { listener, address })
    }

    /// The address to register as the `redirect_uri`.
    #[must_use]
    pub fn redirect_uri(&self) -> String {
        format!("http://localhost:{}", self.address.port())
    }

    #[must_use]
    pub const fn port(&self) -> u16 {
        self.address.port()
    }

    /// Waits for the browser to arrive, and checks the reply belongs to us.
    ///
    /// Connections that are not the callback — a stray probe, a favicon
    /// request — are answered and ignored rather than ending the wait.
    ///
    /// # Errors
    ///
    /// Returns an error when nothing arrives before `timeout`, when TONE3000
    /// reports a failure, or when the state does not match what was sent.
    pub fn wait(&self, expected_state: &str, timeout: Duration) -> Result<Callback, RedirectError> {
        let deadline = Instant::now() + timeout;
        self.listener
            .set_nonblocking(false)
            .map_err(RedirectError::Io)?;
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(RedirectError::TimedOut);
            }
            // A per-accept timeout, so a browser that never connects cannot
            // block the thread past the deadline.
            self.listener
                .set_nonblocking(true)
                .map_err(RedirectError::Io)?;
            let accepted = self.listener.accept();
            self.listener
                .set_nonblocking(false)
                .map_err(RedirectError::Io)?;
            let mut stream = match accepted {
                Ok((stream, _)) => stream,
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(50));
                    continue;
                }
                Err(error) => return Err(RedirectError::Io(error)),
            };
            stream
                .set_read_timeout(Some(Duration::from_secs(5)))
                .map_err(RedirectError::Io)?;

            let Some(target) = read_request_target(&stream) else {
                respond(&mut stream, FAILURE_PAGE);
                continue;
            };
            let query = parse_query(&target);
            if query.is_empty() {
                // The browser asking for something else on the same port.
                respond(&mut stream, FAILURE_PAGE);
                continue;
            }

            if let Some(error) = query.get("error") {
                respond(&mut stream, FAILURE_PAGE);
                return Err(RedirectError::Denied(error.clone()));
            }
            if query.get("state").map(String::as_str) != Some(expected_state) {
                respond(&mut stream, FAILURE_PAGE);
                return Err(RedirectError::StateMismatch);
            }
            let Some(code) = query.get("code").cloned() else {
                respond(&mut stream, FAILURE_PAGE);
                return Err(RedirectError::Missing("authorisation code"));
            };
            respond(&mut stream, SUCCESS_PAGE);
            return Ok(Callback {
                code,
                tone_id: query.get("tone_id").cloned(),
                model_id: query.get("model_id").cloned(),
            });
        }
    }
}

/// Pulls the request target out of the first line, `GET /path?query HTTP/1.1`.
fn read_request_target(stream: &TcpStream) -> Option<String> {
    let mut reader = BufReader::new(stream.take(MAX_REQUEST_BYTES));
    let mut line = String::new();
    reader.read_line(&mut line).ok()?;
    line.split_whitespace().nth(1).map(str::to_owned)
}

/// Decodes the query string of a request target.
fn parse_query(target: &str) -> HashMap<String, String> {
    let Some((_, query)) = target.split_once('?') else {
        return HashMap::new();
    };
    url::form_urlencoded::parse(query.as_bytes())
        .map(|(key, value)| (key.into_owned(), value.into_owned()))
        .collect()
}

fn respond(stream: &mut TcpStream, body: &str) {
    // Best effort: the code is already in hand, and a browser that has hung up
    // is not a reason to fail the flow.
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    let _ = stream.write_all(response.as_bytes());
    let _ = stream.flush();
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Drives the server from another thread, as the browser would.
    fn callback(path: &str, expected_state: &str) -> Result<Callback, RedirectError> {
        let server = RedirectServer::bind(0).expect("bind a loopback port");
        let port = server.port();
        let request = format!("GET {path} HTTP/1.1\r\nHost: localhost\r\n\r\n");
        let visitor = std::thread::spawn(move || {
            let mut stream = TcpStream::connect((Ipv4Addr::LOCALHOST, port))
                .expect("connect to the redirect server");
            stream.write_all(request.as_bytes()).expect("send");
            let mut reply = String::new();
            let _ = stream.read_to_string(&mut reply);
            reply
        });
        let result = server.wait(expected_state, Duration::from_secs(5));
        let _ = visitor.join();
        result
    }

    #[test]
    fn a_matching_callback_yields_its_code_and_tone() {
        let result = callback("/?code=abc123&state=xyz&tone_id=42&model_id=7", "xyz")
            .expect("the callback should be accepted");
        assert_eq!(result.code, "abc123");
        assert_eq!(result.tone_id.as_deref(), Some("42"));
        assert_eq!(result.model_id.as_deref(), Some("7"));
    }

    #[test]
    fn a_callback_for_another_request_is_refused() {
        // Without this check, any page the user visits could hand this listener
        // a code of its choosing.
        let result = callback("/?code=abc123&state=somebody-elses", "xyz");
        assert!(
            matches!(result, Err(RedirectError::StateMismatch)),
            "{result:?}"
        );
    }

    #[test]
    fn a_refusal_is_reported_with_its_reason() {
        let result = callback("/?error=access_denied&state=xyz", "xyz");
        match result {
            Err(RedirectError::Denied(reason)) => assert_eq!(reason, "access_denied"),
            other => panic!("expected a refusal, got {other:?}"),
        }
    }

    #[test]
    fn a_callback_without_a_code_is_refused() {
        let result = callback("/?state=xyz", "xyz");
        assert!(
            matches!(result, Err(RedirectError::Missing(_))),
            "{result:?}"
        );
    }

    #[test]
    fn percent_encoded_values_are_decoded() {
        let result = callback("/?code=a%2Bb%2Fc%3D&state=x%20y", "x y")
            .expect("the callback should be accepted");
        assert_eq!(result.code, "a+b/c=");
    }

    #[test]
    fn waiting_gives_up_rather_than_hanging_forever() {
        // The user closing the tab must not leave a thread parked for good.
        let server = RedirectServer::bind(0).expect("bind a loopback port");
        let started = Instant::now();
        let result = server.wait("xyz", Duration::from_millis(200));
        assert!(matches!(result, Err(RedirectError::TimedOut)), "{result:?}");
        assert!(started.elapsed() < Duration::from_secs(3));
    }

    #[test]
    fn the_redirect_uri_names_the_port_it_is_listening_on() {
        let server = RedirectServer::bind(0).expect("bind a loopback port");
        assert_eq!(
            server.redirect_uri(),
            format!("http://localhost:{}", server.port())
        );
        assert_ne!(
            server.port(),
            0,
            "the operating system should assign a port"
        );
    }

    #[test]
    fn it_listens_on_loopback_only() {
        // The callback carries a single-use code; it has no business being
        // reachable from the network.
        let server = RedirectServer::bind(0).expect("bind a loopback port");
        assert_eq!(server.address.ip(), Ipv4Addr::LOCALHOST);
    }
}
