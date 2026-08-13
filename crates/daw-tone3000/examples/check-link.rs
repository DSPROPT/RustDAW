//! Checks the TONE3000 link without signing in.
//!
//! ```text
//! cargo run -p daw-tone3000 --example check-link
//! ```
//!
//! Builds the authorisation URL this build would use and asks TONE3000 whether
//! it accepts it. A rejected client id or an unregistered redirect URI shows up
//! here, rather than as a browser tab that dead-ends.

fn main() {
    let Some(key) = daw_tone3000::publishable_key() else {
        println!("no publishable key in this build; GET AMPS will just open the site");
        return;
    };
    // Never print the key itself.
    println!("publishable key: configured ({} chars)", key.len());
    println!("redirect port:   {}", daw_tone3000::redirect_port());

    let client = match daw_tone3000::Client::from_env() {
        Ok(client) => client,
        Err(error) => {
            println!("not configured: {error}");
            return;
        }
    };
    let server = match daw_tone3000::RedirectServer::bind(daw_tone3000::redirect_port()) {
        Ok(server) => server,
        Err(error) => {
            println!("cannot listen for the redirect: {error}");
            return;
        }
    };
    let redirect_uri = server.redirect_uri();
    println!("redirect uri:    {redirect_uri}");

    let pkce = daw_tone3000::Pkce::generate().expect("entropy");
    let url = client.authorize_url(&redirect_uri, &pkce);
    println!(
        "\nauthorize endpoint: {}",
        url.split('?').next().unwrap_or(&url)
    );

    match ureq::get(&url).call() {
        Ok(response) => println!(
            "TONE3000 answered {} — the client id and redirect are accepted",
            response.status()
        ),
        Err(ureq::Error::StatusCode(status)) => println!(
            "TONE3000 answered {status} — the client id or redirect URI is probably not registered"
        ),
        Err(error) => println!("could not reach TONE3000: {error}"),
    }
}
