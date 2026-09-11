//! Prints what a control surface is sending, and what it would do.
//!
//! What a pedal actually transmits is the one thing its manual can be relied
//! on to get wrong, and every binding in the application depends on knowing
//! it. Run this, step on things, and read the answer off the screen:
//!
//! ```text
//! cargo run -p daw-control --example watch-midi
//! cargo run -p daw-control --example watch-midi -- SINCO
//! ```

use daw_control::{Bindings, Command, ControlSurface, input_ports, likely_foot_controller};

fn main() {
    let ports = input_ports();
    if ports.is_empty() {
        eprintln!("No MIDI inputs. Plug the pedal in, or pair it, and try again.");
        return;
    }

    let requested = std::env::args().nth(1);
    let found = requested.or_else(|| likely_foot_controller(&ports).cloned());
    let Some(port) = found else {
        eprintln!("No port looked like a foot controller. Pass one of these instead:");
        for name in &ports {
            eprintln!("  {name}");
        }
        return;
    };

    let surface = match ControlSurface::open(&port) {
        Ok(surface) => surface,
        Err(error) => {
            eprintln!("Could not open {port}: {error}");
            return;
        }
    };

    let bindings = Bindings::footswitch();
    println!(
        "Listening to {}. Step on it. Ctrl-C to stop.\n",
        surface.port()
    );
    loop {
        for message in surface.drain() {
            let bound = match bindings.resolve(message) {
                Some(Command::Press(action)) => action.label(),
                Some(Command::Move(action, position)) => {
                    format!("{} at {:.0}%", action.label(), position * 100.0)
                }
                None => "unbound".to_owned(),
            };
            println!("{:<24} {bound}", message.describe());
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}
