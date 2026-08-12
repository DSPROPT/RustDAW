//! Lists the amp captures RustDAW can see, and where it looked.
//!
//! ```text
//! cargo run -p daw-nam --example list-amps
//! ```

fn main() {
    println!("captures are kept in: {}", daw_nam::amp_dir().display());
    println!("\nsearched:");
    for path in daw_nam::search_paths() {
        let mark = if path.is_dir() { "found" } else { "     " };
        println!("  [{mark}] {}", path.display());
    }
    let models = daw_nam::discover();
    println!("\n{} capture(s):", models.len());
    for model in &models {
        println!("  {:<28} {}", model.name, model.path.display());
    }
    if models.is_empty() {
        println!("  (none — download .nam files from https://www.tone3000.com/)");
    }
}
