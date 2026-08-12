use std::path::{Path, PathBuf};

fn collect_cpp(directory: &Path, sources: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(directory).expect("read NAM source directory") {
        let path = entry.expect("read NAM source entry").path();
        if path.is_dir() {
            collect_cpp(&path, sources);
        } else if path.extension().is_some_and(|extension| extension == "cpp") {
            sources.push(path);
        }
    }
}

fn main() {
    let root = PathBuf::from("../../third_party/NeuralAmpModelerCore");
    let nam = root.join("NAM");
    if !nam.join("get_dsp.cpp").exists() {
        panic!("NeuralAmpModelerCore is missing; run `git submodule update --init --recursive`");
    }

    let mut sources = Vec::new();
    collect_cpp(&nam, &mut sources);
    sources.push(PathBuf::from("src/bridge.cpp"));

    cc::Build::new()
        .cpp(true)
        .std("c++20")
        .opt_level(3)
        .define("NAM_ENABLE_A2_FAST", None)
        .include(&root)
        .include(root.join("Dependencies/eigen"))
        .include(root.join("Dependencies/nlohmann"))
        .include(root.join("Dependencies/AudioDSPTools"))
        .files(sources)
        .warnings(false)
        .compile("rustdaw_nam");

    println!("cargo:rerun-if-changed=src/bridge.cpp");
    println!("cargo:rerun-if-changed={}", nam.display());
}
