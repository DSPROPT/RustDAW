#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Single source of truth: the workspace version. Hardcoding it here and in the
# control file meant three places had to agree, and they drifted.
version="$(sed -n 's/^version = "\(.*\)"/\1/p' "$project_root/Cargo.toml" | head -1)"
if [[ -z "$version" ]]; then
    echo "could not read the version from Cargo.toml" >&2
    exit 1
fi
stage="$project_root/target/debian/rustdaw_${version}_amd64"
output="$project_root/dist/rustdaw_${version}_amd64.deb"

cargo build --manifest-path "$project_root/Cargo.toml" --release -p rustdaw

rm -rf -- "$stage"
install -d "$stage/DEBIAN"
install -d "$stage/usr/bin"
install -d "$stage/usr/share/applications"
install -d "$stage/usr/share/icons/hicolor/scalable/apps"
sed "s/^Version: .*/Version: $version/" \
    "$project_root/packaging/debian/DEBIAN/control" > "$stage/DEBIAN/control"
chmod 0644 "$stage/DEBIAN/control"
install -m 0755 "$project_root/target/release/rustdaw" "$stage/usr/bin/rustdaw"
install -m 0644 "$project_root/packaging/io.rustdaw.RustDAW.desktop" "$stage/usr/share/applications/io.rustdaw.RustDAW.desktop"
install -m 0644 "$project_root/packaging/io.rustdaw.RustDAW.svg" "$stage/usr/share/icons/hicolor/scalable/apps/io.rustdaw.RustDAW.svg"

install -d "$project_root/dist"
dpkg-deb --root-owner-group --build "$stage" "$output"
echo "Built $output"
