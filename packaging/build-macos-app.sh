#!/usr/bin/env bash
# Builds RustDAW into a double-clickable macOS .app bundle and installs it to
# /Applications, so it appears in Launchpad. Re-run after any code change — it
# always rebuilds the binary first, so the bundle never goes stale.
#
# Usage: packaging/build-macos-app.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

APP="/Applications/RustDAW.app"
SVG="packaging/io.rustdaw.RustDAW.svg"
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
VERSION="$(grep '^version' Cargo.toml | head -1 | sed 's/[^0-9.]//g')"
VERSION="${VERSION:-0.0.0}"

echo "==> Building release binary"
cargo build --release -p rustdaw

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "==> Rendering icon"
if [ -x "$CHROME" ]; then
  sed 's/<svg /<svg width="1024" height="1024" /' "$SVG" > "$WORK/icon.svg"
  "$CHROME" --headless --disable-gpu --force-device-scale-factor=1 \
    --default-background-color=00000000 --hide-scrollbars \
    --window-size=1024,1024 --screenshot="$WORK/icon_1024.png" \
    "file://$WORK/icon.svg" >/dev/null 2>&1
  ICONSET="$WORK/AppIcon.iconset"; mkdir -p "$ICONSET"
  for s in 16 32 128 256 512; do
    sips -z "$s" "$s" "$WORK/icon_1024.png" --out "$ICONSET/icon_${s}x${s}.png" >/dev/null
    d=$((s * 2))
    sips -z "$d" "$d" "$WORK/icon_1024.png" --out "$ICONSET/icon_${s}x${s}@2x.png" >/dev/null
  done
  iconutil -c icns "$ICONSET" -o "$WORK/AppIcon.icns"
else
  echo "    (Chrome not found; skipping custom icon)"
fi

echo "==> Assembling bundle at $APP"
# Leave any running instance alone until we have replaced the files.
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp target/release/rustdaw "$APP/Contents/MacOS/rustdaw"
[ -f "$WORK/AppIcon.icns" ] && cp "$WORK/AppIcon.icns" "$APP/Contents/Resources/AppIcon.icns"

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>RustDAW</string>
  <key>CFBundleDisplayName</key><string>RustDAW</string>
  <key>CFBundleIdentifier</key><string>io.rustdaw.RustDAW</string>
  <key>CFBundleVersion</key><string>${VERSION}</string>
  <key>CFBundleShortVersionString</key><string>${VERSION}</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleExecutable</key><string>rustdaw</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>LSMinimumSystemVersion</key><string>11.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSMicrophoneUsageDescription</key><string>RustDAW needs the microphone to record audio input.</string>
</dict>
</plist>
PLIST

echo "==> Signing and registering"
xattr -cr "$APP" 2>/dev/null || true
codesign --force --deep --sign - "$APP"
/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister -f "$APP"

echo "==> Done. Launch with: open -a RustDAW"
