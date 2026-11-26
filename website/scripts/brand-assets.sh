#!/bin/sh
# The PNGs made from the mark: public/favicon-32.png, public/apple-touch-icon.png,
# public/og.png and public/og-soon.png. Run it from website/ after changing public/hinny-mark.svg or scripts/og.html.
# Needs rsvg-convert (librsvg), magick (ImageMagick 7) and a Chromium (CHROMIUM=path to choose one).
set -eu
cd "$(dirname "$0")/.."
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
ink='#111111'
mark() { sed "s/currentColor/$ink/g" public/hinny-mark.svg | rsvg-convert -w "$1" -h "$1" -o "$2"; }

# The tab icon for browsers that don't take the SVG: the mark on a white tile, so it shows on a
# dark tab bar too.
mark 26 "$tmp/26.png"
magick -size 32x32 xc:none -fill white -draw 'roundrectangle 0,0 31,31 7,7' "$tmp/26.png" -gravity center -composite -depth 8 public/favicon-32.png

# The home-screen icon: opaque, the system rounds its corners.
mark 128 "$tmp/128.png"
magick -size 180x180 xc:white "$tmp/128.png" -gravity center -composite -depth 8 public/apple-touch-icon.png

# The share images: the site's, and the announcement's ("Coming soon"). The browser's profile
# is kept on disk beside the dependencies, not in the temporary directory (which may be memory),
# and removed afterwards.
profile="$PWD/node_modules/.cache/brand-assets-profile"
trap 'rm -rf "$tmp" "$profile"' EXIT
shot() {
  "${CHROMIUM:-chromium}" --headless --hide-scrollbars --force-device-scale-factor=1 --window-size=1200,630 \
    --user-data-dir="$profile" --screenshot="$PWD/public/$1" "file://$PWD/scripts/og.html$2" >/dev/null 2>&1
}
shot og.png ''
shot og-soon.png '?soon'
magick identify public/favicon-32.png public/apple-touch-icon.png public/og.png public/og-soon.png
