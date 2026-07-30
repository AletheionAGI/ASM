#!/bin/sh
set -eu

if [ "$#" -lt 1 ]; then
    echo "usage: $0 <script.ps1> [arguments...]" >&2
    exit 64
fi

script=$1
shift

if ! command -v pwsh >/dev/null 2>&1; then
    echo "error: pwsh (PowerShell 7+) is required to run $script" >&2
    echo "install PowerShell or run the corresponding Python entry point directly" >&2
    exit 127
fi

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH= cd -- "$script_dir/.." && pwd)
cd "$repo_root"
exec pwsh -NoLogo -NoProfile -NonInteractive -File "$script_dir/$script" "$@"
