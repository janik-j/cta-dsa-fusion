#!/usr/bin/env bash

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

source "$repo_root/scripts/cuda_env.sh"

UV_CACHE_DIR="${UV_CACHE_DIR:-$repo_root/.uv-cache}"
CC="${CC:-gcc}"
CXX="${CXX:-g++}"

usage() {
    cat <<'TXT'
Usage:
  ./scripts/uv.sh sync [--verify] [uv sync args...]
  ./scripts/uv.sh verify [verify_env args...]
  ./scripts/uv.sh run <command> [args...]
  ./scripts/uv.sh preflight

Commands:
  sync       Configure CUDA and run `uv sync` (defaults to `--frozen`)
  verify     Run `python -m utils.verify_env` in the configured env (defaults to `--full`)
  run        Run any `uv run ...` command with the same CUDA runtime setup
  preflight  Run sync and full verification
TXT
}

die() {
    printf '%s\n' "$1" >&2
    exit 1
}

append_uv_preview_feature() {
    local feature="$1"
    local current="${UV_PREVIEW_FEATURES:-}"

    case ",$current," in
        *",$feature,"*) ;;
        *) export UV_PREVIEW_FEATURES="${current:+$current,}$feature" ;;
    esac
}

configure_cuda() {
    command -v uv >/dev/null 2>&1 || die "uv not found on PATH"
    cuda_env_setup || exit 1

    append_uv_preview_feature extra-build-dependencies

    export CUDA_HOME CUDA_MATH_ROOT UV_CACHE_DIR CC CXX
    cuda_env_print_status
    printf 'Using UV_CACHE_DIR=%s\n' "$UV_CACHE_DIR"
}

cmd_sync() {
    local verify_after=0
    local -a sync_args=()

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --verify)
                verify_after=1
                shift
                ;;
            *)
                sync_args+=("$1")
                shift
                ;;
        esac
    done

    [[ ${#sync_args[@]} -gt 0 ]] || sync_args=(--frozen)

    configure_cuda
    uv sync "${sync_args[@]}"

    if [[ "$verify_after" -eq 1 ]]; then
        uv run --no-sync python -m utils.verify_env --full
    fi
}

cmd_verify() {
    local -a verify_args=("$@")
    [[ ${#verify_args[@]} -gt 0 ]] || verify_args=(--full)

    configure_cuda
    uv run python -m utils.verify_env "${verify_args[@]}"
}

cmd_run() {
    [[ $# -gt 0 ]] || die "Usage: ./scripts/uv.sh run <command> [args...]"
    configure_cuda
    uv run "$@"
}

cmd_preflight() {
    configure_cuda
    uv sync --frozen
    uv run --no-sync python -m utils.verify_env --full
}

command_name="${1:-help}"
if [[ $# -gt 0 ]]; then
    shift
fi

case "$command_name" in
    sync)
        cmd_sync "$@"
        ;;
    verify)
        cmd_verify "$@"
        ;;
    run)
        cmd_run "$@"
        ;;
    preflight)
        cmd_preflight
        ;;
    help|-h|--help)
        usage
        ;;
    *)
        usage >&2
        exit 1
        ;;
esac
