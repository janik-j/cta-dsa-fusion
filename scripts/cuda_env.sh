#!/usr/bin/env bash

cuda_required_release="11.8"
cuda_supported_os="Linux"
cuda_supported_arch="x86_64"

cuda_known_homes=(
    /software/u22/nvhpc/25.7/Linux_x86_64/25.7/cuda/11.8
    /software/r8/nvhpc/22.11/Linux_x86_64/2022/cuda/11.8
    /usr/local/cuda-11.8
    /usr/local/cuda
)

cuda_env_die() {
    printf '%s\n' "$1" >&2
    return 1
}

cuda_env_prepend_path() {
    local var_name="$1"
    local path="$2"
    local current="${!var_name-}"

    [[ -d "$path" ]] || return 0

    case ":$current:" in
        *":$path:"*) ;;
        *)
            if [[ -n "$current" ]]; then
                export "$var_name=$path:$current"
            else
                export "$var_name=$path"
            fi
            ;;
    esac
}

cuda_env_find_default_home() {
    local candidate

    for candidate in "${cuda_known_homes[@]}"; do
        if [[ -x "$candidate/bin/nvcc" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    if command -v nvcc >/dev/null 2>&1; then
        candidate=$(command -v nvcc)
        candidate=$(readlink -f "$candidate" 2>/dev/null || printf '%s\n' "$candidate")
        dirname "$(dirname "$candidate")"
        return 0
    fi

    return 1
}

cuda_env_detect_math_root() {
    local cuda_home="$1"
    local base
    local candidate

    for base in \
        "$cuda_home" \
        "$(dirname "$cuda_home")" \
        "$(dirname "$(dirname "$cuda_home")")"; do
        candidate="$base/math_libs/$cuda_required_release/targets/x86_64-linux"
        if [[ -d "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    return 1
}

cuda_env_parse_nvcc_release() {
    local version_output="$1"
    sed -nE 's/.*release ([0-9]+\.[0-9]+).*/\1/p' <<<"$version_output" | head -n 1
}

cuda_env_setup() {
    local nvcc_release

    [[ "$(uname -s)" == "$cuda_supported_os" && "$(uname -m)" == "$cuda_supported_arch" ]] \
        || cuda_env_die "Unsupported platform: expected $cuda_supported_os $cuda_supported_arch, got $(uname -s) $(uname -m)" || return 1

    if [[ -n "${CUDA_HOME:-}" ]]; then
        [[ -d "$CUDA_HOME" ]] || cuda_env_die "Missing CUDA_HOME: $CUDA_HOME" || return 1
        [[ -d "$CUDA_HOME/bin" ]] || cuda_env_die "Missing CUDA_HOME/bin: $CUDA_HOME/bin" || return 1
    else
        if ! CUDA_HOME=$(cuda_env_find_default_home); then
            cat >&2 <<EOF
Unable to locate a CUDA toolkit with nvcc.

This repo builds CUDA extensions from source, so the supported install target is:
  - Linux x86_64
  - NVIDIA GPU
  - CUDA toolkit $cuda_required_release with nvcc available

What to do next:
  - Standard workstation: install CUDA $cuda_required_release and rerun ./scripts/uv.sh sync --verify
  - Module-based cluster: load your CUDA $cuda_required_release module, then rerun ./scripts/uv.sh sync --verify
  - Manual override:
      export CUDA_HOME=/path/to/cuda-$cuda_required_release
      export PATH="\$CUDA_HOME/bin:\$PATH"
      ./scripts/uv.sh sync --verify

If your toolkit splits math headers and libraries (common on NVHPC), also set:
  export CUDA_MATH_ROOT=/path/to/math_libs/$cuda_required_release/targets/x86_64-linux
EOF
            return 1
        fi
    fi

    cuda_env_prepend_path PATH "$CUDA_HOME/bin"
    cuda_env_prepend_path CPATH "$CUDA_HOME/include"
    cuda_env_prepend_path LIBRARY_PATH "$CUDA_HOME/lib64"
    cuda_env_prepend_path LD_LIBRARY_PATH "$CUDA_HOME/lib64"
    cuda_env_prepend_path LIBRARY_PATH "$CUDA_HOME/lib"
    cuda_env_prepend_path LD_LIBRARY_PATH "$CUDA_HOME/lib"

    if [[ -z "${CUDA_MATH_ROOT:-}" ]]; then
        CUDA_MATH_ROOT=$(cuda_env_detect_math_root "$CUDA_HOME" || true)
    fi

    if [[ -n "${CUDA_MATH_ROOT:-}" ]]; then
        cuda_env_prepend_path CPATH "$CUDA_MATH_ROOT/include"
        cuda_env_prepend_path LIBRARY_PATH "$CUDA_MATH_ROOT/lib"
        cuda_env_prepend_path LD_LIBRARY_PATH "$CUDA_MATH_ROOT/lib"
    fi

    command -v nvcc >/dev/null 2>&1 || cuda_env_die "nvcc not found on PATH after configuring CUDA_HOME=$CUDA_HOME" || return 1

    nvcc_release=$(cuda_env_parse_nvcc_release "$(nvcc --version 2>&1)")
    [[ -n "$nvcc_release" ]] || cuda_env_die 'Unable to parse `nvcc --version` output' || return 1

    if [[ "$nvcc_release" != "$cuda_required_release" ]]; then
        cat >&2 <<EOF
Detected nvcc release $nvcc_release, but this repo is pinned to torch==2.1.2+cu118.

Use a CUDA $cuda_required_release toolkit for reproducible source builds, or update the
repo's pinned torch/CUDA stack and lockfile together.
EOF
        return 1
    fi

    export CUDA_HOME CUDA_MATH_ROOT
    export CUDACXX="${CUDACXX:-$CUDA_HOME/bin/nvcc}"
    return 0
}

cuda_env_print_status() {
    local nvcc_release
    nvcc_release=$(cuda_env_parse_nvcc_release "$(nvcc --version 2>&1)")
    printf 'Using CUDA_HOME=%s\n' "$CUDA_HOME"
    printf 'Using CUDA_MATH_ROOT=%s\n' "${CUDA_MATH_ROOT:-<none>}"
    printf 'Using nvcc release=%s\n' "$nvcc_release"
}

cuda_env_print_exports() {
    printf 'export CUDA_HOME=%q\n' "$CUDA_HOME"
    printf 'export PATH=%q\n' "$PATH"
    printf 'export CPATH=%q\n' "${CPATH:-}"
    printf 'export LIBRARY_PATH=%q\n' "${LIBRARY_PATH:-}"
    printf 'export LD_LIBRARY_PATH=%q\n' "${LD_LIBRARY_PATH:-}"
    printf 'export CUDACXX=%q\n' "${CUDACXX:-$CUDA_HOME/bin/nvcc}"
    if [[ -n "${CUDA_MATH_ROOT:-}" ]]; then
        printf 'export CUDA_MATH_ROOT=%q\n' "$CUDA_MATH_ROOT"
    fi
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    set -euo pipefail
    case "${1:-}" in
        --print)
            cuda_env_setup
            cuda_env_print_exports
            ;;
        --status)
            cuda_env_setup
            cuda_env_print_status
            ;;
        *)
            cat <<'EOF'
Usage:
  eval "$(./scripts/cuda_env.sh --print)"
  ./scripts/cuda_env.sh --status

This helper only configures CUDA-related environment variables.
For normal install and runtime commands, use `./scripts/uv.sh`.
EOF
            ;;
    esac
fi
