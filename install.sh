#!/bin/bash

_cleanup_dirs=()

register_cleanup_dir() {
  _cleanup_dirs+=("$1")
}

cleanup_tmp_dirs() {
  local dir
  if [[ ${#_cleanup_dirs[@]} -eq 0 ]]; then
    return
  fi
  for dir in "${_cleanup_dirs[@]}"; do
    rm -rf "$dir"
  done
}

# Stable uses /releases/latest; dev selects the newest .dev tag.
fetch_release() {
  local repo_owner="$1"
  local repo_name="$2"
  local channel="$3"

  echo "Fetching ${channel} release..." >&2

  local api_url release_data
  if [[ "$channel" == "stable" ]]; then
    api_url="https://api.github.com/repos/${repo_owner}/${repo_name}/releases/latest"
  else
    api_url="https://api.github.com/repos/${repo_owner}/${repo_name}/releases?per_page=30"
  fi

  if ! release_data=$(curl -fsSL "$api_url"); then
    if [[ "$channel" == "stable" ]]; then
      error "Failed to fetch the latest stable release."
      echo "There may not be one yet. Retry with --dev for the latest development build." >&2
    else
      error "Failed to fetch release information."
      echo "Please check your internet connection and try again." >&2
    fi
    exit 1
  fi

  if [[ -z "$release_data" ]]; then
    error "No releases found for this repository."
    echo "Please visit https://github.com/${repo_owner}/${repo_name}/releases" >&2
    exit 1
  fi

  echo "$release_data"
}

# Print "<tag>\n<wheel url>" from a GitHub release payload on stdin.
extract_wheel_url() {
  local channel="$1"

  CHANNEL="$channel" python3 -c '
import json
import os
import sys

channel = os.environ["CHANNEL"]
try:
    data = json.load(sys.stdin)
except Exception:
    sys.exit(0)

# /releases/latest returns one release; /releases returns a list, newest first.
if isinstance(data, dict):
    releases = [data]
elif channel == "stable":
    releases = data
else:
    releases = [r for r in data if ".dev" in (r.get("tag_name") or "")]

for release in releases:
    for asset in release.get("assets", []):
        if (asset.get("name") or "").endswith(".whl"):
            print(release.get("tag_name", ""))
            print(asset.get("browser_download_url", ""))
            sys.exit(0)
'
}

fetch_release_vllm_tag() {
  local repo_owner="$1"
  local repo_name="$2"
  local release_tag="$3"
  local metadata_url legacy_url metadata legacy_install vllm_version

  metadata_url="https://raw.githubusercontent.com/${repo_owner}/${repo_name}/${release_tag}/.github/vllm-release-tag.commit"
  if metadata=$(curl -fsL "$metadata_url"); then
    metadata=$(printf '%s' "$metadata" | tr -d '[:space:]')
  else
    # Releases created before this metadata file kept the same pin in install.sh.
    legacy_url="https://raw.githubusercontent.com/${repo_owner}/${repo_name}/${release_tag}/install.sh"
    if ! legacy_install=$(curl -fsSL "$legacy_url"); then
      error "Release ${release_tag} does not declare its compatible vLLM release."
      return 1
    fi
    vllm_version=$(printf '%s\n' "$legacy_install" | sed -n 's/^VLLM_VERSION="\([^"]*\)"/\1/p' | head -n 1)
    metadata="v${vllm_version}"
  fi

  if ! validate_vllm_release_tag "$metadata"; then
    error "Release ${release_tag} has invalid vLLM metadata."
    return 1
  fi
  printf '%s\n' "$metadata"
}

install_vllm() {
  local vllm_release_tag="$1"
  local vllm_version="${vllm_release_tag#v}"
  local vllm_wheel_url="https://github.com/vllm-project/vllm/releases/download/${vllm_release_tag}/vllm-${vllm_version}%2Bcpu-cp312-cp312-macosx_11_0_arm64.whl"

  echo ""
  section "Installing vLLM core"
  echo "Wheel: vLLM ${vllm_version} (prebuilt macOS arm64)"

  if ! uv pip install "$vllm_wheel_url"; then
    error "Failed to install vLLM core from ${vllm_wheel_url}"
    echo "Please check your internet connection and try again." >&2
    exit 1
  fi

  success "Installed vLLM core"
}

install_mps_dependencies() {
  section "Building pinned Triton and Apple backend"
  # These are build tools, not runtime dependencies of the default MLX path.
  uv pip install setuptools wheel 'cmake>=3.20,<4' ninja \
    'nanobind==2.10.2' 'scikit-build-core>=0.10' requests

  local sources triton_url backend_url backend_repo backend_ref build_dir llvm_dir
  sources=$(python - <<'PY'
import tomllib
from packaging.requirements import Requirement

with open("pyproject.toml", "rb") as f:
    extra = tomllib.load(f)["project"]["optional-dependencies"]["mps"]
requirements = {r.name: r for r in map(Requirement, extra)}
for name in ("triton", "triton-apple-backend"):
    print(requirements[name].url)
PY
  )
  triton_url=$(printf '%s\n' "$sources" | sed -n '1p')
  backend_url=$(printf '%s\n' "$sources" | sed -n '2p')
  backend_repo="${backend_url#git+}"
  backend_repo="${backend_repo%@*}"
  backend_ref="${backend_url##*@}"
  backend_ref="${backend_ref%%#*}"
  build_dir=$(mktemp -d)
  register_cleanup_dir "$build_dir"
  git init --quiet "$build_dir/backend"
  git -C "$build_dir/backend" fetch --quiet --depth 1 "$backend_repo" "$backend_ref"
  git -C "$build_dir/backend" checkout --quiet --detach FETCH_HEAD

  # Upstream selects the pinned macOS LLVM SDK and verifies its checksum.
  llvm_dir=$(cd "$build_dir" && python backend/ci/download_llvm.py)
  (
    export LLVM_SYSPATH="$build_dir/$llvm_dir"
    export LLVM_INSTALL_DIR="$LLVM_SYSPATH"
    export TRITON_EXT_ENABLED=1 TRITON_BUILD_PROTON=OFF
    export MAX_JOBS="${MAX_JOBS:-8}"
    export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-8}"
    uv pip install --no-build-isolation "$triton_url"
    uv pip install --no-build-isolation "$backend_url"
    uv pip install --no-build-isolation -e '.[mps]'
  )
  # Temporary vLLM 0.30 workaround, confined to this MPS environment. Replace
  # the file atomically so uv's shared package cache cannot be modified.
  python - <<'PY'
from importlib.util import find_spec
from pathlib import Path
import sys

path = Path(find_spec("vllm").origin).parent / "triton_utils/importing.py"
if not path.resolve().is_relative_to(Path(sys.prefix).resolve()):
    raise RuntimeError("Refusing to patch vLLM outside the MPS environment")
source = path.read_text()
old = 'if "cpu" in version("vllm"):'
new = 'if "cpu" in version("vllm") and current_platform.is_cpu():'
if new not in source:
    if source.count(old) != 1:
        raise RuntimeError("vLLM Triton detection changed; review the MPS install workaround")
    temporary = path.with_suffix(".py.mps")
    temporary.write_text(source.replace(old, new))
    temporary.replace(path)
print("Applied the vLLM CPU-wheel Triton workaround in the MPS environment")
PY
  VLLM_METAL_BACKEND=mps python - <<'PY'
import torch
import triton
from vllm.triton_utils import HAS_TRITON
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample

if not HAS_TRITON or not torch.backends.mps.is_available():
    raise RuntimeError("Triton or PyTorch MPS is unavailable")
target = triton.runtime.driver.active.get_current_target()
if target.backend != "mps":
    raise RuntimeError(f"Expected the Triton Apple backend, got {target}")
logits = torch.arange(32, dtype=torch.float32, device="mps")[None, :]
mapping = torch.zeros(1, dtype=torch.int32, device="mps")
seed = torch.zeros(1, dtype=torch.int64, device="mps")
sampled = gumbel_sample(logits, mapping, torch.zeros(1, device="mps"),
                        seed, seed, apply_temperature=False, is_drafting=False)
if sampled.item() != 31:
    raise RuntimeError("Upstream Triton sampling verification failed")
print(f"Verified upstream vLLM sampling with Triton {triton.__version__} on {target}")
PY
}

download_and_install_wheel() {
  local wheel_url="$1"
  local package_name="$2"
  local release_tag="$3"

  local wheel_name
  wheel_name=$(basename "$wheel_url")
  echo "Release: ${release_tag}"
  echo "Wheel:   $wheel_name"
  success "Found release"

  local tmp_dir
  tmp_dir=$(mktemp -d)
  register_cleanup_dir "$tmp_dir"

  echo ""
  echo "Downloading wheel..."
  local wheel_path="$tmp_dir/$wheel_name"

  if ! curl -fsSL "$wheel_url" -o "$wheel_path"; then
    error "Failed to download wheel."
    exit 1
  fi

  success "Downloaded wheel"

  # Install vllm-metal package
  if ! uv pip install "$wheel_path"; then
    error "Failed to install ${package_name}."
    exit 1
  fi

  success "Installed ${package_name}"
}

main() {
  set -eu -o pipefail
  trap cleanup_tmp_dirs EXIT

  local repo_owner="vllm-project"
  local repo_name="vllm-metal"
  local package_name="vllm-metal"

  # Override the default dev channel with --stable or VLLM_METAL_CHANNEL.
  local channel="${VLLM_METAL_CHANNEL:-dev}"
  local mps=false

  for arg in "$@"; do
    case "$arg" in
      --mps)
        mps=true
        ;;
      --dev)
        channel="dev"
        ;;
      --stable)
        channel="stable"
        ;;
      -h|--help)
        cat <<'EOF'
Usage: install.sh [--dev | --stable] [--mps]

Options:
      --dev         Install the latest development build cut from main.
                    This is the default and the currently recommended channel.
      --stable      Install the latest tagged stable release. Stable releases
                    are cut by hand and may lag behind the dev channel.
      --mps         Build the pinned experimental Triton Apple backend from a
                    source checkout, in a separate .venv-vllm-metal-mps environment.
  -h, --help        Show this help.

The channel can also be set with VLLM_METAL_CHANNEL=dev|stable.
EOF
        exit 0
        ;;
      *)
        echo "Unknown argument: $arg" >&2
        echo "Run with --help for usage." >&2
        exit 1
        ;;
    esac
  done

  case "$channel" in
    dev|stable) ;;
    *)
      echo "Invalid channel: $channel (expected 'dev' or 'stable')." >&2
      exit 1
      ;;
  esac

  # Source shared library functions
  # Try local lib.sh first (when running ./install.sh), fall back to remote (when piped from curl)
  local local_lib=""
  local script_dir=""
  if [[ -n "${BASH_SOURCE[0]:-}" ]]; then
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]:-}")" && pwd)"
    local_lib="$script_dir/scripts/lib.sh"
  fi

  if [[ -n "$local_lib" && -f "$local_lib" ]]; then
    # shellcheck source=/dev/null
    source "$local_lib"
  else
    # Fetch from remote (curl | bash case)
    local lib_url="https://raw.githubusercontent.com/$repo_owner/$repo_name/main/scripts/lib.sh"
    local lib_tmp
    lib_tmp=$(mktemp)
    if ! curl -fsSL "$lib_url" -o "$lib_tmp"; then
      echo "Error: Failed to fetch lib.sh from $lib_url" >&2
      rm -f "$lib_tmp"
      exit 1
    fi
    # shellcheck source=/dev/null
    source "$lib_tmp"
    rm -f "$lib_tmp"
  fi

  if ! is_apple_silicon; then
    error "vllm-metal requires Apple Silicon arm64. Detected: $(uname -m)."
    exit 1
  fi

  if $mps && [[ -z "$local_lib" || ! -f "$local_lib" ]]; then
    error "Experimental MPS installation requires a source checkout."
    exit 1
  fi

  if ! ensure_uv; then
    exit 1
  fi

  local venv="$HOME/.venv-vllm-metal"
  if [[ -n "$local_lib" && -f "$local_lib" ]]; then
    # Source checkouts read the release tag file, run the editable install,
    # and build native artifacts relative to the repo root, so anchor the
    # working directory there instead of the caller's cwd.
    cd "$script_dir" || exit 1
    venv="$script_dir/.venv-vllm-metal"
    if $mps; then
      venv="${venv}-mps"
    fi
  fi

  ensure_venv "$venv"
  if ! require_arm64_python python; then
    exit 1
  fi

  if [[ -n "$local_lib" && -f "$local_lib" ]]; then
    local vllm_release_tag
    vllm_release_tag=$(read_vllm_release_tag)
    install_vllm "$vllm_release_tag"

    # Source checkouts build native artifacts; release installs use the wheel.
    uv pip install -e .
    if $mps; then
      export TOOLCHAINS="${TOOLCHAINS:-Metal}"
    fi
    ensure_metal_toolchain
    build_native_artifacts
    if $mps; then
      install_mps_dependencies
    fi
  else
    local release_data selected release_tag wheel_url vllm_release_tag
    release_data=$(fetch_release "$repo_owner" "$repo_name" "$channel")

    # extract_wheel_url prints the tag on the first line, the URL on the second.
    selected=$(printf '%s' "$release_data" | extract_wheel_url "$channel")
    release_tag=$(printf '%s' "$selected" | sed -n '1p')
    wheel_url=$(printf '%s' "$selected" | sed -n '2p')

    if [[ "$channel" == "stable" &&
          ! "$release_tag" =~ ^v[0-9]+\.[0-9]+\.[0-9]+(\.post[0-9]+)?$ ]]; then
      error "No stable release is available yet. Use --dev for the latest development build."
      exit 1
    fi

    if [[ -z "$wheel_url" ]]; then
      error "No wheel file found in the latest ${channel} release."
      exit 1
    fi

    vllm_release_tag=$(fetch_release_vllm_tag "$repo_owner" "$repo_name" "$release_tag")
    install_vllm "$vllm_release_tag"
    download_and_install_wheel "$wheel_url" "$package_name" "$release_tag"
  fi

  echo ""
  success "Installation complete!"
  echo ""
  echo "To use vllm, activate the virtual environment:"
  echo "  source $venv/bin/activate"
  if $mps; then
    echo "  export VLLM_METAL_BACKEND=mps TOOLCHAINS=Metal"
  fi
  echo ""
  echo "Or add the venv to your PATH:"
  echo "  export PATH=\"$venv/bin:\$PATH\""
}

main "$@"
