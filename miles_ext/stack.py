"""The stack: what the fork image is built from, and the image a launcher runs.

Build inputs (read by fork/docker/build.py) and the published result
(read by launchers) sit in one file, so a stack bump is one diff here plus the
rebased fork branch.
"""

from pathlib import Path

# Upstream release and commit this branch is based on, and its published image. The fork
# image is this image's layers untouched plus two of ours (Harbor deps, our
# source over /root/miles), so CUDA, torch, SGLang and Megatron stay exactly
# what upstream tested.
UPSTREAM_TAG = "v0.1.3"
# The upstream commit main last merged: UPSTREAM_TAG itself, or between releases
# the commit BASE_IMAGE was built from (RFC 0001). The ledger and the src
# layer's whiteouts diff against it.
UPSTREAM_BASE = "v0.1.3"
# What image labels and W&B tags call the upstream: the tag, plus the commit when main is past it
UPSTREAM_LABEL = UPSTREAM_TAG if UPSTREAM_BASE == UPSTREAM_TAG else f"{UPSTREAM_TAG}+{UPSTREAM_BASE[:10]}"
# radixark/miles:v0.1.3, the release image (2026-10-23): Miles v0.1.3 (74f47d37c8), sglang-miles
# 3a938213c1 on SGLang v0.5.22, Megatron-LM miles-main 2fbe25a (the tag's release-lock.json)
BASE_IMAGE = "radixark/miles:v0.1.3@sha256:bb235de60c2dd481c5255e1eb9f104b37ce47d125eb288267b4b00f44b51a845"
# The SGLang checkout BASE_IMAGE ships (sglang-miles, installed editable). The fork's SGLang
# patches (fork/deps/sglang/) are pinned to this commit: a sync that moves it re-checks them.
SGLANG_SRC = "/sgl-workspace/sglang"
SGLANG_COMMIT = "3a938213c1"

# Harbor: the branch Miles documents for v0.1.2 and v0.1.3 (harbor-miles-v0.20.0), pinned
# by commit. Runtime changes we need go through Harbor's import_path (agents,
# environments) in miles_ext; see fork/README.md "Harbor" for when that stops
# being enough.
HARBOR_REPO = "https://github.com/harbor-framework/harbor"
HARBOR_COMMIT = "917b3c1be7cf6553a7a9fe25de8adb6f8e639287"
MINI_SWE_AGENT_VERSION = "2.4.6"
# Sail SDK for the vendored Sail provider (miles_ext/harbor/sail), pinned
# exactly: the vendored file imports SDK internals (sail._core_client, sail.image)
SAIL_SDK_VERSION = "0.13.0"

# The deps layer: one install line for fork/docker/build.py and the launcher's
# build-from-commit image. uv: the Harbor branch has a uv-workspace
# dependency; --python: the image's interpreter is the /opt/sglang venv, not /usr
DEPS_INSTALL = (
    'uv pip install --python "$(command -v python)" '
    f'"harbor[modal] @ git+{HARBOR_REPO}@{HARBOR_COMMIT}" '
    f'"mini-swe-agent=={MINI_SWE_AGENT_VERSION}" "sail=={SAIL_SDK_VERSION}"'
)

FORK_REPO = "https://github.com/listar2000/miles"
MILES_SRC = Path("/root/miles")

# Published fork image (fork/docker/build.py prints the digest after a push). Release tags are fork-<upstream tag>-r<N>, so they never match
# upstream's release-docker trigger (v[0-9]+.*).
FORK_RELEASE = "fork-v0.1.2-r6"
FORK_IMAGE = "docker.io/starlimobius/miles:fork-v0.1.2-r6@sha256:a3715f98ac5fbd36405df78fd3d83ac4b0f817bd9a56187adcca780d8caa9126"  # built from 28dcfeb141

# Trainer versions (miles_ext/platform/versions.py): main's newest build is the
# moving tag LATEST_TAG; each build also keeps an immutable <sha10> tag.
# A stable version is a dated tag on a commit and on its image, listed in
# miles_ext/stable.json. Until the platform moves to versions, FORK_IMAGE stays
# the image it runs.
IMAGE_REPO = "docker.io/starlimobius/miles-mobius"  # miles-mobius images; r1-r6 are in starlimobius/miles
LATEST_TAG = "main"
STABLE_RE = r"stable-\d{4}-\d{2}-\d{2}(\.\d+)?"
