# The fork image: upstream's release image plus our commit checked out over
# /root/miles. CUDA, torch, SGLang, Megatron and every compiled extension stay
# exactly what upstream built and tested for this release; only Python source
# and the agentic stack (Harbor, mini-swe-agent) change.
#
# Build and push with fork/docker/build_on_modal.py, which fills every ARG
# from miles_ext/stack.py and the commit being released. The commit must be
# pushed to FORK_REPO first: the build fetches it, so the image can only hold
# code that is on GitHub.

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG HARBOR_REPO
ARG HARBOR_COMMIT
ARG MINI_SWE_AGENT_VERSION
ARG FORK_REPO
ARG FORK_COMMIT
ARG FORK_RELEASE

# Harbor (Miles' harbor-miles-* branch, pinned by commit) and mini-swe-agent.
# uv: the Harbor branch has a uv-workspace dependency; --python: the image's
# interpreter is the /opt/sglang venv, not /usr. This changes a few side
# packages in the image (openai, websockets, filelock, pyyaml); see fork/README.md.
RUN uv pip install --python "$(command -v python)" \
      "harbor[modal] @ git+${HARBOR_REPO}@${HARBOR_COMMIT}" \
      "mini-swe-agent==${MINI_SWE_AGENT_VERSION}"

# Our branch over the upstream checkout the base image was built from (same
# repo history, so this is a fast-forward of a few commits). --force keeps
# untracked build outputs in place; the editable install is re-registered in
# case setup.py changed. The marker scan fails the build if the fixes are missing.
RUN cd /root/miles \
 && git fetch --no-tags "${FORK_REPO}" "${FORK_COMMIT}" \
 && git checkout --force --detach "${FORK_COMMIT}" \
 && pip install -e . --no-deps --no-build-isolation -q \
 && python fork/tools/ledger.py markers \
 && python -c "import miles_ext.harbor.upstream as u; u.harbor_agent_function(); print('miles_ext ok')"

# miles_ext is a top-level package next to miles/ that upstream's setup.py does
# not list; PYTHONPATH makes it importable without editing setup.py.
ENV PYTHONPATH=/root/miles \
    MILES_FORK_RELEASE=${FORK_RELEASE} \
    MILES_FORK_COMMIT=${FORK_COMMIT}

LABEL org.opencontainers.image.source="${FORK_REPO}" \
      org.opencontainers.image.revision="${FORK_COMMIT}" \
      org.opencontainers.image.version="${FORK_RELEASE}" \
      org.opencontainers.image.base.name="${BASE_IMAGE}"
