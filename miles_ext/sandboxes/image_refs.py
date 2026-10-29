"""Container image references as sandbox providers accept them."""

import re

_TAG_AND_DIGEST = re.compile(r":[^/@]+@sha256:")


def digest_only(ref: str) -> str:
    """``repo:tag@sha256:...`` -> ``repo@sha256:...`` (the digest alone names the same image); any other ref unchanged."""
    return _TAG_AND_DIGEST.sub("@sha256:", ref) if "@sha256:" in ref else ref
