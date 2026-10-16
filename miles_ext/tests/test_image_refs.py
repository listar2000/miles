"""Sail refuses an image ref with both a tag and a digest; TB4 v4.0.0 pins every image that way:

    python -m pytest miles_ext/tests/test_image_refs.py
"""

import pytest

from miles_ext.harbor.image_refs import digest_only

DIGEST = "sha256:" + "f" * 64
TAGGED = f"harborframework/terminal-bench:freight-dispatch-shift-environment-b8a25b24cf23c8d7@{DIGEST}"


@pytest.mark.parametrize("ref,want", [
    (TAGGED, f"harborframework/terminal-bench@{DIGEST}"),
    (f"localhost:5000/repo:tag@{DIGEST}", f"localhost:5000/repo@{DIGEST}"),
    (f"repo@{DIGEST}", f"repo@{DIGEST}"),
    ("repo:tag", "repo:tag"),
    ("localhost:5000/repo:tag", "localhost:5000/repo:tag"),
])
def test_digest_only(ref, want):
    assert digest_only(ref) == want
