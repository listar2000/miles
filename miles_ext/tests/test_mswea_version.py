"""The sandbox installs the mini-swe-agent version this image carries, not Harbor's latest:

python -m pytest miles_ext/tests/test_mswea_version.py
"""

from miles_ext.harbor import agent_function as af


def test_sandbox_install_pins_the_image_mini_swe_agent(monkeypatch):
    monkeypatch.setattr(af, "version", lambda name: "2.4.6")
    assert af._sandbox_kwargs("http://s/v1", "k", {}, "m", 1024)["version"] == "2.4.6"
