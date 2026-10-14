"""Extensions carried by this Miles fork (see fork/README.md).

Everything here is new code that plugs into Miles through its own extension
points (``--custom-*-path`` hooks, Harbor ``import_path`` agents); nothing in
this package edits upstream modules. Bug fixes to upstream code are not here:
they are commits on the upstream files, listed in fork/README.md.

harbor/   Miles agent function running one Harbor trial per sample, the two
          mini-swe-agent harnesses, the reward hook, and the session proxy
          that lets in-sandbox agents reach the session servers; harbor/sail
          is Harbor's Sail sandbox provider (vendored) for recipes on Sail
recipes/  a run as data (YAML) -> Miles argv and Harbor env; model presets
launch/   test launcher for Modal (not part of the platform API)
stack.py  what the fork image is built from, and the image it produced
"""

__version__ = "0.1.2+fork.1"
