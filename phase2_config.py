"""Compatibility entry point; use analysis_config for new integrations."""
import sys
import analysis_config as _implementation

sys.modules[__name__] = _implementation
