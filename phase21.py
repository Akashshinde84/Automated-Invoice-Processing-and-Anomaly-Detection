"""Compatibility entry point; use model_evaluation for new integrations."""
import sys
import model_evaluation as _implementation

if __name__ == '__main__':
    raise SystemExit(_implementation.main())

sys.modules[__name__] = _implementation
