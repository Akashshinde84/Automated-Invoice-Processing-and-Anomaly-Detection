"""Compatibility entry point; use invoice_analysis for new integrations."""
import sys
import invoice_analysis as _implementation

if __name__ == '__main__':
    raise SystemExit(_implementation.main())

sys.modules[__name__] = _implementation
