#!/usr/bin/env python3
"""Backwards-compatible entry point for the audio classification pipeline.

The implementation now lives in the ``omni_classifier`` package. This shim keeps
the documented ``python audio_classifier_service.py --config ...`` command
working. New code should import from ``omni_classifier`` directly.
"""

from __future__ import annotations

from omni_classifier.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
