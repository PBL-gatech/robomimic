#!/usr/bin/env python3
"""Helpers for formatting GUI evaluation results."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class ResultsViewerWidget:
    """Stores evaluation results and formats metadata for display."""

    def __init__(self, parent=None):
        self.parent = parent
        self.results: List[Dict[str, Any]] = []

    def set_results(self, results: List[Dict[str, Any]]):
        self.results = list(results)

    def add_result(self, result: Dict[str, Any]):
        self.results.append(result)

    def clear(self):
        self.results = []

    def display_name(self, result: Dict[str, Any]) -> str:
        checkpoint = result.get("checkpoint")
        demo_key = result.get("demo_key")
        csv_path = result.get("csv_path")
        if checkpoint:
            name = Path(checkpoint).name
            if demo_key is not None:
                return f"{name} | {demo_key}"
            return name
        if csv_path:
            return Path(csv_path).name
        return "Result"

    def load_metadata(self, result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        metadata = result.get("metadata")
        if isinstance(metadata, dict):
            return metadata
        metadata_path = Path(result.get("metadata_path", ""))
        if metadata_path.exists():
            with metadata_path.open(encoding="utf-8") as handle:
                return json.load(handle)
        return None

    def metadata_text(self, result: Dict[str, Any]) -> str:
        metadata = self.load_metadata(result)
        if metadata is None:
            return "No metadata available for this result."
        return json.dumps(metadata, indent=2, sort_keys=True)
