from __future__ import annotations

import json
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .config import SourceConfig


@dataclass(frozen=True)
class SourceDocument:
    source_name: str
    uri: str
    text: str
    license_note: str


def iter_source(source: SourceConfig) -> Iterator[SourceDocument]:
    if not source.enabled:
        return
    if source.kind == "text":
        yield SourceDocument(source.name, source.location, Path(source.location).read_text(encoding="utf-8"), source.license_note)
        return
    if source.kind == "jsonl":
        with Path(source.location).open(encoding="utf-8") as fh:
            for index, line in enumerate(fh):
                if index >= source.max_documents:
                    break
                if not line.strip():
                    continue
                record = json.loads(line)
                text = record.get("text", record.get("input", "")) if isinstance(record, dict) else str(record)
                if text:
                    yield SourceDocument(source.name, f"{source.location}#L{index + 1}", str(text), source.license_note)
        return
    request = urllib.request.Request(source.location, headers={"User-Agent": "jev/0.1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        body = response.read().decode("utf-8", errors="replace")
    if source.kind == "url":
        yield SourceDocument(source.name, source.location, body, source.license_note)
    elif source.kind == "rss":
        root = ET.fromstring(body)
        for index, item in enumerate(root.findall(".//item")):
            if index >= source.max_documents:
                break
            title = item.findtext("title", "")
            description = item.findtext("description", "")
            link = item.findtext("link", source.location)
            if title or description:
                yield SourceDocument(source.name, link, f"{title}\n{description}", source.license_note)

