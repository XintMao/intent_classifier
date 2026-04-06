"""
Extract topic metadata from a random sample of PubMed/PMC JATS XML files.

Produces a JSON array of lightweight topic records (title, keywords, MeSH terms,
journal, article type, subjects) that downstream scripts use for query generation.
"""

import argparse
import json
import logging
import random
import re
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def setup_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file),
        ],
    )


def strip_namespaces(xml_text: str) -> str:
    """Remove XML namespace declarations and prefixes so plain tag names work.

    Handles three cases:
      1. xmlns declarations:    xmlns:xlink="..."  → removed
      2. element tag prefixes:  <mml:math>         → <math>
      3. attribute name prefixes: xlink:href="..."  → href="..."
         (this was the bug: forgetting case 3 left unbound prefixes after
          case 1 removed their declarations, causing the XML parser to fail)
    """
    # 1. Remove xmlns declarations (both default and prefixed)
    xml_text = re.sub(r'\s+xmlns(?::\w+)?="[^"]*"', "", xml_text)
    # 2. Remove namespace prefixes from element tag names: <mml:math> → <math>
    xml_text = re.sub(r"<(/?)[\w.-]+:([\w.-]+)", r"<\1\2", xml_text)
    # 3. Remove namespace prefixes from attribute names: xlink:href="…" → href="…"
    xml_text = re.sub(r'(?<=[(\s,])[\w.-]+:([\w.-]+)\s*=', r'\1=', xml_text)
    return xml_text


def _text(el) -> str:
    """Return all text content under an element, stripped."""
    if el is None:
        return ""
    parts = []
    if el.text:
        parts.append(el.text.strip())
    for child in el:
        parts.append(_text(child))
        if child.tail:
            parts.append(child.tail.strip())
    return " ".join(p for p in parts if p)


def parse_xml_file(path: Path) -> dict | None:
    """Parse a single PMC JATS XML file and return a topic dict, or None on failure."""
    try:
        import defusedxml.ElementTree as ET
    except ImportError:
        import xml.etree.ElementTree as ET  # type: ignore[no-redef]

    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
        raw = strip_namespaces(raw)
        root = ET.fromstring(raw)
    except Exception as exc:
        logger.warning("Failed to parse %s: %s", path.name, exc)
        return None

    # article-type attribute
    article_type = root.get("article-type", "")

    front = root.find(".//front")
    if front is None:
        front = root  # fallback: search whole tree

    # Journal title
    journal_el = front.find(".//journal-title")
    journal = _text(journal_el) if journal_el is not None else ""

    # Article title
    title_el = front.find(".//article-title")
    title = _text(title_el) if title_el is not None else ""

    if not title:
        logger.warning("No article title in %s — skipping", path.name)
        return None

    # Keywords (generic kwd-group, excluding MeSH)
    keywords: list[str] = []
    mesh_terms: list[str] = []
    for kwd_group in front.findall(".//kwd-group"):
        group_type = (kwd_group.get("kwd-group-type") or "").lower()
        kwds = [_text(k) for k in kwd_group.findall(".//kwd") if _text(k)]
        if "mesh" in group_type:
            mesh_terms.extend(kwds)
        else:
            keywords.extend(kwds)

    # Subjects
    subjects: list[str] = []
    for subj_group in front.findall(".//subj-group"):
        sg_type = (subj_group.get("subj-group-type") or "").lower()
        if "discipline" in sg_type or sg_type == "":
            for subj in subj_group.findall(".//subject"):
                text = _text(subj)
                if text:
                    subjects.append(text)

    return {
        "file": path.name,
        "title": title,
        "journal": journal,
        "article_type": article_type,
        "keywords": keywords,
        "mesh_terms": mesh_terms,
        "subjects": subjects,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract topics from PMC XML files")
    parser.add_argument("--input-dir", required=True, help="Directory containing PMC XML files")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument("--sample-size", type=int, default=5000, help="Number of XML files to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    project_root = Path("/leonardo_scratch/large/userexternal/xmao0000/intent-classifier")
    setup_logging(project_root / "logs" / "extract_topics.log")

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Scanning %s for XML files…", input_dir)
    all_files = list(input_dir.glob("*.xml"))
    logger.info("Found %d XML files", len(all_files))

    random.seed(args.seed)
    sample = random.sample(all_files, min(args.sample_size, len(all_files)))
    logger.info("Sampling %d files (seed=%d)", len(sample), args.seed)

    # Progress bar — use tqdm if available
    try:
        from tqdm import tqdm
        iterator = tqdm(sample, desc="Parsing XML", unit="file")
    except ImportError:
        iterator = sample  # type: ignore[assignment]

    results: list[dict] = []
    failed = 0
    for i, path in enumerate(iterator):
        record = parse_xml_file(path)
        if record is not None:
            results.append(record)
        else:
            failed += 1
        # Fallback progress every 500 files when tqdm is not available
        if not hasattr(iterator, "update") and (i + 1) % 500 == 0:
            logger.info("Progress: %d / %d processed (%d failed)", i + 1, len(sample), failed)

    logger.info(
        "Done. Extracted %d records, %d failed out of %d sampled",
        len(results), failed, len(sample),
    )

    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    logger.info("Saved topics to %s", output_path)


if __name__ == "__main__":
    main()
