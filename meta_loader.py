# meta_loader.py
from os.path import abspath, basename
import os, re, yaml

# front-matter между --- ... ---
_FM_RE = re.compile(r'^---\s*\n(.*?)\n---\s*\n?', re.S)

def _read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def _parse_front_matter(text: str) -> dict:
    m = _FM_RE.match(text)
    if not m:
        return {}
    try:
        fm = yaml.safe_load(m.group(1)) or {}
        if not isinstance(fm, dict):
            return {}
        return fm
    except Exception:
        return {}

def load_doc_meta(md_root: str = "md") -> dict:
    meta = {}
    for root, _, files in os.walk(md_root):
        for name in files:
            if not name.endswith(".md"):
                continue
            path = os.path.join(root, name)
            _DOC_PATHS[basename(name)] = abspath(path)
            fm = _parse_front_matter(_read_file(path))
            meta[name] = {
                "doc_type": fm.get("doc_type"),
                "subtype": fm.get("subtype"),
                "topic": fm.get("topic"),
                "verbatim": bool(fm.get("verbatim", False)),
                "cta_text": fm.get("cta_text"),
                "cta_action": fm.get("cta_action"),
                "preferred_format": fm.get("preferred_format") or [],
                "verbatim_ids": fm.get("verbatim_ids") or [],
                "suggest_h3": fm.get("suggest_h3") or [],
                "suggest_refs": fm.get("suggest_refs") or [],
            }
    return meta

_DOC_META = None
_DOC_PATHS = {}  # basename -> абсолютный путь

def get_doc_path(doc_name: str):
    global _DOC_PATHS, _DOC_META
    if not _DOC_PATHS:          # <— добавь эти две строки
        _DOC_META = load_doc_meta()
    return _DOC_PATHS.get(doc_name)

def get_doc_meta(doc_name: str) -> dict:
    """doc_name — basename файла, например 'clinic-contacts.md'"""
    global _DOC_META
    if _DOC_META is None:
        _DOC_META = load_doc_meta()
    return _DOC_META.get(doc_name, {})
