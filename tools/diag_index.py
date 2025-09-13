# tools/diag_index.py
import os, re, sys, importlib, inspect, json, argparse
from typing import Iterable, Tuple

DEF_MD_ROOT = "md"

def abspath_safe(p):
    try: return os.path.abspath(p)
    except Exception: return p

def color(s):
    # простая подсветка без внешних либ
    return s.replace("!!!", "\x1b[31m!!!\x1b[0m")

def grep_lines(text: str, pat: re.Pattern, max_hits=3) -> Iterable[str]:
    hits = 0
    for line in (text or "").splitlines():
        if pat.search(line):
            yield color(line.strip())
            hits += 1
            if hits >= max_hits:
                break

def try_import_app():
    try:
        sys.path.insert(0, os.getcwd())
        return importlib.import_module("app")
    except Exception:
        return None

def guess_corpus_vars(ns: dict):
    # возможные имена для корпуса/чанков/векторов
    keys = ["CORPUS","corpus","INDEX","index","EMBEDS","embeds","VECTORS","vectors","CHUNKS","chunks"]
    found = {}
    for k in keys:
        if k in ns:
            found[k] = ns[k]
    return found

def iter_corpus_items(obj):
    """
    Пробуем извлечь элементы корпуса в унифицированном виде:
    (doc_basename, full_path_or_doc, h2_id, h3_id, text)
    Поддерживаем список dict, список кортежей и произвольные объекты с .meta/.text/.file
    """
    if obj is None:
        return
    try:
        for it in obj:
            meta = {}
            text = None
            doc = None
            h2 = None
            h3 = None
            # варианты структур
            if isinstance(it, tuple) and len(it) >= 1:
                it = it[0]  # (chunk, score) -> chunk
            if isinstance(it, dict):
                meta = it.get("meta", {}) or {}
                text = it.get("text")
                doc = meta.get("doc") or it.get("file")
                h2 = meta.get("h2_id")
                h3 = meta.get("h3_id")
            else:
                meta = getattr(it, "meta", {}) or {}
                text = getattr(it, "text", None)
                doc = meta.get("doc") or getattr(it, "file", None)
                h2 = meta.get("h2_id")
                h3 = meta.get("h3_id")
            yield (os.path.basename(doc) if doc else None, doc, h2, h3, text or "")
    except Exception:
        return

def scan_md(md_root: str):
    for root, _, files in os.walk(md_root):
        for f in files:
            if f.endswith(".md"):
                path = os.path.join(root, f)
                try:
                    with open(path, "r", encoding="utf-8") as fh:
                        yield (f, path, None, None, fh.read())
                except Exception:
                    pass

def main():
    ap = argparse.ArgumentParser(description="Diagnose indexed files and grep pattern")
    ap.add_argument("--grep", default="!!!", help="regex to search, default '!!!'")
    ap.add_argument("--md", default=DEF_MD_ROOT, help="md root folder")
    ap.add_argument("--max-lines", type=int, default=3, help="max lines per file/section")
    args = ap.parse_args()

    pat = re.compile(args.grep, re.I)

    print("=== IN-MEMORY CORPUS (if any) ===")
    app = try_import_app()
    corpus_vars = guess_corpus_vars(vars(app)) if app else {}
    if corpus_vars:
        for name, obj in corpus_vars.items():
            print(f"\n[Var: {name}] type={type(obj)} size={getattr(obj,'__len__',lambda: '?')() if hasattr(obj,'__len__') else '?'}")
            for (doc_base, doc_full, h2, h3, text) in iter_corpus_items(obj) or []:
                hits = list(grep_lines(text, pat, max_hits=args.max_lines))
                if hits:
                    print(f"  → HIT in {doc_base}  path={abspath_safe(doc_full)}  h2_id={h2} h3_id={h3}")
                    for line in hits:
                        print(f"     {line}")
    else:
        print("  (no obvious corpus variable found)")

    print("\n=== MD FOLDER SCAN ===")
    for (doc_base, path, h2, h3, text) in scan_md(args.md):
        hits = list(grep_lines(text, pat, max_hits=args.max_lines))
        if hits:
            print(f"  → HIT in {doc_base}  path={abspath_safe(path)}")
            for line in hits:
                print(f"     {line}")

if __name__ == "__main__":
    main()

