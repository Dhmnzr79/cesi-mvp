import os, json, re, glob, numpy as np, frontmatter
from dotenv import load_dotenv
from openai import OpenAI
# --- logging (устойчиво) ---
try:
    from logging_setup import log_json, setup_logging  # если есть твой модуль
except Exception:
    import logging, json as _json
    def log_json(logger, msg, **fields):
        try:
            logger.info(f"{msg} " + _json.dumps(fields, ensure_ascii=False))
        except Exception:
            logger.info(msg)
    def setup_logging():
        logger = logging.getLogger("builder")
        if not logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
            logger.addHandler(h)
            logger.setLevel(logging.INFO)
        return logger
# инициализация
logger = setup_logging()
# --- /logging ---

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY is not set in .env")

client = OpenAI(api_key=api_key)
EMB_MODEL = os.getenv("MODEL_EMBED", "text-embedding-3-small")

ALIAS_RX = re.compile(r"<!--\s*aliases:\s*\[(.*?)\]\s*-->", re.I|re.S)

def extract_local_aliases(block_text:str) -> list[str]:
    m = ALIAS_RX.search(block_text or "")
    if not m: 
        return []
    # Вытащим "..." из массива (может быть с запятыми/пробелами)
    return re.findall(r'"([^"]+)"', m.group(1))

def split_md_to_chunks(text):
    # режем по H2/H3. H2: ## , H3: ###
    lines = text.splitlines()
    chunks, h2, h2_id, h3, h3_id, buf = [], None, None, None, None, []
    def flush():
        if buf:
            chunks.append({"h2": h2, "h2_id": h2_id, "h3": h3, "h3_id": h3_id,
                           "text": "\n".join(buf).strip()})
    h2rx = re.compile(r"^##\s+(.+?)(?:\s*\{#([a-z0-9\-\_]+)\})?\s*$", re.I)
    h3rx = re.compile(r"^###\s+(.+?)(?:\s*\{#([a-z0-9\-\_]+)\})?\s*$", re.I)
    for ln in lines:
        m2 = h2rx.match(ln); m3 = h3rx.match(ln)
        if m2:
            flush(); buf=[]; h2, h2_id = m2.group(1).strip(), (m2.group(2) or "").strip()
            h3, h3_id = None, None
        elif m3:
            flush(); buf=[]; h3, h3_id = m3.group(1).strip(), (m3.group(2) or "").strip()
        else:
            buf.append(ln)
    flush()
    # чистим пустые
    return [c for c in chunks if c["text"]]

def embed_batch(texts):
    resp = client.embeddings.create(model=EMB_MODEL, input=texts)
    return [np.array(d.embedding, dtype=np.float32) for d in resp.data]

def main():
    log_json(logger, "Starting index build")
    os.makedirs("data", exist_ok=True)
    corpus, embeds = [], []
    for path in glob.glob("md/**/*.md", recursive=True):
        fm = frontmatter.load(path)
        meta = fm.metadata or {}
        followups = meta.get("followups") or []
        doc_id = meta.get("doc_id") or os.path.splitext(os.path.basename(path))[0]
        for ch in split_md_to_chunks(fm.content):
            # локальные алиасы из комментария под H2/H3
            local_aliases = extract_local_aliases(ch["text"])
            # алиасы из фронт-маттера (шапки файла)
            doc_aliases = meta.get("aliases") or []

            item = {
                "doc": doc_id,
                "file": os.path.basename(path),
                "topic": meta.get("topic"),
                "verbatim": bool(meta.get("verbatim", False)),
                "preferred_format": meta.get("preferred_format", []),
                "cta_action": meta.get("cta_action"),
                "cta_text": meta.get("cta_text"),
                "empathy_enabled": bool(meta.get("empathy_enabled", False)),
                "empathy_tag": meta.get("empathy_tag"),
                "followups": meta.get("followups", []),
                "h2": ch["h2"], "h2_id": ch["h2_id"],
                "h3": ch["h3"], "h3_id": ch["h3_id"],
                "text": ch["text"],
                "aliases": list(set(doc_aliases + local_aliases)),
            }
            corpus.append(item)
    
    # эмбеддим не «голый чанк», а «заголовки + алиасы + чанк»
    def text_for_embedding(row):
        parts = []
        if row.get("h2"): parts.append(row["h2"])
        if row.get("h3"): parts.append(row["h3"])
        if row.get("aliases"):
            parts.append(" | ".join(row["aliases"]))
        # сам чанк, но без html-комментария aliases
        clean = ALIAS_RX.sub("", row["text"]).strip()
        parts.append(clean)
        return "\n".join([p for p in parts if p])

    # эмбеддинги (батчами по 64)
    B=64
    for i in range(0, len(corpus), B):
        texts = [text_for_embedding(c)[:4000] for c in corpus[i:i+B]]
        embeds.extend(embed_batch(texts))
    arr = np.vstack(embeds).astype(np.float32)
    # нормализуем для косинусной близости
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
    arr = arr / norms
    np.save("data/embeddings.npy", arr)
    with open("data/corpus.jsonl","w",encoding="utf-8") as f:
        for row in corpus: f.write(json.dumps(row, ensure_ascii=False)+"\n")
    
    log_json(logger, "Index build completed", 
             chunks_count=len(corpus), embeddings_shape=arr.shape)
    print(f"OK: chunks={len(corpus)}  -> data/embeddings.npy, data/corpus.jsonl")

if __name__ == "__main__":
    main()
