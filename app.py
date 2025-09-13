import os, json, numpy as np, time
import re
from datetime import datetime
from uuid import uuid4
from dotenv import load_dotenv
from meta_loader import get_doc_meta, get_doc_path
load_dotenv()

from flask import Flask, request, jsonify, send_from_directory
from openai import OpenAI
from logging_setup import get_logger, make_request_context, log_json

app = Flask(__name__, static_folder="static")
logger = get_logger("bot")
DEBUG_TOKEN = os.getenv("DEBUG_TOKEN", "dev-debug")

CONTACTS_RE = re.compile(r"(адрес|где.*находитесь|как\s+(доехать|проехать)|время\s+работы|график|телефон|whatsapp|карта|расположение)", re.I)
PRICES_RE = re.compile(r"(цена|стоимост|сколько\s+стоит|прайс|расценк|по\s+цене|сколько\s+будет|сколько\s+руб)", re.I)

# ---- JSON sanitize helpers ----
def _to_plain(o):
    import numpy as _np
    if isinstance(o, (_np.floating,)):
        return float(o)
    if isinstance(o, (_np.integer,)):
        return int(o)
    if isinstance(o, _np.ndarray):
        return o.tolist()
    if isinstance(o, set):
        return list(o)
    return o

def _sanitize(x):
    if isinstance(x, dict):
        return {k: _sanitize(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_sanitize(v) for v in x]
    return _to_plain(x)

def safe_jsonify(payload):
    return jsonify(_sanitize(payload))
# ---- end helpers ----

# --- ids helper ---
def _h_ids(d: dict):
    """Безопасно достаём h2/h3 из чанка (поддержка h2_id/h3_id)."""
    if not isinstance(d, dict): 
        return (None, None)
    h2 = d.get("h2") or d.get("h2_id")
    h3 = d.get("h3") or d.get("h3_id")
    return (h2, h3)

def _get_ids(chunk: dict):
    """Вернёт (h2_id, h3_id) безопасно."""
    if not isinstance(chunk, dict): return (None, None)
    return (chunk.get("h2_id"), chunk.get("h3_id"))

def _is_overview_by_ids(h2_id, h3_id):
    h2 = (h2_id or "").strip().lower()
    h3 = (h3_id or "").strip().lower()
    return (not h2 and not h3) or (h2 == "overview") or (h3 == "overview")

def _extract_id_from_heading(txt: str):
    """Возвращает id из строки заголовка вида 'Заголовок {#id}'. Если нет — None."""
    if not isinstance(txt, str): return None
    m = re.search(r"\{\s*#([^\}]+)\s*\}", txt)
    return m.group(1).strip() if m else None
# --- end helper ---

def _heading_label(md_file: str, sect_id: str):
    """
    Возвращает текст заголовка для H3/H2 по его {#id}.
    Если не нашли — вернём id, приведённый к "Человекочитаемо".
    """
    if not md_file or not sect_id:
        return (sect_id or "").replace('-', ' ').capitalize()
    try:
        path = get_doc_path(os.path.basename(md_file)) or md_file
        with open(path, 'r', encoding='utf-8') as f:
            txt = f.read()
        # сначала H3, потом H2
        rx3 = re.compile(rf'^###\s+(.*?)\s*\{{#{re.escape(sect_id)}\}}\s*$', re.M | re.I)
        rx2 = re.compile(rf'^##\s+(.*?)\s*\{{#{re.escape(sect_id)}\}}\s*$',  re.M | re.I)
        m = rx3.search(txt) or rx2.search(txt)
        if m:
            return m.group(1).strip()
    except Exception:
        pass
    return (sect_id or "").replace('-', ' ').capitalize()


def get_chunk_by_ref(ref: str):
    if not ref or "#" not in ref: return None
    fname, anchor = ref.split("#", 1)
    base = os.path.basename(fname)
    a = (anchor or "").strip().lower()
    corpus = _load_corpus_if_needed()
    cands = [ch for ch in corpus if os.path.basename(ch.get("file","") or "") == base]
    if not cands: return None
    if a in ("overview", "", None):
        for ch in cands:
            if not ch.get("h2_id") and not ch.get("h3_id"):
                ch["_score"] = 1.0; return ch
        ch = cands[0]; ch["_score"] = 1.0; return ch
    for ch in cands:
        hid2 = ch.get("h2_id") or _extract_id_from_heading(ch.get("h2"))
        hid3 = ch.get("h3_id") or _extract_id_from_heading(ch.get("h3"))
        if a in {
            (hid3 or "").lower(),
            (hid2 or "").lower(),
            str(ch.get("h3") or "").lower(),
            str(ch.get("h2") or "").lower()
        }:
            ch["_score"] = 1.0; return ch
    return None

# === UX-утилиты ===
# quick_refs — только suggest_refs; фильтруем самоссылку по текущему H2/H3
def _build_quick_refs(meta: dict, md_file: str, current_h2_id: str, current_h3_id: str):
    out = []
    cur_anchor = (current_h3_id or current_h2_id or "overview")
    cur_ref = f"{os.path.basename(md_file or '')}#{cur_anchor}".lower() if md_file else None
    for r in (meta.get("suggest_refs") or []):
        if isinstance(r, str):
            ref = r if "#" in r else None
            label = r.split("#",1)[0] if ref else None
        else:
            ref = r.get("ref"); label = r.get("label") or (ref.split("#",1)[0] if ref else None)
        if not (label and ref): continue
        if cur_ref and ref.lower() == cur_ref:  # самоссылку выкидываем
            continue
        out.append({"label": label, "ref": ref})
    return out

# followups — из suggest_h3 (id внутренних секций этого же файла),
# исключаем текущий H2/H3 (если совпало), label берём из заголовка
def _build_followups(meta: dict, md_file: str, current_h2_id: str, current_h3_id: str):
    out = []
    for s in (meta.get("suggest_h3") or []):
        h_id = s if isinstance(s, str) else (s.get("h3_id") or s.get("id"))
        if not h_id: continue
        if str(h_id).lower() in { str(current_h2_id or '').lower(), str(current_h3_id or '').lower() }:
            continue
        label = _heading_label(md_file, h_id) if '_heading_label' in globals() else (str(h_id).replace('-',' ').capitalize())
        out.append({"label": label, "ref": f"{os.path.basename(md_file)}#{h_id}"})
    return out

def _build_cta(meta: dict):
    if meta.get("cta_text") and meta.get("cta_action"):
        return {"text": meta["cta_text"], "action": meta["cta_action"]}
    return None

# Заглушка оффера (можно оставить None; офферы прикрутим позже)
def _pick_relevant_offer(meta: dict):
    return None

# === Разбор секций в .md и кэш индекса по файлам ===
_RE_H2 = re.compile(r"^##\s+.*?\{#([a-z0-9\-]+)\}\s*$", re.I | re.M)
_RE_H3 = re.compile(r"^###\s+.*?\{#([a-z0-9\-]+)\}\s*$", re.I | re.M)
_SECTION_CACHE = {}  # {abs_path: {"text": ..., "h2": [(pos,id)], "h3":[(pos,id)]}}

def _load_doc_text(md_path):
    with open(md_path, "r", encoding="utf-8") as f:
        return f.read()

def _build_section_index(md_path):
    abs_path = os.path.abspath(md_path)
    cached = _SECTION_CACHE.get(abs_path)
    if cached:
        return cached
    try:
        text = _load_doc_text(abs_path)
    except Exception:
        text = ""
    h2 = [(m.start(), m.group(1)) for m in _RE_H2.finditer(text)]
    h3 = [(m.start(), m.group(1)) for m in _RE_H3.finditer(text)]
    data = {"text": text, "h2": h2, "h3": h3}
    _SECTION_CACHE[abs_path] = data
    return data

def _infer_section_ids(md_path, fragment):
    """
    Определяем секцию H2/H3 для фрагмента.
    1) Пытаемся найти иглу (первая некомментарная строка) обычным find().
    2) Если не нашли — делаем нормализацию Markdown и ищем по блокам каждого H2.
    """
    if not md_path or not fragment:
        return (None, None)
    idx = _build_section_index(md_path)
    doc_text = idx["text"] or ""

    # --- соберём "иглы"
    lines = (fragment or "").splitlines()
    needles = []
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("<!--"):
            continue
        needles.append(s[:120])
        break
    if not needles:
        needles.append((fragment or "").strip()[:120])

    # 1) обычный поиск
    pos = -1
    for nd in needles:
        if not nd: 
            continue
        pos = doc_text.find(nd)
        if pos >= 0:
            break
    if pos >= 0:
        # нашли точную подстроку — определим ближайшие H2/H3 слева
        h2_id = None; h3_id = None
        for p, hid in idx["h2"]:
            if p <= pos: h2_id = hid
            else: break
        for p, hid in idx["h3"]:
            if p <= pos: h3_id = hid
            else: break
        return (h2_id, h3_id)

    # 2) поиск по H2-блокам с нормализацией Markdown
    def _norm(s:str)->str:
        s = s or ""
        # уберём ** __ ` и лишние пробелы/переносы
        s = re.sub(r"[*_`]", "", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    needles_n = [_norm(x) for x in needles if x]
    if not needles_n:
        # если единственный H2 — считаем overview
        return (idx["h2"][0][1], None) if len(idx["h2"]) == 1 else (None, None)

    # построим границы H2-блоков
    h2s = idx["h2"]
    if not h2s:
        return (None, None)
    bounds = []
    for i,(p,hid) in enumerate(h2s):
        p2 = h2s[i+1][0] if i+1 < len(h2s) else len(doc_text)
        bounds.append((p, p2, hid))

    # ищем, в каком H2-блоке встречается нормализованная игла
    for start, end, hid in bounds:
        block = _norm(doc_text[start:end])
        if any(nd and nd in block for nd in needles_n):
            # внутри блока можно попытаться найти ближайший H3 (не обязательно)
            h3_id = None
            for p,h3id in idx["h3"]:
                if start <= p < end:
                    h3_id = h3_id or h3id  # берём первый H3 в блоке
            return (hid, h3_id)

    # fallback: если один H2 — overview
    if len(h2s) == 1:
        return (h2s[0][1], None)
    return (None, None)

def _doc_type_of(item):
    try:
        # формат (chunk, score)
        return getattr(item[0], "meta", {}).get("doc_type") or None
    except Exception:
        # формат dict
        return (item.get("doc_type") if isinstance(item, dict) else None)

def _score_of(item):
    try:
        return float(item[1])
    except Exception:
        try:
            return float(item.get("score"))
        except Exception:
            return None

# лог при старте
log_json(logger, "app_start", env=os.getenv("APP_ENV"), version=os.getenv("APP_VERSION"))

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY is not set in .env")

client = OpenAI(api_key=api_key)
EMB_MODEL = os.getenv("MODEL_EMBED","text-embedding-3-small")
CHAT_MODEL= os.getenv("MODEL_CHAT","gpt-4o-mini")
PORT      = int(os.getenv("PORT", "9000"))

# загрузка индекса (ленивая)
_CORPUS = None

def _load_corpus_if_needed():
    global _CORPUS
    if _CORPUS is None:
        try:
            with open(os.path.join("data","corpus.jsonl"), "r", encoding="utf-8") as f:
                _CORPUS = [json.loads(line) for line in f if line.strip()]
        except FileNotFoundError:
            _CORPUS = []
    return _CORPUS

EMB    = np.load("data/embeddings.npy")  # уже нормализованы

def _chunk_info(ch, sc=None):
    """
    Универсальный безопасный сбор метаданных о кандидате.
    Поддерживает как dict, так и объект с .meta/.text/.id
    """
    meta = {}
    text = None
    cid = None
    doc = None
    h2 = None
    h3 = None
    doc_type = None
    subtype = None

    if isinstance(ch, dict):
        meta = ch.get("meta", {}) or {}
        text = ch.get("text")
        cid  = ch.get("id")
        # ключевой момент — file/h2_id/h3_id берём из самого dict
        doc  = ch.get("file") or meta.get("doc") or ch.get("doc")
        h2   = ch.get("h2_id") or meta.get("h2_id")
        h3   = ch.get("h3_id") or meta.get("h3_id")
        doc_type = meta.get("doc_type") or ch.get("doc_type")
        subtype  = meta.get("subtype")  or ch.get("subtype")
    else:
        meta = getattr(ch, "meta", {}) or {}
        text = getattr(ch, "text", None)
        cid = getattr(ch, "id", None)
        doc = meta.get("doc") or getattr(ch, "file", None)
        h2 = meta.get("h2_id")
        h3 = meta.get("h3_id")
        doc_type = meta.get("doc_type")
        subtype = meta.get("subtype")

    # doc может быть только basename — найдём полный путь через meta_loader
    doc_base = os.path.basename(doc) if doc else None
    full_md_path = None
    if doc_base:
        full_md_path = get_doc_path(doc_base)
    if not full_md_path:
        # запасной путь: перебор стандартных расположений
        guess = doc if os.path.exists(doc or "") else os.path.join("md", doc_base or "")
        full_md_path = guess if os.path.exists(guess) else None

    if (h2 is None and h3 is None) and full_md_path and text:
        h2_guess, h3_guess = _infer_section_ids(full_md_path, text)
        h2 = h2 or h2_guess
        h3 = h3 or h3_guess

    # Подмешать метаданные из фронт-маттера по имени файла
    doc_base = os.path.basename(doc) if doc else None
    fm = get_doc_meta(doc_base) if doc_base else {}
    if not doc_type:
        doc_type = fm.get("doc_type")
    if not subtype:
        subtype = fm.get("subtype")

    return {
        "id": cid,
        "doc": doc,
        "doc_type": doc_type,
        "subtype": subtype,
        "h2_id": h2,
        "h3_id": h3,
        "score": (round(float(sc), 4) if sc is not None else None),
        "snippet": (text[:180] if isinstance(text, str) else None)
    }

def embed_q(q:str)->np.ndarray:
    v = client.embeddings.create(model=EMB_MODEL, input=q).data[0].embedding
    v = np.array(v, dtype=np.float32)
    v = v / (np.linalg.norm(v) + 1e-9)
    return v

def retrieve(q:str, topk:int=4):
    v = embed_q(q)
    sims = EMB @ v
    idx = np.argsort(-sims)[:max(topk,8)]  # возьмём немного с запасом
    # дедуп по (file,h2_id,h3_id)
    seen, out = set(), []
    corpus = _load_corpus_if_needed()
    for i in idx:
        c = corpus[int(i)]
        key = (c["file"], c.get("h2_id") or c.get("h2"), c.get("h3_id") or c.get("h3"))
        if key in seen: continue
        seen.add(key); c2 = dict(c); c2["_score"]=float(sims[int(i)])
        out.append(c2)
        if len(out) == topk: break
    
    # Логирование результата retrieve
    try:
        chunks_used = [_chunk_info(item, item.get("_score")) for item in out[:topk]]
    except Exception:
        chunks_used = []

    log_json(logger, "retrieval_result",
             used_query=q,
             k=topk,
             dedup_keys=["file","h2_id","h3_id"],
             chunks_used=chunks_used,
             top_score=(chunks_used[0]["score"] if chunks_used else None))
    
    return out

def llm_rerank(q, cands):  # cands: list[dict], len<=3
    # Логирование входных кандидатов
    t0 = time.time()
    try:
        cand_infos = [_chunk_info(ch, ch.get("_score")) for ch in cands]
    except Exception:
        cand_infos = [_chunk_info(ch, None) for ch in cands]
    log_json(logger, "rerank", question=q[:200], candidates=cand_infos)
    
    # Сведём к коротким отрывкам
    prompt = "Выбери самый уместный фрагмент для ответа на вопрос пользователя. Ответи номером 1, 2 или 3."
    msgs = [{"role":"system","content":"Ты выбираешь лучший фрагмент."},
            {"role":"user","content": f"{prompt}\n\nВопрос: {q}\n\n1) {cands[0]['text'][:600]}\n\n2) {cands[1]['text'][:600] if len(cands)>1 else ''}\n\n3) {cands[2]['text'][:600] if len(cands)>2 else ''}"}]
    try:
        out = client.chat.completions.create(model=CHAT_MODEL, messages=msgs, temperature=0)
        n = "".join([ch for ch in out.choices[0].message.content if ch.isdigit()])[:1]
        idx = int(n)-1
        result = cands[idx] if 0 <= idx < len(cands) else cands[0]
    except Exception:
        result = cands[0]
    
    # Логирование результата реранка
    lat = int((time.time() - t0) * 1000)
    log_json(logger, "rerank_result",
             latency_ms=lat,
             chosen=_chunk_info(result, result.get("_score") if isinstance(result, dict) else None))
    
    return result

def compose_answer(q:str, chunk:dict)->str:
    # минимальный ответ — без копипасты, короткий перефраз (можно выключить)
    prompt = f"""Отвечай кратко, дружелюбно и по делу (2–4 предложения).
Перефразируй содержание справки ниже и ответь на вопрос.
Не придумывай новые факты. Если есть цифры — сохрани их.

Вопрос: {q}

Справка:
{chunk['text']}
"""
    resp = client.chat.completions.create(model=CHAT_MODEL,
        messages=[{"role":"system","content":"Ты ассистент стоматологической клиники. Пиши просто и понятно."},
                  {"role":"user","content":prompt}],
        temperature=0.2)
    return resp.choices[0].message.content.strip()

@app.before_request
def _before():
    request.ctx = make_request_context(session_id=request.cookies.get("sid"))
    request.ctx["path"] = request.path
    request.ctx["method"] = request.method
    request.ctx["t0"] = time.time()

@app.after_request
def _after(resp):
    latency = int((time.time() - request.ctx["t0"]) * 1000)
    log_json(logger, "http_request",
             **{**request.ctx,
                "status": resp.status_code,
                "latency_ms": latency,
                "ip": request.remote_addr})
    return resp

@app.get("/_debug/ping")
def debug_ping():
    # опционально проверка токена заголовком X-Debug-Token
    if request.headers.get("X-Debug-Token") and request.headers.get("X-Debug-Token") != DEBUG_TOKEN:
        return jsonify({"error":"unauthorized"}), 401
    return jsonify({"ok": True})

@app.post("/ask")
def ask():
    try:
        data = request.get_json(force=True) or {}
        q = (data.get("q") or "").strip()
        ref = (data.get("ref") or "").strip()
        
        # Если есть ref - попробуем найти чанк по ссылке
        if ref:
            ch = get_chunk_by_ref(ref)
            if ch:
                top = ch
                # Генерируем ответ для найденного чанка
                answer = compose_answer(q or f"Информация из {ref}", top)
                
                # Фолбэк для пустого ответа
                if not isinstance(answer, str) or not answer.strip():
                    fallback = (top.get("text") or "").strip()
                    answer = (fallback[:800] + ("…" if len(fallback) > 800 else "")) or \
                             "Пока не нашёл точный ответ. Можете уточнить вопрос?"
                
                # Логирование результата
                log_json(logger, "Answer generated from ref", 
                         file=top.get("file"), score=round(float(top.get("_score",0.0)),3), answer_length=len(answer))
                
                # Сборка UX-данных
                md_file = top.get("file")
                h2_id, h3_id = _get_ids(top)
                h2_val = top.get("h2") or top.get("h2_id")
                h3_val = top.get("h3") or top.get("h3_id")
                is_overview = _is_overview_by_ids(h2_id, h3_id)

                meta_doc = get_doc_meta(os.path.basename(md_file or "")) or {}

                quick_refs = _build_quick_refs(meta_doc, md_file, h2_id, h3_id)
                fups_full  = _build_followups(meta_doc, md_file, h2_id, h3_id)

                # показываем followups только на overview и максимум 1
                followups = fups_full[:1] if is_overview else []

                score = float(round(float(top.get("_score", 0.0)), 3))

                payload = {
                    "answer": answer,
                    "quick_replies": quick_refs,   # только suggest_refs
                    "cta": _build_cta(meta_doc),
                    "offer": _pick_relevant_offer(meta_doc),
                    "meta": {
                        "file": md_file,
                        "h2": h2_val, "h3": h3_val,   # для читаемости
                        "h2_id": h2_id, "h3_id": h3_id,  # для стабильной логики
                        "score": score,
                        "followups": followups,
                        "is_overview": bool(is_overview),
                        "cta_mode": meta_doc.get("cta_mode"),
                        "tags": (list(meta_doc.get("tags")) if isinstance(meta_doc.get("tags"), set) else (meta_doc.get("tags") or []))
                    }
                }

                return safe_jsonify(payload)
        
        if not q: 
            return safe_jsonify({
                "answer": "Уточните вопрос.",
                "quick_replies": [], "cta": None, "offer": None, 
                "meta": {"error": "empty_question"}
            })
        
        # Безопасное логирование запроса (без секретов)
        log_json(logger, "Processing question", question=q[:100], question_length=len(q))
        
        cands = retrieve(q, topk=3)
        if not cands:
            log_json(logger, "No candidates found", question=q[:50])
            return safe_jsonify({
                "answer": "Пока не нашёл подходящий материал в базе. Сформулируйте вопрос иначе или выберите один из вариантов ниже.",
                "quick_replies": [], "cta": None, "offer": None, 
                "meta": {"file": None}
            })
        
        # Приоритетный выбор для контактных запросов
        is_contacts_intent = bool(CONTACTS_RE.search(q or ""))
        if is_contacts_intent:
            picked = None
            for it in cands:
                dt = (_doc_type_of(it) or "")
                if dt.lower() == "contacts":
                    picked = it
                    break
            if picked is not None:
                # нормализуем к (final_chunk, final_score)
                try:
                    final_chunk, final_score = picked
                except Exception:
                    final_chunk, final_score = picked, _score_of(picked)
                use_rerank = False
                top_score = _score_of(cands[0]) if cands else None
                log_json(logger, "selection",
                         question=q[:200],
                         original_top_score=(round(float(top_score), 4) if top_score is not None else None),
                         rerank_applied=False,
                         chosen=_chunk_info(final_chunk, final_score))
                
                # Генерация ответа для контактного запроса
                answer = compose_answer(q, final_chunk)
                
                # Фолбэк для пустого ответа
                if not isinstance(answer, str) or not answer.strip():
                    fallback = (final_chunk.get("text") or "").strip()
                    answer = (fallback[:800] + ("…" if len(fallback) > 800 else "")) or \
                             "Пока не нашёл точный ответ. Можете уточнить вопрос?"
                
                log_json(logger, "Answer generated", 
                         file=final_chunk["file"], score=round(float(final_score),3), answer_length=len(answer))
                
                # Сборка UX-данных
                meta = get_doc_meta(os.path.basename(final_chunk.get("file",""))) or {}
                md_file = final_chunk.get("file")
                h2_id, h3_id = _get_ids(final_chunk)
                h2_val = final_chunk.get("h2") or final_chunk.get("h2_id")
                h3_val = final_chunk.get("h3") or final_chunk.get("h3_id")
                is_overview = _is_overview_by_ids(h2_id, h3_id)

                quick_refs = _build_quick_refs(meta, md_file, h2_id, h3_id)
                fups_full  = _build_followups(meta, md_file, h2_id, h3_id)

                # показываем followups только на overview и максимум 1
                followups = fups_full[:1] if is_overview else []
                
                # score всегда приводим к float
                score = float(round(float(final_score), 3))
                
                return safe_jsonify({
                    "answer": answer,
                    "quick_replies": quick_refs,   # только suggest_refs
                    "cta": _build_cta(meta),
                    "offer": _pick_relevant_offer(meta),
                    "meta": {
                        "file": final_chunk.get("file"),
                        "h2": h2_val, "h3": h3_val,   # для читаемости
                        "h2_id": h2_id, "h3_id": h3_id,  # для стабильной логики
                        "score": score,
                        "followups": followups,
                        "is_overview": bool(is_overview),
                        "cta_mode": meta.get("cta_mode"),
                        "tags": (list(meta.get("tags")) if isinstance(meta.get("tags"), set) else (meta.get("tags") or []))
                    }
                })
        
        # Приоритетный выбор для ценовых запросов
        is_price_intent = bool(PRICES_RE.search(q or ""))
        if is_price_intent:
            picked = next((it for it in cands if (_doc_type_of(it) or "").lower() == "prices"), None)
            if picked is not None:
                try:
                    final_chunk, final_score = picked
                except Exception:
                    final_chunk, final_score = picked, _score_of(picked)
                log_json(logger, "selection",
                         question=q[:200],
                         original_top_score=(round(float(_score_of(cands[0])),4) if cands else None),
                         rerank_applied=False,
                         chosen=_chunk_info(final_chunk, final_score))
                answer = compose_answer(q, final_chunk)
                
                # Фолбэк для пустого ответа
                if not isinstance(answer, str) or not answer.strip():
                    fallback = (final_chunk.get("text") or "").strip()
                    answer = (fallback[:800] + ("…" if len(fallback) > 800 else "")) or \
                             "Пока не нашёл точный ответ. Можете уточнить вопрос?"
                
                log_json(logger, "Answer generated", file=final_chunk["file"], score=round(float(final_score),3), answer_length=len(answer))
                
                # Сборка UX-данных
                meta = get_doc_meta(os.path.basename(final_chunk.get("file",""))) or {}
                md_file = final_chunk.get("file")
                h2_id, h3_id = _get_ids(final_chunk)
                h2_val = final_chunk.get("h2") or final_chunk.get("h2_id")
                h3_val = final_chunk.get("h3") or final_chunk.get("h3_id")
                is_overview = _is_overview_by_ids(h2_id, h3_id)

                quick_refs = _build_quick_refs(meta, md_file, h2_id, h3_id)
                fups_full  = _build_followups(meta, md_file, h2_id, h3_id)

                # показываем followups только на overview и максимум 1
                followups = fups_full[:1] if is_overview else []
                
                # score всегда приводим к float
                score = float(round(float(final_score), 3))
                
                return safe_jsonify({
                    "answer": answer,
                    "quick_replies": quick_refs,   # только suggest_refs
                    "cta": _build_cta(meta),
                    "offer": _pick_relevant_offer(meta),
                    "meta": {
                        "file": final_chunk.get("file"),
                        "h2": h2_val, "h3": h3_val,   # для читаемости
                        "h2_id": h2_id, "h3_id": h3_id,  # для стабильной логики
                        "score": score,
                        "followups": followups,
                        "is_overview": bool(is_overview),
                        "cta_mode": meta.get("cta_mode"),
                        "tags": (list(meta.get("tags")) if isinstance(meta.get("tags"), set) else (meta.get("tags") or []))
                    }
                })
        
        top = cands[0]
        best = float(top["_score"])

        # если уверенность средняя — дёрнем лёгкий реранк
        use_rerank = 0.45 <= best <= 0.62 and len(cands) >= 2
        if use_rerank:
            log_json(logger, "Applying rerank", original_score=best, candidates_count=len(cands))
            top = llm_rerank(q, cands[:3])
        
        # Логирование выбора финального кандидата
        log_json(logger, "selection",
                 question=q[:200],
                 original_top_score=(round(float(best), 4) if best is not None else None),
                 rerank_applied=bool(use_rerank),
                 chosen=_chunk_info(top, top.get("_score") if isinstance(top, dict) else None))
        
        answer = compose_answer(q, top)  # если хочешь без LLM — просто answer = top["text"]
        
        # Фолбэк для пустого ответа
        if not isinstance(answer, str) or not answer.strip():
            fallback = (top.get("text") or "").strip()
            answer = (fallback[:800] + ("…" if len(fallback) > 800 else "")) or \
                     "Пока не нашёл точный ответ. Можете уточнить вопрос?"
        
        # Логирование результата
        log_json(logger, "Answer generated", 
                 file=top["file"], score=round(float(top.get("_score",0.0)),3), answer_length=len(answer))
        
        # Сборка UX-данных
        meta = get_doc_meta(os.path.basename(top.get("file",""))) or {}
        md_file = top.get("file")
        h2_id, h3_id = _get_ids(top)
        h2_val = top.get("h2") or top.get("h2_id")
        h3_val = top.get("h3") or top.get("h3_id")
        is_overview = _is_overview_by_ids(h2_id, h3_id)

        quick_refs = _build_quick_refs(meta, md_file, h2_id, h3_id)
        fups_full  = _build_followups(meta, md_file, h2_id, h3_id)

        # показываем followups только на overview и максимум 1
        followups = fups_full[:1] if is_overview else []
        
        # score всегда приводим к float
        score = float(round(float(top.get("_score", 0.0)), 3))
        
        return safe_jsonify({
            "answer": answer,
            "quick_replies": quick_refs,   # только suggest_refs
            "cta": _build_cta(meta),
            "offer": _pick_relevant_offer(meta),
            "meta": {
                "file": top.get("file"),
                "h2": h2_val, "h3": h3_val,
                "h2_id": h2_id, "h3_id": h3_id,
                "score": score,
                "followups": followups,
                "is_overview": bool(is_overview),
                "cta_mode": meta.get("cta_mode"),
                "tags": (list(meta.get("tags")) if isinstance(meta.get("tags"), set) else (meta.get("tags") or []))
            }
        })
        
    except Exception as e:
        logger.exception("ask_failed", extra={"q": q, "err": str(e)})
        return safe_jsonify({
            "answer": "Извините, не получилось ответить. Попробуйте переформулировать вопрос.",
            "quick_replies": [], "cta": None, "offer": None, 
            "meta": {"error": "internal"}
        }), 200

# отладка: смотреть кандидатов
@app.get("/__debug/retrieval")
def dbg():
    q = request.args.get("q","")
    c = retrieve(q, topk=5)
    for x in c: x.pop("text", None)
    return jsonify({"q": q, "candidates": c})

# статика (тестер)
@app.get("/static/<path:path>")
def static_files(path):
    return send_from_directory("static", path)

# эндпоинт приёма заявок
@app.post("/lead")
def create_lead():
    try:
        data = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"ok": False, "error": "bad_json"}), 400

    name  = (data.get("name") or "").strip()
    phone = (data.get("phone") or "").strip()
    intent = (data.get("intent") or "").strip()

    # Мини-валидация
    if not phone or len(phone) < 6:
        return jsonify({"ok": False, "error": "invalid_phone"}), 400

    os.makedirs("leads", exist_ok=True)
    ts = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%S")  # без двоеточий
    fname = f"{ts}_{uuid4().hex[:6]}.json"               # небольшой уникальный хвост
    rec = {"ts": ts, "name": name, "phone": phone, "intent": intent}

    with open(os.path.join("leads", fname), "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False)

    return jsonify({"ok": True})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=PORT, debug=True)
