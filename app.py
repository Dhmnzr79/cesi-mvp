import os, json, numpy as np
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory
from openai import OpenAI

load_dotenv()
client = OpenAI()
EMB_MODEL = os.getenv("MODEL_EMBED","text-embedding-3-small")
CHAT_MODEL= os.getenv("MODEL_CHAT","gpt-4o-mini")
PORT      = int(os.getenv("PORT", "9000"))

# загрузка индекса
CORPUS = [json.loads(x) for x in open("data/corpus.jsonl", encoding="utf-8")]
EMB    = np.load("data/embeddings.npy")  # уже нормализованы
assert EMB.shape[0] == len(CORPUS)

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
    for i in idx:
        c = CORPUS[int(i)]
        key = (c["file"], c.get("h2_id") or c.get("h2"), c.get("h3_id") or c.get("h3"))
        if key in seen: continue
        seen.add(key); c2 = dict(c); c2["_score"]=float(sims[int(i)])
        out.append(c2)
        if len(out) == topk: break
    return out

def llm_rerank(q, cands):  # cands: list[dict], len<=3
    # Сведём к коротким отрывкам
    prompt = "Выбери самый уместный фрагмент для ответа на вопрос пользователя. Ответи номером 1, 2 или 3."
    msgs = [{"role":"system","content":"Ты выбираешь лучший фрагмент."},
            {"role":"user","content": f"{prompt}\n\nВопрос: {q}\n\n1) {cands[0]['text'][:600]}\n\n2) {cands[1]['text'][:600] if len(cands)>1 else ''}\n\n3) {cands[2]['text'][:600] if len(cands)>2 else ''}"}]
    try:
        out = client.chat.completions.create(model=CHAT_MODEL, messages=msgs, temperature=0)
        n = "".join([ch for ch in out.choices[0].message.content if ch.isdigit()])[:1]
        idx = int(n)-1
        return cands[idx] if 0 <= idx < len(cands) else cands[0]
    except Exception:
        return cands[0]

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

app = Flask(__name__, static_folder="static")

@app.post("/ask")
def ask():
    data = request.get_json(force=True)
    q = (data.get("q") or "").strip()
    if not q: return jsonify({"answer":"Уточните вопрос."})
    cands = retrieve(q, topk=3)
    if not cands:
        return jsonify({"answer":"Пока не нашёл подходящий ответ. Сформулируйте вопрос иначе."})
    top = cands[0]
    best = float(top["_score"])

    # если уверенность средняя — дёрнем лёгкий реранк
    if 0.45 <= best <= 0.62 and len(cands) >= 2:
        top = llm_rerank(q, cands[:3])
    
    answer = compose_answer(q, top)  # если хочешь без LLM — просто answer = top["text"]
    return jsonify({
        "answer": answer,
        "meta": {
            "file": top["file"], "h2": top["h2"], "h3": top["h3"], "score": round(top["_score"],3),
            "followups": top.get("followups", [])[:2]
        }
    })

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

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=PORT, debug=True)
