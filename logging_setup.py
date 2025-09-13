# logging_setup.py
import json, logging, os, sys, uuid
from logging.handlers import RotatingFileHandler
from datetime import datetime

LOG_DIR = os.getenv("BOT_LOG_DIR", "logs")
LOG_FILE = os.path.join(LOG_DIR, os.getenv("BOT_LOG_FILE", "app.jsonl"))

SENSITIVE_KEYS = ("api_key","apikey","token","secret","authorization","password")

def _sanitize(d):
    if not isinstance(d, dict): return d
    clean = {}
    for k, v in d.items():
        if isinstance(k, str) and any(s in k.lower() for s in SENSITIVE_KEYS):
            clean[k] = "***"
        else:
            clean[k] = v
    return clean

class JsonLineFormatter(logging.Formatter):
    def format(self, record):
        base = {
            "ts": datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "extra_data", None)
        if isinstance(extra, dict):
            base.update(extra)
        return json.dumps(base, ensure_ascii=False)

def get_logger(name="bot"):
    os.makedirs(LOG_DIR, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fh = RotatingFileHandler(LOG_FILE, maxBytes=10_000_000, backupCount=5, encoding="utf-8")
    ch = logging.StreamHandler(sys.stdout)
    fmt = JsonLineFormatter()
    fh.setFormatter(fmt); ch.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(ch)
    return logger

def make_request_context(session_id=None):
    return {
        "session_id": session_id or str(uuid.uuid4()),
        "request_id": str(uuid.uuid4()),
        "app_version": os.getenv("APP_VERSION", "dev"),
        "env": os.getenv("APP_ENV", "local"),
    }

def log_json(logger, message, **fields):
    logger.info(message, extra={"extra_data": _sanitize(fields)})