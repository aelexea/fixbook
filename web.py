#!/usr/bin/env python3
"""Local web page for fixbook. Everything stays on this machine.

Run:  python web.py   and open http://127.0.0.1:8765
Books and results are kept in the `books` folder next to this file.
"""
import json
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

import fixbook

# ---- Settings ----
PORT = 8765
MAX_UPLOAD = 200 * 1024 * 1024   # book size limit, bytes

ROOT = Path(__file__).resolve().parent
BOOKS = ROOT / "books"
EXTS = (".txt", ".fb2", ".epub")
OLLAMA = fixbook.OLLAMA_URL.rsplit("/api/", 1)[0]
DEFAULTS = {"model": fixbook.MODEL, "temperature": fixbook.TEMPERATURE}

job = {"book": None, "running": False}
stop_flag = threading.Event()


def ollama(path, payload=None):
    data = json.dumps(payload).encode() if payload else None
    req = urllib.request.Request(OLLAMA + path, data, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.load(r)


def models():
    """Downloaded models, or None if Ollama is not responding."""
    try:
        return sorted(m["name"] for m in ollama("/api/tags")["models"])
    except (OSError, KeyError, ValueError):
        return None


def can_think(model):
    try:
        return "thinking" in ollama("/api/show", {"model": model}).get("capabilities", [])
    except (OSError, ValueError):
        return False


def book_path(name):
    p = BOOKS / Path(name or "").name
    if p.suffix.lower() not in EXTS:
        raise ValueError("Only .txt, .fb2 and .epub are supported")
    return p


def out_path(src):
    return src.with_name(src.stem + ".fixed" + src.suffix)


def gloss_path(src):
    return src.with_name(src.name + ".glossary.txt")


chapters_cache = {}   # (name, mtime) -> chapter count


def chapters(p):
    key = (p.name, p.stat().st_mtime)
    if key not in chapters_cache:
        try:
            chapters_cache[key] = fixbook.count_chapters(p)
        except (SystemExit, Exception):
            chapters_cache[key] = 0
    return chapters_cache[key]


def books():
    res = []
    for p in sorted(BOOKS.glob("*"), key=lambda p: -p.stat().st_mtime):
        if (p.suffix.lower() in EXTS and not p.stem.endswith(".fixed")
                and not p.name.endswith(".glossary.txt")):
            g = gloss_path(p)
            res.append({
                "name": p.name,
                "size": p.stat().st_size,
                "chapters": chapters(p),
                "result": out_path(p).exists(),
                "started": p.with_name(p.name + ".progress.json").exists(),
                "glossary": g.read_text(encoding="utf-8") if g.exists() else "",
            })
    return res


def worker(src, model, temperature, only_glossary, upto):
    fixbook.MODEL = model
    fixbook.TEMPERATURE = temperature
    fixbook.THINK = False if can_think(model) else None

    def progress(i, total, failed, left, before, after):
        job.update(done=i, total=total, failed=failed, left=round(left),
                   before=before, after=after)

    try:
        # Glossary: on request, or automatically when the book has none yet.
        if only_glossary or (fixbook.AUTO_GLOSSARY and not gloss_path(src).exists()):
            job["stage"] = "glossary"
            text = fixbook.build_glossary(src, lambda i, n: job.update(done=i, total=n),
                                          stop_flag.is_set)
            if text is None:
                job["stopped"] = True
                return
            job["gloss_ver"] = time.time()   # the page refreshes its glossary field when this changes
            if only_glossary:
                job["finished"] = True
                return
        job.update(stage="edit", done=0, total=0)
        res = fixbook.run(src, progress, stop_flag.is_set, upto)
        if res:
            job.update(finished=True, failed=res[1])
        else:
            job["stopped"] = True
    except SystemExit as e:
        job["error"] = str(e.code).strip()
    except Exception as e:
        job["error"] = f"{type(e).__name__}: {e}"
    finally:
        job["running"] = False


class Handler(BaseHTTPRequestHandler):
    def send(self, code, body, ctype="application/json", extra=None):
        if not isinstance(body, bytes):
            body = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def local(self):
        """Only serve requests addressed to this local page."""
        if self.headers.get("Host") in (f"127.0.0.1:{PORT}", f"localhost:{PORT}"):
            return True
        self.send(403, {"error": "forbidden"})
        return False

    def do_GET(self):
        if not self.local():
            return
        url = urlparse(self.path)
        try:
            if url.path == "/":
                self.send(200, (ROOT / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/api/state":
                self.send(200, {"models": models(), "defaults": DEFAULTS,
                                "books": books(), "job": job})
            elif url.path == "/api/download":
                out = out_path(book_path(parse_qs(url.query).get("book", [""])[0]))
                self.send(200, out.read_bytes(), "application/octet-stream",
                          {"Content-Disposition": "attachment; filename*=UTF-8''" + quote(out.name)})
            else:
                self.send(404, {"error": "not found"})
        except (ValueError, OSError) as e:
            self.send(400, {"error": str(e)})

    def do_POST(self):
        if not self.local():
            return
        url = urlparse(self.path)
        size = int(self.headers.get("Content-Length") or 0)
        try:
            if url.path == "/api/upload":
                if size > MAX_UPLOAD:
                    raise ValueError("The file is too large")
                dst = book_path(parse_qs(url.query).get("name", [""])[0])
                dst.write_bytes(self.rfile.read(size))
                self.send(200, {"name": dst.name})
            elif url.path == "/api/start":
                req = json.loads(self.rfile.read(size))
                src = book_path(req["book"])
                if job["running"]:
                    raise ValueError("Another book is being processed")
                if not src.exists():
                    raise ValueError("Book not found")
                only_glossary = bool(req.get("only_glossary"))
                text = req.get("glossary", "").strip()
                if only_glossary:
                    pass                      # the builder will overwrite the glossary file
                elif text:
                    gloss_path(src).write_text(text + "\n", encoding="utf-8")
                else:
                    gloss_path(src).unlink(missing_ok=True)
                stop_flag.clear()
                job.clear()
                upto = int(req.get("upto") or 0) or None      # empty or 0 = whole book
                job.update(book=src.name, running=True, model=req["model"], upto=upto)
                threading.Thread(target=worker, daemon=True, args=(
                    src, req["model"], float(req["temperature"]), only_glossary, upto)).start()
                self.send(200, {"ok": True})
            elif url.path == "/api/stop":
                stop_flag.set()
                self.send(200, {"ok": True})
            elif url.path == "/api/quit":
                self.send(200, {"ok": True})
                threading.Thread(target=self.server.shutdown).start()
            else:
                self.send(404, {"error": "not found"})
        except (ValueError, KeyError, OSError) as e:
            self.send(400, {"error": str(e)})

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    BOOKS.mkdir(exist_ok=True)
    print(f"Fixbook: http://127.0.0.1:{PORT}")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
