#!/usr/bin/env python3
"""Polishes a machine-translated book into good literary Russian with a local model via Ollama.

Usage:   python fixbook.py book.fb2 [up_to_chapter]      (.epub and .txt work too)
Result:  book.fixed.fb2 next to the source file.
Progress is saved to book.fb2.progress.json, so after a stop just run it again.
book.fb2.glossary.txt (names and terms) is sent with every request; it is built
automatically by the model when missing.
"""
import collections
import hashlib
import html
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

# ---- Settings ----
MODEL = "gemma4:31b-it-qat"   # Ollama model
THINK = False                 # model thinking; None = do not send the parameter (for models without thinking)
OLLAMA_URL = "http://localhost:11434/api/chat"
CHUNK_CHARS = 2000            # chunk size, characters
CONTEXT_CHARS = 600           # how much of the previous edited chunk is given as context
TEMPERATURE = 0.3             # higher = bolder rewriting, lower = closer to the source
NUM_CTX = 8192                # model context window, tokens
MIN_RATIO, MAX_RATIO = 0.5, 1.6   # accepted answer length relative to the source chunk
RETRIES = 1                   # retries after a bad answer
TIMEOUT = 600                 # seconds per chunk

AUTO_GLOSSARY = True          # build the glossary before editing when its file is missing
GLOSSARY_SCAN_CHARS = 0       # characters to read from the start of the book for the glossary; 0 = whole book
GLOSSARY_CHUNK_CHARS = 6000   # chunk size while building the glossary
GLOSSARY_MIN_CHUNKS = 2       # a term is kept if it shows up in at least this many chunks
GLOSSARY_MAX_TERMS = 150      # maximum number of glossary lines

# The prompts below are in Russian on purpose: the model edits Russian text.
# Glossary step 1: list proper names and invented terms found in a fragment.
GLOSSARY_FIND_PROMPT = """Ты помогаешь редактору романа. Из фрагмента выпиши имена собственные \
и придуманные термины: имена и прозвища персонажей, названия мест, организаций, техник, \
титулов, рангов, предметов. По одному в строке, в именительном падеже, с тем же написанием, \
что в тексте. Без пояснений, без нумерации, без обычных слов. Если таких нет — ответь словом: нет."""

# Glossary step 2: merge spelling variants of the same name into one.
GLOSSARY_MERGE_PROMPT = """Ниже список имён и терминов из романа в плохом машинном переводе, \
с числом упоминаний. Один и тот же персонаж или термин может быть записан по-разному: опечатки, \
разные окончания, разная транскрипция. Объедини такие варианты и выбери для каждого одно \
написание — самое частое или самое грамотное.

Формат ответа, по одному в строке:
Правильное написание
или, если были варианты:
Правильное написание (в тексте также: вариант, вариант)

Ничего не добавляй от себя, не пиши пояснений и чисел."""

# Main editing prompt: rewrite a badly machine-translated fragment as good literary Russian.
SYSTEM_PROMPT = """Ты — опытный литературный редактор. Тебе дают фрагмент романа \
в плохом машинном переводе на русский (часто с китайского). Перепиши его хорошим, \
живым литературным русским языком.

Правила:
- Сохраняй смысл, все события, реплики и детали. Ничего не сокращай и не добавляй.
- Исправляй кальки, корявый порядок слов, канцелярит, неверные падежи, род и местоимения.
- Имена, названия и термины оставляй такими же, как в тексте и в глоссарии.
- Прямую речь оформляй по-русски: с тире, без кавычек-лапок.
- Каждый абзац остаётся отдельным абзацем: не объединяй, не разбивай, не переставляй.
- Отвечай только текстом абзацев с их номерами, без пояснений и комментариев."""

# Paragraphs and headings. Groups: 1 = opening tag, 3 = content, 4 = closing tag.
P_XML = re.compile(r"(<(p|h[1-6])(?:\s[^>]*)?(?<!/)>)(.*?)(</\2>)", re.S)
P_TXT = re.compile(r"^()()(.*\S.*)()$", re.M)

# A chapter starts at a short paragraph with the word «Глава» (chapter) and a number;
# a new number means a new chapter.
CHAPTER_RE = re.compile(r"\bглава\s+(\d+)", re.I)
CHAPTER_MAX_LEN = 120
HTML_EXT = (".xhtml", ".html", ".htm")


def read_text(data):
    """Bytes -> (text, encoding)."""
    m = re.match(rb"\s*<\?xml[^>]*encoding=[\"']([\w-]+)", data[:200])
    for enc in ([m.group(1).decode()] if m else []) + ["utf-8", "cp1251"]:
        try:
            return data.decode(enc), enc
        except (UnicodeDecodeError, LookupError):
            pass
    sys.exit("Could not detect the file encoding.")


def plain_of(m, xml):
    """Plain text of a paragraph, or None if the paragraph must be left alone."""
    inner = m.group(3)
    if xml:
        if re.search(r"<(img|image|svg)\b", inner):
            return None
        inner = html.unescape(re.sub(r"<[^>]+>", "", inner))
    text = " ".join(inner.split())
    return text if re.search(r"[^\W\d_]", text) else None


def paragraphs(text, xml):
    pat = P_XML if xml else P_TXT
    return [p for p in (plain_of(m, xml) for m in pat.finditer(text)) if p]


def rebuild(text, xml, fixed):
    """Puts edited paragraphs (the `fixed` iterator) in place of the original ones."""
    def sub(m):
        if plain_of(m, xml) is None:
            return m.group(0)
        new = next(fixed)
        if new is None:               # beyond the chapter limit: keep as is
            return m.group(0)
        return m.group(1) + (html.escape(new, quote=False) if xml else new) + m.group(4)
    return (P_XML if xml else P_TXT).sub(sub, text)


def ask(user, system=None):
    """Asks the model; on an Ollama error exits with a readable message."""
    try:
        return post(user, system or SYSTEM_PROMPT)
    except urllib.error.HTTPError as e:
        sys.exit(f"\nOllama returned error {e.code}: {e.read().decode(errors='replace')}\n"
                 f"Check that the model is downloaded: ollama pull {MODEL}")
    except OSError as e:
        sys.exit(f"\nOllama is not responding ({e}). Start it and run again, progress is saved.")


def post(user, system):
    payload = {
        "model": MODEL,
        "stream": False,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "options": {"temperature": TEMPERATURE, "num_ctx": NUM_CTX},
    }
    if THINK is not None:
        payload["think"] = THINK
    body = json.dumps(payload).encode()
    req = urllib.request.Request(OLLAMA_URL, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.load(r)["message"]["content"]


def parse(answer, n):
    """Model answer -> list of n paragraphs, or None."""
    parts = re.split(r"^\s*\[(\d+)\]\s*", answer, flags=re.M)
    texts = [" ".join(t.split()) for t in parts[2::2]]
    if parts[1::2] != [str(k) for k in range(1, n + 1)] or not all(texts):
        return None
    return texts


def fix_chunk(chunk, tail, glossary):
    """Edited paragraphs of a chunk, or None if the model failed.
    The request text is Russian: glossary, tail of the previous chunk, numbered paragraphs."""
    user = ""
    if glossary:
        user += "Глоссарий (имена и термины пиши именно так):\n" + glossary + "\n\n"
    if tail:
        user += ("Конец предыдущего, уже отредактированного фрагмента "
                 "(только для контекста, не повторяй его):\n" + tail + "\n\n")
    user += (f"Отредактируй абзацы ниже. Их {len(chunk)}, верни ровно столько же, "
             "с теми же номерами в квадратных скобках.\n\n")
    user += "\n".join(f"[{k}] {p}" for k, p in enumerate(chunk, 1))
    src_len = len("".join(chunk))
    for _ in range(RETRIES + 1):
        res = parse(ask(user), len(chunk))
        if res and MIN_RATIO <= len("".join(res)) / src_len <= MAX_RATIO:
            return res
    return None


def glossary_path(src):
    return src.with_name(src.name + ".glossary.txt")


def load(src):
    """Book -> (parts, xml, zip). Parts: name -> (text, encoding); txt and fb2 have one part."""
    ext = src.suffix.lower()
    if ext not in (".txt", ".fb2", ".epub"):
        sys.exit("Only .txt, .fb2 and .epub are supported")
    if ext == ".epub":
        zf = zipfile.ZipFile(src)
        parts = {n: read_text(zf.read(n)) for n in zf.namelist() if n.lower().endswith(HTML_EXT)}
        return parts, True, zf
    return {"": read_text(src.read_bytes())}, ext != ".txt", None


def build_glossary(src, progress=None, stop=None):
    """Builds the book glossary with the model and writes it next to the book.
    Returns the glossary text, or None if stopped. progress(index, total)."""
    parts, xml, _ = load(src)
    chunks, cur, size, total = [], [], 0, 0
    for p in (p for text, _ in parts.values() for p in paragraphs(text, xml)):
        if GLOSSARY_SCAN_CHARS and total >= GLOSSARY_SCAN_CHARS:
            break
        if cur and size + len(p) > GLOSSARY_CHUNK_CHARS:
            chunks.append("\n".join(cur))
            cur, size = [], 0
        cur.append(p)
        size += len(p)
        total += len(p)
    if cur:
        chunks.append("\n".join(cur))

    # Step 1: the model lists names and terms per chunk; count in how many chunks each one appears.
    counts = collections.Counter()
    for i, chunk in enumerate(chunks, 1):
        if stop and stop():
            return None
        counts.update({t for t in map(clean_term, ask(chunk, GLOSSARY_FIND_PROMPT).splitlines()) if t})
        if progress:
            progress(i, len(chunks))
    need = min(GLOSSARY_MIN_CHUNKS, len(chunks))
    top = [(t, n) for t, n in counts.most_common(GLOSSARY_MAX_TERMS) if n >= need]

    # Step 2: one request to merge spelling variants. On a weird answer keep the raw list.
    lines = [t for t, _ in top]
    if top:
        answer = ask("\n".join(f"{t} — {n}" for t, n in top), GLOSSARY_MERGE_PROMPT)
        merged = [t for t in map(clean_term, answer.splitlines()) if t]
        if len(top) / 4 <= len(merged) <= len(top):
            lines = merged
    text = "\n".join(lines)
    glossary_path(src).write_text(text + "\n", encoding="utf-8")
    return text


def clean_term(line):
    term = re.sub(r"^[\s\-–—•*\d.)]+", "", line).strip(" .;,")
    return term if 2 <= len(term) <= 120 and term.lower() != "нет" else ""


def chapter_starts(paras):
    """Indexes of paragraphs where chapters start."""
    starts, last = [], None
    for i, p in enumerate(paras):
        m = CHAPTER_RE.search(p) if len(p) <= CHAPTER_MAX_LEN else None
        if m and m.group(1) != last:
            starts.append(i)
            last = m.group(1)
    return starts


def count_chapters(src):
    parts, xml, _ = load(src)
    return len(chapter_starts([p for text, _ in parts.values() for p in paragraphs(text, xml)]))


def run(src, progress=None, stop=None, upto=None):
    """Edits the book `src`. Returns (result file, unedited chunk count), or None if stopped.
    progress(index, total, unedited, minutes_left, source_chunk, edited_chunk);
    stop() -> True to stop before the next chunk;
    upto: edit only the first `upto` chapters (by order), the rest of the text stays as is."""
    ext = src.suffix.lower()
    parts, xml, zf = load(src)
    out = src.with_name(src.stem + ".fixed" + src.suffix)
    prog_path = src.with_name(src.name + ".progress.json")
    gloss_path = glossary_path(src)
    glossary = gloss_path.read_text(encoding="utf-8").strip() if gloss_path.exists() else ""

    paras = [p for text, _ in parts.values() for p in paragraphs(text, xml)]
    untouched = 0                       # paragraphs at the end of the book left unedited
    if upto:
        starts = chapter_starts(paras)
        if upto < len(starts):
            untouched = len(paras) - starts[upto]
            paras = paras[:starts[upto]]
    chunks, cur, size = [], [], 0
    for p in paras:
        if cur and size + len(p) > CHUNK_CHARS:
            chunks.append(cur)
            cur, size = [], 0
        cur.append(p)
        size += len(p)
    if cur:
        chunks.append(cur)
    print(f"Paragraphs: {len(paras)}, characters: {sum(map(len, paras))}, chunks: {len(chunks)}")

    done = json.loads(prog_path.read_text(encoding="utf-8")) if prog_path.exists() else {}
    fixed, tail, failed, made, t0 = [], "", 0, 0, time.time()
    todo = sum(1 for c in chunks if key_of(c) not in done)
    for i, chunk in enumerate(chunks, 1):
        key = key_of(chunk)
        res = done.get(key)
        if res is None:
            if stop and stop():
                return None
            src_chunk = chunk
            res = fix_chunk(chunk, tail, glossary)
            made += 1
            if res is None:
                failed += 1
                res = chunk
            else:
                done[key] = res
                tmp = prog_path.with_name(prog_path.name + ".tmp")
                tmp.write_text(json.dumps(done, ensure_ascii=False), encoding="utf-8")
                os.replace(tmp, prog_path)
            left = (time.time() - t0) / made * (todo - made) / 60
            if progress:
                progress(i, len(chunks), failed, left, src_chunk, res)
        fixed += res
        tail = "\n".join(res)[-CONTEXT_CHARS:]

    it = iter(fixed + [None] * untouched)
    err = "xmlcharrefreplace" if xml else "replace"
    new = {n: rebuild(text, xml, it).encode(enc, err) for n, (text, enc) in parts.items()}
    if ext == ".epub":
        with zipfile.ZipFile(out, "w") as zout:
            for info in zf.infolist():
                zout.writestr(info, new.get(info.filename) or zf.read(info))
    else:
        out.write_bytes(new[""])
    return out, failed


def main():
    if len(sys.argv) not in (2, 3):
        sys.exit("Usage: python fixbook.py book.fb2 [up_to_chapter]   (.epub and .txt work too)")
    upto = int(sys.argv[2]) if len(sys.argv) == 3 else None

    def show(i, total, failed, left, *_):
        print(f"\r{i}/{total}  ~{left:.0f} min left  unedited: {failed}   ",
              end="", flush=True)

    src = Path(sys.argv[1])
    if AUTO_GLOSSARY and not glossary_path(src).exists():
        build_glossary(src, lambda i, n: print(f"\rGlossary: {i}/{n}   ", end="", flush=True))
        print(f"\nGlossary written: {glossary_path(src)}")
    out, failed = run(src, show, upto=upto)
    print(f"\nDone: {out}")
    if failed:
        print(f"Unedited chunks: {failed}. Run again and only those will be retried.")


def key_of(chunk):
    return hashlib.sha1("\n".join(chunk).encode()).hexdigest()


if __name__ == "__main__":
    main()
