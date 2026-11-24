# Fixbook

Polish machine-translated books into readable literary Russian with a local model.
Everything runs on your own machine through [Ollama](https://ollama.com): no cloud, no accounts,
the book never leaves your computer.

Built for web novels that are only available as raw machine translation. You only need the
translated text, not the original.

## What it does

- Takes a book in `.txt`, `.fb2` or `.epub` and writes `book.fixed.*` in the same format.
- Splits the text into small chunks by paragraph and asks the model to rewrite each one,
  passing the end of the previous edited chunk as context.
- Builds a glossary of names and terms with the same model before editing, so characters
  and terms keep one spelling through the whole book. You can edit it by hand.
- Can edit only the first N chapters and leave the rest untouched; raise the number later
  and it continues where it stopped.
- Saves progress after every chunk. Stop it any time and run again.
- Rejects bad answers (lost paragraphs, text that shrank or grew too much), retries once,
  then keeps the original chunk and reports how many were left unedited.

## Requirements

- Python 3.10 or newer, standard library only.
- Ollama running locally with a model that writes good Russian.
  The default is `gemma4:31b-it-qat`, which fits in 24 GB of VRAM:

```bash
ollama pull gemma4:31b-it-qat
```

## Web page

```bash
python web.py
```

Then open <http://127.0.0.1:8765>. Drop a book, pick a model, optionally set
"Edit up to chapter" and review the glossary, press Start. The page shows progress,
time left and a before/after view of the latest chunk. Books and results are stored
in the `books` folder.

The server only listens on `127.0.0.1` and keeps working after the tab is closed,
so a long book is not interrupted. Use "Shut down server" at the bottom of the page to stop it.

`fixbook-app.sh` starts the server if needed and opens the page in its own window.
To get a launcher icon on Linux, create `~/.local/share/applications/fixbook.desktop`:

```ini
[Desktop Entry]
Type=Application
Name=Fixbook
Exec=bash /path/to/fixbook/fixbook-app.sh
Icon=/path/to/fixbook/icon.svg
Terminal=false
Categories=Utility;
```

## Command line

```bash
python fixbook.py book.fb2
```

Edit only the first 150 chapters:

```bash
python fixbook.py book.fb2 150
```

Files created next to the book:

| File | Purpose |
|---|---|
| `book.fixed.fb2` | the result |
| `book.fb2.glossary.txt` | glossary, one name or term per line; edit freely |
| `book.fb2.progress.json` | finished chunks; delete it to redo the book from scratch |

## Settings

All settings are constants at the top of `fixbook.py`:

| Constant | Meaning |
|---|---|
| `MODEL` | Ollama model tag |
| `THINK` | `False` turns model thinking off; `None` for models without thinking |
| `TEMPERATURE` | higher rewrites more boldly, lower stays closer to the source |
| `CHUNK_CHARS` | chunk size; lower it if the model loses paragraphs |
| `CONTEXT_CHARS` | how much of the previous chunk is passed as context |
| `SYSTEM_PROMPT` | the editing instructions given to the model |
| `AUTO_GLOSSARY` | build the glossary automatically when it is missing |
| `GLOSSARY_SCAN_CHARS` | how much of the book to read for the glossary, `0` = all of it |
| `GLOSSARY_MIN_CHUNKS`, `GLOSSARY_MAX_TERMS` | glossary filtering |
| `CHAPTER_RE` | how a chapter heading is recognised |

The port of the web page is `PORT` in `web.py` (and the same number in `fixbook-app.sh`).

## Limitations

- The model only sees the translation. It makes the text smooth and consistent, but it cannot
  recover meaning the machine translator got wrong.
- The prompts and the chapter pattern (`Глава 12`) are written for Russian text. For another
  target language, rewrite the prompts and `CHAPTER_RE` in `fixbook.py`.
- Inline formatting inside paragraphs (italic, bold) is dropped; images and the rest of the
  markup are kept as they are.
- One book is processed at a time. A large novel takes hours on a single GPU.
