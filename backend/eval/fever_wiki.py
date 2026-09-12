"""
fever_wiki.py — an on-disk index of FEVER's wiki-pages dump.

FEVER ships claims with *pointers* to evidence (page title + sentence index);
the sentence text lives in a separate dump of ~5.4M sentences. The obvious
implementation — load it all into a dict — costs roughly 4-6 GB of RAM, which
is more than many development machines have to spare and fails in the least
helpful possible way: after several minutes of indexing.

This builds a SQLite index instead. Peak memory is one batch of rows
(thousands, not millions), so it runs in tens of megabytes regardless of dump
size, and the index is reusable: build once, then every later conversion
opens it instantly rather than re-parsing the dump.

Schema note: the table is `WITHOUT ROWID` with `(page, sentence_id)` as the
primary key. That stores rows directly in the B-tree keyed by the lookup key,
which is both smaller and faster for this access pattern than a normal table
plus a separate index — the only query this ever runs is an exact-match
lookup on that pair.
"""

import json
import logging
import os
import sqlite3

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
_BATCH = 20_000


class WikiSentenceIndex:
    """Exact-match lookup of (page, sentence_id) -> sentence text."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.execute("PRAGMA query_only = ON")

    # ── lookup ──
    def resolve(self, page: str, sentence_id) -> str:
        try:
            sentence_id = int(sentence_id)
        except (TypeError, ValueError):
            return ""
        row = self._conn.execute(
            "SELECT text FROM sentences WHERE page = ? AND sentence_id = ?",
            (page, sentence_id),
        ).fetchone()
        return row[0] if row else ""

    def as_resolver(self):
        """A `resolve_evidence(page, sentence_id)` callable for
        `eval/adapters.py::from_fever`."""
        return self.resolve

    def __len__(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM sentences").fetchone()[0]

    def close(self):
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ── build ──
    @staticmethod
    def build(wiki_dir: str, db_path: str, force: bool = False, progress=None) -> dict:
        """Stream the dump into a SQLite index. Returns a build report.

        Existing index is reused unless `force` — re-parsing a multi-gigabyte
        dump because a command was run twice is a waste, and the dump is
        immutable in practice.
        """
        if os.path.exists(db_path) and not force:
            with WikiSentenceIndex(db_path) as idx:
                n = len(idx)
            return {"status": "reused", "path": db_path, "sentences": n, "files": 0}

        if os.path.exists(db_path):
            os.remove(db_path)

        files = sorted(
            os.path.join(wiki_dir, name)
            for name in os.listdir(wiki_dir)
            if name.endswith((".jsonl", ".json"))
        )
        if not files:
            raise FileNotFoundError(f"No .jsonl/.json files found in {wiki_dir}")

        os.makedirs(os.path.dirname(os.path.abspath(db_path)) or ".", exist_ok=True)
        conn = sqlite3.connect(db_path)
        try:
            # Bulk-load pragmas: this file is a rebuildable cache, so trading
            # crash-durability for speed is the right call here (and only here).
            conn.execute("PRAGMA journal_mode = OFF")
            conn.execute("PRAGMA synchronous = OFF")
            conn.execute(
                "CREATE TABLE sentences ("
                " page TEXT NOT NULL, sentence_id INTEGER NOT NULL, text TEXT NOT NULL,"
                " PRIMARY KEY (page, sentence_id)) WITHOUT ROWID"
            )
            conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")

            total, skipped_lines, batch = 0, 0, []
            for path in files:
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            page = json.loads(line)
                        except json.JSONDecodeError:
                            skipped_lines += 1
                            continue
                        page_id = page.get("id")
                        if not page_id:
                            skipped_lines += 1
                            continue
                        for entry in (page.get("lines") or "").split("\n"):
                            parts = entry.split("\t")
                            if len(parts) < 2 or not parts[0].isdigit():
                                continue
                            text = parts[1].strip()
                            if not text:
                                continue
                            batch.append((page_id, int(parts[0]), text))
                            if len(batch) >= _BATCH:
                                # INSERT OR IGNORE: a dump can repeat a page,
                                # and a duplicate key must not abort the build.
                                conn.executemany(
                                    "INSERT OR IGNORE INTO sentences VALUES (?, ?, ?)", batch
                                )
                                total += len(batch)
                                batch.clear()
                if progress:
                    progress(os.path.basename(path), total)

            if batch:
                conn.executemany("INSERT OR IGNORE INTO sentences VALUES (?, ?, ?)", batch)
                total += len(batch)

            conn.executemany(
                "INSERT OR REPLACE INTO meta VALUES (?, ?)",
                [("schema_version", str(SCHEMA_VERSION)),
                 ("source_dir", os.path.abspath(wiki_dir)),
                 ("files", str(len(files)))],
            )
            conn.commit()
            actual = conn.execute("SELECT COUNT(*) FROM sentences").fetchone()[0]
        finally:
            conn.close()

        if skipped_lines:
            logger.warning("Skipped %d unparseable wiki lines while indexing", skipped_lines)
        return {
            "status": "built", "path": db_path, "sentences": actual,
            "rows_seen": total, "files": len(files), "skipped_lines": skipped_lines,
        }
