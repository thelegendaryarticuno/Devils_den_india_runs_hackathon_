from __future__ import annotations

import csv
import html
import io
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

from src.features import extract_features
from src.reasoning import build_reasoning
from src.scoring import ScoredCandidate, final_csv_scores, score_candidate, sort_candidates


ROOT = Path(__file__).parent
SAMPLE_PATH = ROOT / "sample_candidates.json"
MAX_CANDIDATES = 100
DEFAULT_TOP_K = 20


def load_records(raw_text: str | None = None) -> list[dict[str, Any]]:
    text = (raw_text or "").strip()
    if not text:
        text = SAMPLE_PATH.read_text(encoding="utf-8")

    if text.startswith("[") or text.startswith("{"):
        payload = json.loads(text)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            return [payload]
        raise ValueError("JSON input must be a candidate object or a list of candidates.")

    records: list[dict[str, Any]] = []
    for line in text.splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def rank_records(records: list[dict[str, Any]], top_k: int) -> list[list[str]]:
    records = records[:MAX_CANDIDATES]
    scored: list[ScoredCandidate] = []
    for candidate in records:
        features = extract_features(candidate)
        scored.append(
            ScoredCandidate(
                candidate_id=features.candidate_id,
                score=score_candidate(features),
                features=features,
            )
        )

    top_k = max(1, min(top_k, len(scored), MAX_CANDIDATES))
    ranked = sort_candidates(scored)[:top_k]
    display_scores = final_csv_scores(ranked)

    rows: list[list[str]] = []
    for rank, (item, display_score) in enumerate(zip(ranked, display_scores), start=1):
        rows.append(
            [
                item.candidate_id,
                str(rank),
                f"{display_score:.6f}",
                build_reasoning(item.features, rank),
            ]
        )
    return rows


def rows_to_csv(rows: list[list[str]]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["candidate_id", "rank", "score", "reasoning"])
    writer.writerows(rows)
    return buffer.getvalue()


def render_page(rows: list[list[str]] | None = None, error: str = "", top_k: int = DEFAULT_TOP_K) -> str:
    rows = rows or []
    table_rows = "\n".join(
        "<tr>"
        + "".join(f"<td>{html.escape(cell)}</td>" for cell in row)
        + "</tr>"
        for row in rows
    )
    csv_text = html.escape(rows_to_csv(rows)) if rows else ""
    status = (
        f"<p class='ok'>Ranked {len(rows)} candidates. Copy or download from the CSV box below.</p>"
        if rows
        else "<p>Using the bundled sample candidates by default. Paste JSON or JSONL to test a custom sample.</p>"
    )
    if error:
        status = f"<p class='error'>{html.escape(error)}</p>"

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Natanyx Redrob Ranker</title>
  <style>
    body {{ margin: 0; font-family: Inter, system-ui, -apple-system, Segoe UI, sans-serif; background: #0f172a; color: #e5e7eb; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 36px 20px; }}
    h1 {{ margin: 0 0 8px; font-size: 32px; }}
    p {{ color: #b6c0d4; line-height: 1.55; }}
    form {{ display: grid; gap: 14px; margin: 24px 0; }}
    textarea, input {{ width: 100%; box-sizing: border-box; background: #020617; color: #e5e7eb; border: 1px solid #334155; border-radius: 8px; padding: 12px; font: inherit; }}
    textarea {{ min-height: 170px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 13px; }}
    button {{ width: fit-content; background: #38bdf8; color: #082f49; border: 0; border-radius: 8px; padding: 11px 18px; font-weight: 700; cursor: pointer; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 20px; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid #243244; padding: 10px; text-align: left; vertical-align: top; }}
    th {{ color: #f8fafc; background: #111c30; position: sticky; top: 0; }}
    .ok {{ color: #86efac; }}
    .error {{ color: #fca5a5; }}
    .csv {{ margin-top: 20px; }}
  </style>
</head>
<body>
<main>
  <h1>Natanyx Redrob Ranker</h1>
  <p>Deterministic CPU-only sandbox for ranking Redrob Senior AI Engineer candidates. No network calls, no hosted LLM APIs, and no GPU inference during ranking.</p>
  <form method="post" action="/">
    <label>Rows to rank, max 100
      <input name="top_k" type="number" min="1" max="100" value="{top_k}">
    </label>
    <label>Candidate JSON or JSONL
      <textarea name="candidates" placeholder="Leave empty to run the bundled sample candidates."></textarea>
    </label>
    <button type="submit">Rank candidates</button>
  </form>
  {status}
  <table>
    <thead><tr><th>candidate_id</th><th>rank</th><th>score</th><th>reasoning</th></tr></thead>
    <tbody>{table_rows}</tbody>
  </table>
  <div class="csv">
    <label>CSV output
      <textarea readonly>{csv_text}</textarea>
    </label>
  </div>
</main>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_text("ok\n", content_type="text/plain")
            return
        rows = rank_records(load_records(), DEFAULT_TOP_K)
        self._send_text(render_page(rows=rows))

    def do_POST(self) -> None:
        length = int(self.headers.get("content-length", "0"))
        body = self.rfile.read(length).decode("utf-8", errors="replace")
        form = parse_qs(body)
        raw_candidates = form.get("candidates", [""])[0]
        top_k_text = form.get("top_k", [str(DEFAULT_TOP_K)])[0]
        try:
            top_k = int(top_k_text)
            rows = rank_records(load_records(raw_candidates), top_k)
            self._send_text(render_page(rows=rows, top_k=top_k))
        except Exception as exc:
            self._send_text(render_page(error=str(exc)))

    def log_message(self, fmt: str, *args: Any) -> None:
        print(fmt % args)

    def _send_text(self, text: str, content_type: str = "text/html; charset=utf-8") -> None:
        payload = text.encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main() -> None:
    host = "0.0.0.0"
    port = int(os.environ.get("PORT", os.environ.get("GRADIO_SERVER_PORT", "7860")))
    print(f"Natanyx Redrob Ranker listening on http://{host}:{port}")
    ThreadingHTTPServer((host, port), Handler).serve_forever()


if __name__ == "__main__":
    main()
