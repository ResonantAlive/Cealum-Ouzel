"""Low-overhead live loss dashboard for local training runs.

The trainer writes a compact JSONL record only at its existing logging cadence.
Rendering and HTTP serving run in daemon threads, so neither matplotlib nor a
browser can delay the CUDA training stream.
"""

from __future__ import annotations

import json
import math
import queue
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable


_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Training dashboard</title><style>
body{margin:0 auto;max-width:1100px;padding:18px;font:15px system-ui,sans-serif;background:#fafafa;color:#202124}
h1{font-size:20px;margin:0 0 6px}p{color:#667085;margin:0 0 16px}img{display:block;width:100%;background:#fff;border:1px solid #e5e7eb;border-radius:8px;margin:14px 0}
</style></head><body><h1>Training loss</h1><p>Updates automatically every 5 seconds.</p>
<img src="loss.png" alt="Loss and validation loss"><img src="ppl.png" alt="PPL and validation PPL">
<script>setInterval(()=>document.querySelectorAll('img').forEach((x)=>x.src=x.src.split('?')[0]+'?t='+Date.now()),5000)</script>
</body></html>"""


class TrainingDashboard:
    def __init__(
        self,
        output_dir: Path,
        *,
        enabled: bool,
        host: str = "127.0.0.1",
        port: int = 6006,
        render_interval_seconds: float = 5.0,
        history_metrics: Iterable[tuple[Path, str]] = (),
        phase: str = "current",
    ) -> None:
        self.enabled = enabled
        self.output_dir = Path(output_dir) / "dashboard"
        self.metrics_path = self.output_dir / "metrics.jsonl"
        self._records: list[dict[str, Any]] = []
        self._history_metrics = [(Path(path), str(label)) for path, label in history_metrics]
        self._phase = str(phase)
        self._render_requests: queue.Queue[None] = queue.Queue(maxsize=1)
        # Matplotlib owns the GIL while it transforms and rasterizes the
        # complete history.  Requesting a redraw for every optimizer step can
        # therefore delay CPU kernel submission and data prefetch, even though
        # rendering happens in a daemon thread.  The browser already refreshes
        # at five seconds, so coalesce requests to the same cadence.
        self._render_interval_seconds = max(0.0, float(render_interval_seconds))
        self._next_render_request_at = 0.0
        self._closed = False
        self._server: ThreadingHTTPServer | None = None
        if not enabled:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "index.html").write_text(_HTML, encoding="utf-8")
        self._load_history()
        self._load_existing()
        self._renderer = threading.Thread(target=self._render_loop, daemon=True)
        self._renderer.start()
        self._request_render()
        try:
            handler = lambda *args, **kwargs: SimpleHTTPRequestHandler(
                *args, directory=str(self.output_dir), **kwargs
            )
            self._server = ThreadingHTTPServer((host, port), handler)
            threading.Thread(target=self._server.serve_forever, daemon=True).start()
            print(f"loss dashboard: http://{host}:{port} ({self.output_dir})", flush=True)
        except OSError as exc:
            print(f"loss dashboard disabled: cannot bind {host}:{port}: {exc}", flush=True)

    @staticmethod
    def _read_records(path: Path, *, phase: str) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        records: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                value = json.loads(line)
                if isinstance(value, dict):
                    value = dict(value)
                    value.setdefault("_dashboard_phase", phase)
                    records.append(value)
            except json.JSONDecodeError:
                continue
        return records

    def _load_history(self) -> None:
        histories: list[list[dict[str, Any]]] = []
        for path, phase in self._history_metrics:
            if path.resolve() == self.metrics_path.resolve():
                continue
            histories.append(self._read_records(path, phase=phase))

        # Successive continuations commonly overlap by a few logging steps.
        # Keep the later run at each hand-off: it owns the optimizer/sampler
        # state that actually continued. This produces one monotonic history
        # rather than plotting two divergent lines over the overlap.
        for index, records in enumerate(histories):
            later_first_tokens = [
                self._finite(record, "tokens")
                for later in histories[index + 1 :]
                for record in later[:1]
                if self._finite(record, "tokens") is not None
            ]
            cutoff = min(later_first_tokens) if later_first_tokens else None
            if cutoff is not None:
                records = [
                    record
                    for record in records
                    if (token := self._finite(record, "tokens")) is None or token < cutoff
                ]
            self._records.extend(records)

    def _load_existing(self) -> None:
        if not self.metrics_path.exists():
            return
        self._records.extend(self._read_records(self.metrics_path, phase=self._phase))

    @classmethod
    def _json_value(cls, value: Any) -> Any:
        if value is None or isinstance(value, (int, float, str)):
            return value
        if isinstance(value, (list, tuple)):
            return [cls._json_value(item) for item in value]
        return None

    def log(self, **record: Any) -> None:
        if not self.enabled or self._closed:
            return
        sanitized = {
            key: self._json_value(value)
            for key, value in record.items()
            if value is None
            or isinstance(value, (int, float, str, list, tuple))
        }
        sanitized["_dashboard_phase"] = self._phase
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(sanitized, ensure_ascii=False, separators=(",", ":")) + "\n")
        self._records.append(sanitized)
        self._request_render()

    def _request_render(self, *, force: bool = False) -> None:
        if not force and self._render_interval_seconds:
            now = time.monotonic()
            if now < self._next_render_request_at:
                return
            self._next_render_request_at = now + self._render_interval_seconds
        try:
            self._render_requests.put_nowait(None)
        except queue.Full:
            pass

    @staticmethod
    def _finite(record: dict[str, Any], name: str) -> float | None:
        value = record.get(name)
        if isinstance(value, (int, float)) and math.isfinite(value):
            return float(value)
        return None

    def _render_loop(self) -> None:
        while True:
            self._render_requests.get()
            try:
                self._render()
            except Exception as exc:  # Rendering must never terminate training.
                print(f"loss dashboard render warning: {exc}", flush=True)
            # ``close`` may race an in-flight render. Finish the most recently
            # queued snapshot before exiting so both loss.png and ppl.png are
            # atomically published even for a very short smoke run.
            if self._closed and self._render_requests.empty():
                return

    def _render(self) -> None:
        # Import only on the background renderer. The core trainer therefore
        # remains usable if visualization dependencies are intentionally absent.
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # A restart immediately after a periodic checkpoint may replay a few
        # already logged steps. Keep the newer record at an identical phase and
        # cumulative-token position, so the visual history remains a single
        # line without editing either append-only JSONL source.
        unique_records: dict[tuple[str, float], dict[str, Any]] = {}
        unkeyed_records: list[dict[str, Any]] = []
        for record in self._records:
            token = self._finite(record, "tokens")
            if token is None:
                unkeyed_records.append(record)
            else:
                unique_records[(str(record.get("_dashboard_phase", "")), token)] = record
        records = sorted(
            [*unique_records.values(), *unkeyed_records],
            key=lambda item: self._finite(item, "tokens")
            if self._finite(item, "tokens") is not None
            else self._finite(item, "step")
            if self._finite(item, "step") is not None
            else float("inf"),
        )
        if not records:
            return
        x = [self._finite(item, "tokens") or self._finite(item, "step") or index + 1 for index, item in enumerate(records)]
        x_label = "Processed tokens" if any(self._finite(item, "tokens") is not None for item in records) else "Step"
        x_scale = 1e9 if x_label == "Processed tokens" and max(x) >= 1e9 else 1.0
        x = [value / x_scale for value in x]
        if x_scale != 1.0:
            x_label += " (B)"
        phase_boundaries = [
            x[index]
            for index in range(1, len(records))
            if records[index].get("_dashboard_phase") != records[index - 1].get("_dashboard_phase")
        ]
        self._plot(plt, records, x, x_label, "loss", "val_loss", "Loss", self.output_dir / "loss.png", phase_boundaries)
        self._plot(plt, records, x, x_label, "ppl", "val_ppl", "Perplexity", self.output_dir / "ppl.png", phase_boundaries)

    def _plot(
        self,
        plt,
        records: list[dict[str, Any]],
        x: list[float],
        x_label: str,
        train_key: str,
        val_key: str,
        y_label: str,
        path: Path,
        phase_boundaries: list[float],
    ) -> None:
        train = [self._finite(item, train_key) for item in records]
        val = [self._finite(item, val_key) for item in records]
        figure, axis = plt.subplots(figsize=(10, 5), dpi=150, layout="constrained")
        train_x = [position for position, value in zip(x, train) if value is not None]
        train_y = [value for value in train if value is not None]
        val_x = [position for position, value in zip(x, val) if value is not None]
        val_y = [value for value in val if value is not None]
        if train_y:
            axis.plot(train_x, train_y, color="#d62728", linewidth=1.5, label=y_label)
        if val_y:
            axis.plot(val_x, val_y, color="#1f77b4", linewidth=2.0, marker="o", markersize=3, label=f"Val {y_label}")
        for boundary in phase_boundaries:
            axis.axvline(boundary, color="#98a2b3", linewidth=0.8, linestyle="--", alpha=0.75, zorder=0)
        axis.set(xlabel=x_label, ylabel=y_label)
        axis.grid(True, alpha=0.25)
        if train_y or val_y:
            axis.legend(frameon=False)
        temporary = path.with_suffix(".tmp.png")
        figure.savefig(temporary, format="png")
        temporary.replace(path)
        plt.close(figure)

    def close(self) -> None:
        if not self.enabled or self._closed:
            return
        self._closed = True
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        # Request one final render and wait for it. This only runs at shutdown;
        # the normal training hot path remains fully asynchronous.
        self._request_render(force=True)
        self._renderer.join(timeout=30.0)
        if self._renderer.is_alive():
            print("loss dashboard final render timed out", flush=True)
