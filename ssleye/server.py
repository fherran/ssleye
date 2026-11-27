"""Dashboard server."""
import json
import mimetypes
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

STATIC = Path(__file__).resolve().parent / "static"
STREAMS = {"scalars", "snapshots", "alerts", "stream", "samples", "downstream"}


def _read_json(p):
    for _ in range(5):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, PermissionError):
            time.sleep(0.05)
        except FileNotFoundError:
            return None
    return None


def _inside(base, p):
    try:
        p.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def make_handler(root):
    root = Path(root)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json", cache=False):
            if isinstance(body, (dict, list)):
                body = json.dumps(body, separators=(",", ":")).encode("utf-8")
            elif isinstance(body, str):
                body = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "max-age=3600" if cache else "no-store")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

        def _file(self, p, cache=False):
            if not p.is_file():
                return self._send(404, {"error": "not found"})
            ctype = mimetypes.guess_type(str(p))[0] or "application/octet-stream"
            if p.suffix == ".js":
                ctype = "text/javascript"
            self._send(200, p.read_bytes(), ctype, cache)

        def do_GET(self):
            u = urlparse(self.path)
            parts = [unquote(x) for x in u.path.strip("/").split("/") if x]
            q = parse_qs(u.query)
            if not parts or parts == ["index.html"]:
                return self._file(STATIC / "pulse.html")   # Pulse is the dashboard
            if parts[0] == "static" and len(parts) >= 2:
                p = STATIC.joinpath(*parts[1:])
                return self._file(p) if _inside(STATIC, p) else self._send(403, {"error": "forbidden"})
            if parts[0] != "api" or len(parts) < 2:
                return self._send(404, {"error": "not found"})
            if parts[1] == "runs":
                return self._send(200, self._runs())
            if parts[1] == "run" and len(parts) >= 5:
                run = root / parts[2] / parts[3]
                if not _inside(root, run) or not run.is_dir():
                    return self._send(404, {"error": "no such run"})
                what = parts[4]
                if what == "meta":
                    meta = _read_json(run / "meta.json") or {}
                    meta["probe"] = _read_json(run / "probe.json")
                    meta["settings"] = _read_json(run / "settings.json")
                    meta["last_write"] = self._last_write(run)
                    meta["now"] = time.time()
                    return self._send(200, meta)
                if what == "jsonl" and len(parts) >= 6 and parts[5] in STREAMS:
                    since = int(q.get("since", ["0"])[0])
                    return self._send(200, self._tail(run / f"{parts[5]}.jsonl", since))
                if what == "file" and len(parts) >= 6:
                    p = run.joinpath(*parts[5:])
                    if not _inside(run, p):
                        return self._send(403, {"error": "forbidden"})
                    return self._file(p, cache=p.suffix == ".png" and (p.parent.name == "stream" or "samples" in p.parts))
                if what == "snapshot" and len(parts) >= 6:
                    p = run / "snapshots" / f"{int(parts[5]):08d}.json"
                    return self._file(p, cache=True)
            return self._send(404, {"error": "not found"})

        def do_POST(self):
            parts = [unquote(x) for x in urlparse(self.path).path.strip("/").split("/") if x]
            if len(parts) != 5 or parts[:2] != ["api", "run"] or parts[4] != "settings":
                return self._send(404, {"error": "not found"})
            run = root / parts[2] / parts[3]
            if not _inside(root, run) or not run.is_dir():
                return self._send(404, {"error": "no such run"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 65536:
                    raise ValueError("bad length")
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                probe = _read_json(run / "probe.json") or {}
                n = int(probe.get("n", 0))
                mode = body.get("mode")
                if mode == "manual":
                    idx = []
                    for i in body.get("display_idx", []):
                        i = int(i)
                        if not 0 <= i < n:
                            raise ValueError(f"image {i} is outside the probe set")
                        if i not in idx:
                            idx.append(i)
                    if not 1 <= len(idx) <= 64:
                        raise ValueError("pick between 1 and 64 images")
                    clean = {"mode": "manual", "display_idx": idx}
                elif mode == "auto":
                    k = int(body.get("n_display", 16))
                    if not 1 <= k <= 64:
                        raise ValueError("sample size must be between 1 and 64")
                    clean = {"mode": "auto", "n_display": k}
                else:
                    raise ValueError("mode must be auto or manual")
            except (ValueError, TypeError, json.JSONDecodeError) as e:
                return self._send(400, {"error": str(e)})
            clean["saved"] = time.time()
            tmp = run / "settings.json.tmp"
            tmp.write_text(json.dumps(clean), encoding="utf-8")
            for i in range(20):
                try:
                    tmp.replace(run / "settings.json")
                    break
                except PermissionError:
                    time.sleep(0.05 * (i + 1))
            return self._send(200, clean)

        @staticmethod
        def _last_write(run):
            p = run / "scalars.jsonl"
            return p.stat().st_mtime if p.exists() else None

        def _runs(self):
            out = []
            if not root.exists():
                return out
            for proj in sorted(x for x in root.iterdir() if x.is_dir()):
                for run in sorted(x for x in proj.iterdir() if x.is_dir()):
                    meta = _read_json(run / "meta.json")
                    if not meta:
                        continue
                    out.append({"project": proj.name, "name": run.name, "status": meta.get("status"),
                                "created": meta.get("created"), "last_write": self._last_write(run)})
            out.sort(key=lambda r: r.get("created") or 0, reverse=True)
            return out

        @staticmethod
        def _tail(p, since):
            if not p.exists():
                return {"offset": 0, "items": []}
            with open(p, "rb") as f:
                f.seek(since)
                data = f.read()
            end = data.rfind(b"\n")
            if end < 0:
                return {"offset": since, "items": []}
            chunk = data[:end + 1]
            items = []
            for line in chunk.splitlines():
                if line.strip():
                    try:
                        items.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
            return {"offset": since + len(chunk), "items": items}

    return Handler


def serve(root="ssleye_runs", host="127.0.0.1", port=5005):
    httpd = ThreadingHTTPServer((host, port), make_handler(root))
    print(f"[ssleye] dashboard on http://{host}:{port}  (runs from {Path(root).resolve()})", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
