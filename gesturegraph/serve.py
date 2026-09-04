from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch

from .data import normalize_sequence, resample_sequence
from .model import build_model


def load_model(path: Path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(
        checkpoint.get("model_name", "stgcn"),
        len(checkpoint["labels"]),
        int(checkpoint.get("frames", 64)),
        float(checkpoint.get("dropout", .15)),
        checkpoint.get("ablation", "none"),
        checkpoint.get("model_config"),
    )
    model.load_state_dict(checkpoint["model_state"]); model.eval()
    return model, checkpoint["labels"], int(checkpoint.get("frames", 64))


def handler_factory(model, labels, frames, recordings_dir):
    class GestureHandler(SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            if self.path != "/api/predict":
                super().log_message(format, *args)

        def end_headers(self):
            self.send_header("Cache-Control", "no-store")
            super().end_headers()

        def do_GET(self):
            if self.path == "/api/health":
                return self.respond({"status": "ok", "model": "HandSTGCN" if model else None, "labels": labels})
            if self.path == "/api/dataset":
                counts = Counter()
                for path in recordings_dir.glob("*.json"):
                    try:
                        counts[json.loads(path.read_text(encoding="utf-8"))["label"]] += 1
                    except (KeyError, json.JSONDecodeError):
                        pass
                return self.respond({"total": sum(counts.values()), "labels": dict(sorted(counts.items()))})
            return super().do_GET()

        def do_POST(self):
            if self.path == "/api/recordings":
                return self.save_recording()
            if self.path != "/api/predict":
                self.send_error(404); return
            if model is None:
                return self.respond({"error": "no trained model loaded"}, status=503)
            try:
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length))
                sequence = resample_sequence(normalize_sequence(np.asarray(payload["sequence"], dtype=np.float32)), frames)
                inputs = torch.from_numpy(sequence).permute(2, 0, 1).unsqueeze(0)
                with torch.no_grad():
                    probabilities = model(inputs).softmax(dim=1)[0].numpy()
                scores = {label: float(probabilities[index]) for index, label in enumerate(labels)}
                best = int(probabilities.argmax())
                self.respond({"label": labels[best], "confidence": float(probabilities[best]), "scores": scores})
            except (ValueError, KeyError, json.JSONDecodeError) as error:
                self.respond({"error": str(error)}, status=400)

        def save_recording(self):
            try:
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length))
                label = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(payload["label"]).strip())
                if not label:
                    raise ValueError("label is required")
                sequence = np.asarray(payload["sequence"], dtype=np.float32)
                resample_sequence(sequence, frames)  # Validate without changing the captured frames.
                recordings_dir.mkdir(parents=True, exist_ok=True)
                filename = f"{label}-{time.time_ns()}.json"
                destination = recordings_dir / filename
                destination.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                count = sum(1 for path in recordings_dir.glob(f"{label}-*.json"))
                self.respond({"file": str(destination), "label": label, "count": count})
            except (ValueError, KeyError, json.JSONDecodeError) as error:
                self.respond({"error": str(error)}, status=400)

        def respond(self, payload, status=200):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    return GestureHandler


def main():
    parser = argparse.ArgumentParser(description="Serve GestureGraph and a trained ST-GCN")
    parser.add_argument("--model", default="runs/stgcn/best.pt", help="Optional checkpoint; collection works without it")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--recordings", default="data/recordings")
    args = parser.parse_args()
    model_path = Path(args.model)
    model, labels, frames = load_model(model_path) if model_path.exists() else (None, [], 64)
    server = ThreadingHTTPServer(("localhost", args.port), handler_factory(model, labels, frames, Path(args.recordings)))
    model_text = f"model labels: {', '.join(labels)}" if model else "collection mode (no trained model)"
    print(f"GestureGraph: http://localhost:{args.port} | {model_text}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nGestureGraph stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
