"""Check the real headless Task3 HTTP service without using a webcam or recording."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / ".local/task3-check/service")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    dataset = output / "no_record_dataset"
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    with (output / "service.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, "-u", "-m", "simulation.mujoco.dual_arm.record_mujoco_dual_arm",
             "--headless", "--port", str(port), "--dataset-root", str(dataset), "--max-seconds", "30"],
            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        try:
            deadline = time.monotonic() + 90
            while True:
                if process.poll() is not None:
                    raise RuntimeError("Service exited during startup; inspect service.log")
                try:
                    with urlopen(base + "/health", timeout=2) as response:
                        health = json.load(response)
                    assert health["service"] == "mujoco-task3-recorder"
                    assert not health["recording"] and health["frames"] == 0
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        raise TimeoutError("Task3 service startup timed out")
                    time.sleep(0.2)
            images = {}
            for name in ("top", "front", "side"):
                deadline = time.monotonic() + 15
                while True:
                    with urlopen(base + f"/{name}-preview", timeout=3) as response:
                        data = response.read()
                        if response.status == 200:
                            assert data[:2] == b"\xff\xd8"
                            images[name] = len(data)
                            (output / f"{name}.jpg").write_bytes(data)
                            break
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"No fresh {name} preview")
                    time.sleep(0.15)
            assert not dataset.exists()
            assert not dataset.with_name(dataset.name + "_failures").exists()
            request = Request(base + "/control", data=json.dumps({"command": "record_stop", "eventId": "service-check-stop"}).encode(),
                              headers={"Content-Type": "application/json"}, method="POST")
            with urlopen(request, timeout=3) as response:
                assert json.load(response)["queued"]
            process.wait(timeout=30)
            assert process.returncode == 0
            result = {"valid": True, "real_http_service": True, "headless": True,
                      "preview_bytes": images, "startup_created_dataset": False,
                      "camera_access": False, "exit_code": process.returncode}
            (output / "report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(result))
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=15)


if __name__ == "__main__":
    main()
