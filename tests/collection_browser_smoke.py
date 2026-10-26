"""Check real browser DOM after repeated health polling; never open the camera.

Run against an already launched collection service, optionally with --browser.
A dedicated headless profile is created under .local/checks, never the user profile.
"""
from pathlib import Path
import argparse
from datetime import datetime
import json
import os
import re
import subprocess
import urllib.request
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8014/dual_arm.html")
    parser.add_argument("--browser", type=Path)
    args = parser.parse_args()
    parsed = urlsplit(args.url)
    if parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1"}:
        parser.error("Only the local collection page is allowed")
    candidates = [args.browser] if args.browser else [
        Path(os.environ.get("ProgramFiles", "C:/Program Files"))/"Google/Chrome/Application/chrome.exe",
        Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)"))/"Microsoft/Edge/Application/msedge.exe",
    ]
    browser = next((p for p in candidates if p and p.is_file()), None)
    if browser is None:
        parser.error("Chrome/Edge not found; use --browser")
    output = ROOT/".local/checks/browser_connection"/datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True)
    with urllib.request.urlopen(f"{parsed.scheme}://{parsed.netloc}/api/dual/health", timeout=5) as response:
        health = json.load(response)
    assert health["ok"] and Path(health["project_root"]).resolve() == ROOT
    assert not health.get("recording") and not health.get("saving"), "Run this check only while idle"
    command = [str(browser), "--headless=new", "--disable-gpu", "--no-first-run",
               "--no-default-browser-check", "--disable-background-networking", "--enable-logging=stderr",
               f"--user-data-dir={output/'profile'}", "--dump-dom", "--virtual-time-budget=5000", args.url]
    result = subprocess.run(command, capture_output=True, timeout=40,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    (output/"page.html").write_bytes(result.stdout)
    (output/"browser.log").write_bytes(result.stderr)
    html = result.stdout.decode("utf-8", errors="replace")
    match = re.search(r'<span id="connection"[^>]*>(.*?)</span>', html)
    connection = match.group(1) if match else None
    report = {"url": args.url, "http_health_ok": True, "browser_exit": result.returncode,
              "connection_text": connection, "camera_requested": False,
              "passed": result.returncode == 0 and connection == "双臂仿真已连接"}
    (output/"result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**report, "evidence": str(output)}, ensure_ascii=False))
    assert report["passed"], "Real browser connection failed; inspect page.html and browser.log"

if __name__ == "__main__":
    main()
