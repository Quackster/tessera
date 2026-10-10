#!/usr/bin/env python3
"""External HTTP API harness for tessera.

Launch `tessera-cli serve` as a child process, wait for readiness, then
drive three capability probe families over the OpenAI-compatible HTTP
API:

  * steering       - the model obeys a system prompt (format, suffix,
                     language override).
  * long_context   - the model recalls a fact placed deep in a long
                     prompt (needle in a haystack).
  * memory         - a chat session remembers a fact stated on an
                     earlier turn and returns it on a later turn.

Each probe streams nothing but records the full reply and a pass/fail.
The script is memory-light: it holds only the reply text and reuses one
requests session. Run one model per invocation so the device is not
oversubscribed (see AGENTS.md rule 20).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import requests

# File-local tunables.
HEALTH_POLL_SECONDS = 2.0
SERVER_LOAD_TIMEOUT_SECONDS = 300.0
REQUEST_TIMEOUT_SECONDS = 240.0
SHUTDOWN_TIMEOUT_SECONDS = 30.0
DEFAULT_PORT = 8091
FILLER_SENTENCE = "The wind moved softly over the quiet green field at dusk. "
NEEDLE_CODE = "48217"
NEEDLE_FACT = (
    "Remember this one fact: the magic number of the blue vault is "
    f"{NEEDLE_CODE}.\n"
)
MEMORY_CODE = "9137"
MEMORY_PROJECT = "Falcon"
STEER_SUFFIX = "##OK##"


@dataclass
class ProbeResult:
    name: str
    family: str
    passed: bool
    detail: str
    reply: str = ""
    elapsed_s: float = 0.0


@dataclass
class Harness:
    base_url: str
    session: requests.Session = field(default_factory=requests.Session)

    def tokenize(self, text: str) -> Optional[int]:
        try:
            response = self.session.post(
                f"{self.base_url}/tokenize",
                json={"content": text},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            if response.status_code == 200:
                return len(response.json().get("tokens", []))
        except requests.RequestException:
            pass
        return None

    def chat(
        self,
        messages: list[dict],
        *,
        session_id: str = "",
        max_tokens: int = 64,
        enable_thinking: bool = False,
    ) -> tuple[Optional[dict], float]:
        body: dict = {
            "messages": messages,
            "max_completion_tokens": max_tokens,
            "enable_thinking": enable_thinking,
            "stream": False,
        }
        if session_id:
            body["session_id"] = session_id
        started = time.time()
        try:
            response = self.session.post(
                f"{self.base_url}/v1/chat/completions",
                json=body,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except requests.RequestException as exc:
            return None, time.time() - started
        elapsed = time.time() - started
        if response.status_code != 200:
            return {"error": f"HTTP {response.status_code}: {response.text[:200]}"}, elapsed
        data = response.json()
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        return {
            "content": message.get("content", "") or "",
            "reasoning": message.get("reasoning_content", "") or "",
            "finish_reason": choice.get("finish_reason", ""),
            "usage": data.get("usage", {}),
        }, elapsed

    def create_session(self) -> Optional[str]:
        try:
            response = self.session.post(
                f"{self.base_url}/v1/sessions",
                json={"title": "memory-probe"},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            if response.status_code in (200, 201):
                return response.json().get("id")
        except requests.RequestException:
            pass
        return None

    def session_history(self, session_id: str) -> Optional[list]:
        try:
            response = self.session.get(
                f"{self.base_url}/api/sessions/{session_id}",
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            if response.status_code == 200:
                return response.json().get("messages", [])
        except requests.RequestException:
            pass
        return None


def build_long_prompt(target_tokens: int, depth: float) -> str:
    """Fill to about `target_tokens` with the needle at `depth` fraction."""
    filler_tokens = max(1, round(len(FILLER_SENTENCE) / 4))
    repeats = max(1, target_tokens // filler_tokens)
    head = max(1, int(repeats * depth))
    body = FILLER_SENTENCE * head + NEEDLE_FACT + FILLER_SENTENCE * (repeats - head)
    return body


def probe_steering_format(harness: Harness) -> ProbeResult:
    system = (
        "You are a strict formatter. For every answer, output exactly one "
        "line and nothing else, in the form ANSWER=<value>. Never add other "
        "words."
    )
    reply, elapsed = harness.chat(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": "What is 2 plus 3?"},
        ],
        max_tokens=32,
    )
    if reply is None or "error" in reply:
        return ProbeResult("steering.format", "steering", False,
                          f"request failed: {reply}", "", elapsed)
    content = reply["content"].strip()
    passed = re.fullmatch(r"ANSWER=5", content) is not None
    return ProbeResult(
        "steering.format", "steering", passed,
        f"content={content!r}", content, elapsed,
    )


def probe_steering_suffix(harness: Harness) -> ProbeResult:
    system = (
        f"Always end every one of your answers with the exact token "
        f"{STEER_SUFFIX} at the very end. Never omit it."
    )
    reply, elapsed = harness.chat(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": "Name one primary color."},
        ],
        max_tokens=48,
    )
    if reply is None or "error" in reply:
        return ProbeResult("steering.suffix", "steering", False,
                          f"request failed: {reply}", "", elapsed)
    content = reply["content"].strip()
    passed = content.endswith(STEER_SUFFIX)
    return ProbeResult(
        "steering.suffix", "steering", passed,
        f"ends_with_suffix={passed} content={content!r}", content, elapsed,
    )


def probe_steering_language(harness: Harness) -> ProbeResult:
    system = (
        "You always answer in English, even when the user writes in another "
        "language. Never answer in French."
    )
    reply, elapsed = harness.chat(
        [
            {"role": "system", "content": system},
            {"role": "user",
             "content": "Reponds en francais: quelle est la capitale de la France?"},
        ],
        max_tokens=48,
    )
    if reply is None or "error" in reply:
        return ProbeResult("steering.language", "steering", False,
                          f"request failed: {reply}", "", elapsed)
    content = reply["content"].strip()
    lowered = content.lower()
    passed = "paris" in lowered and "capitale" not in lowered
    return ProbeResult(
        "steering.language", "steering", passed,
        f"content={content!r}", content, elapsed,
    )


def probe_long_context(harness: Harness, target_tokens: int, depth: float) -> ProbeResult:
    name = f"long_context.needle_{target_tokens}"
    prompt = build_long_prompt(target_tokens, depth)
    measured = harness.tokenize(prompt)
    reply, elapsed = harness.chat(
        [
            {"role": "system",
             "content": "Read the passage carefully and answer the question."},
            {"role": "user",
             "content": prompt +
             "\nQuestion: What is the magic number of the blue vault? "
             "Reply with only the number."},
        ],
        max_tokens=24,
    )
    if reply is None or "error" in reply:
        return ProbeResult(name, "long_context", False,
                          f"request failed: {reply}", "", elapsed)
    content = reply["content"].strip()
    passed = NEEDLE_CODE in content
    return ProbeResult(
        name, "long_context", passed,
        f"tokens={measured} depth={depth} content={content!r}", content, elapsed,
    )


def probe_memory_session(harness: Harness) -> ProbeResult:
    name = "memory.session_recall"
    session_id = harness.create_session()
    if not session_id:
        return ProbeResult(name, "memory", False, "could not create session")
    turn1, elapsed1 = harness.chat(
        [{"role": "user",
          "content": (f"Please remember these facts for later. My access code is "
                      f"{MEMORY_CODE} and my project is called {MEMORY_PROJECT}. "
                      f"Reply with just the word OK.")}],
        session_id=session_id,
        max_tokens=24,
    )
    if turn1 is None or "error" in turn1:
        return ProbeResult(name, "memory", False,
                          f"turn 1 failed: {turn1}", "", elapsed1)
    turn2, elapsed2 = harness.chat(
        [{"role": "user",
          "content": ("What is my access code and what is my project? "
                      "Answer with just the two values.")}],
        session_id=session_id,
        max_tokens=32,
    )
    if turn2 is None or "error" in turn2:
        return ProbeResult(name, "memory", False,
                          f"turn 2 failed: {turn2}", "", elapsed2)
    content = turn2["content"].strip()
    history = harness.session_history(session_id) or []
    passed = (MEMORY_CODE in content
              and MEMORY_PROJECT.lower() in content.lower()
              and len(history) >= 4)
    return ProbeResult(
        name, "memory", passed,
        f"content={content!r} stored_messages={len(history)}", content,
        elapsed1 + elapsed2,
    )


def run_probes(harness: Harness, families: set[str],
               long_context_sizes: list[int]) -> list[ProbeResult]:
    results: list[ProbeResult] = []
    if "steering" in families:
        results.append(probe_steering_format(harness))
        results.append(probe_steering_suffix(harness))
        results.append(probe_steering_language(harness))
    if "long_context" in families:
        for size in long_context_sizes:
            results.append(probe_long_context(harness, size, depth=0.10))
    if "memory" in families:
        results.append(probe_memory_session(harness))
    return results


def wait_for_ready(base_url: str, proc: subprocess.Popen,
                   timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200:
                return True
            if response.status_code == 503:
                try:
                    body = response.json()
                    if body.get("status") == "failed":
                        return False
                except ValueError:
                    pass
        except requests.RequestException:
            pass
        time.sleep(HEALTH_POLL_SECONDS)
    return False


def launch_server(args: argparse.Namespace, log_path: str) -> subprocess.Popen:
    command = [
        args.cli, "serve",
        "--model", args.model,
        "--gpu", str(args.gpu),
        "--port", str(args.port),
        "--context", str(args.context),
        "--no-auto-title",
    ]
    if args.draft:
        command += ["--draft", args.draft]
    if args.speculate:
        command += ["--speculate"]
    if args.kv_flag:
        command += [args.kv_flag]
    log_file = open(log_path, "w")
    return subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT)


def stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=SHUTDOWN_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=SHUTDOWN_TIMEOUT_SECONDS)


def parse_families(value: str) -> set[str]:
    known = {"steering", "long_context", "memory"}
    if value == "all":
        return known
    selected = {part.strip() for part in value.split(",") if part.strip()}
    unknown = selected - known
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown families: {sorted(unknown)}")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", required=True, help="path to tessera-cli")
    parser.add_argument("--model", required=True, help="model path")
    parser.add_argument("--draft", default="", help="DFlash2 draft directory")
    parser.add_argument("--speculate", action="store_true", help="enable MTP")
    parser.add_argument("--gpu", type=int, default=1, help="device index")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--kv-flag", default="", help="e.g. --kv-q8")
    parser.add_argument("--families", type=parse_families, default="all")
    parser.add_argument("--long-context-sizes", default="1024,2048",
                        help="comma separated token targets")
    parser.add_argument("--backend", default="rocm", help="backend label")
    parser.add_argument("--out", default="", help="results JSON path")
    parser.add_argument("--smoke", action="store_true",
                        help="one tiny request then exit")
    args = parser.parse_args()

    sizes = [int(x) for x in args.long_context_sizes.split(",") if x.strip()]
    base_url = f"http://127.0.0.1:{args.port}"
    os.makedirs("/tmp/tessera-harness", exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9]+", "_", os.path.basename(args.model.rstrip("/")))
    log_path = f"/tmp/tessera-harness/{safe}.log"

    print(f"harness: launching {args.cli} serve for {args.model} on gpu {args.gpu}")
    proc = launch_server(args, log_path)
    started = time.time()
    try:
        if not wait_for_ready(base_url, proc, SERVER_LOAD_TIMEOUT_SECONDS):
            print(f"harness: server not ready; log at {log_path}")
            tail = ""
            try:
                with open(log_path) as handle:
                    tail = "".join(handle.readlines()[-20:])
            except OSError:
                pass
            print(tail)
            return 2
        print(f"harness: ready in {time.time() - started:.1f}s")

        harness = Harness(base_url)
        if args.smoke:
            reply, elapsed = harness.chat(
                [{"role": "user", "content": "Reply with the single word READY."}],
                max_tokens=8)
            print(f"smoke: {elapsed:.1f}s {reply}")
            return 0

        results = run_probes(harness, args.families, sizes)
    finally:
        stop_server(proc)

    passed = sum(1 for r in results if r.passed)
    total = len(results)
    print(f"\n=== {args.model} ({args.backend}, gpu {args.gpu}) ===")
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.name:28s} {r.elapsed_s:6.1f}s  {r.detail}")
    print(f"  {passed}/{total} probes passed")

    report = {
        "model": args.model,
        "backend": args.backend,
        "gpu": args.gpu,
        "passed": passed,
        "total": total,
        "results": [r.__dict__ for r in results],
    }
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as handle:
            json.dump(report, handle, indent=2)
        print(f"harness: wrote {args.out}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
