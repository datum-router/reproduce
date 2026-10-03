#!/usr/bin/env python3
"""Web UI for the job-apply browser agent.

Wraps the observe/decide/act loop from agent.py (subclassed, not rewritten)
and exposes it through a small Flask app:

    GET  /                      the UI page (form, live log, screenshots)
    POST /api/runs              start a run in a background thread -> {run_id}
    GET  /api/runs/<id>/status  JSON {state, steps_done, log_tail, shots, summary}
    GET  /runs/<id>/shots/<f>   serve a run screenshot

Safety: web runs NEVER auto-submit. There is no --auto-submit here at all;
the agent always stops before the final Submit/Apply button and returns a
summary for a human to review.

Run locally:
    python webapp/app.py            # http://localhost:5000 (PORT env to change)
"""

import datetime
import html
import json
import os
import queue
import re
import sys
import threading
import time
import traceback
import uuid

from flask import Flask, jsonify, request, send_from_directory

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from agent import Agent, RUNS_DIR  # noqa: E402
from llm_client import LLMClient  # noqa: E402

app = Flask(__name__)

# ---------- run registry ----------

runs = {}  # run_id -> record
runs_lock = threading.Lock()
RUN_ID_RE = re.compile(r"^[0-9a-f]{12}$")


def _now():
    return datetime.datetime.now().strftime("%H:%M:%S")


def load_profile_text():
    for name in ("profile.json", "profile.example.json"):
        path = os.path.join(BASE_DIR, name)
        if os.path.exists(path):
            with open(path) as f:
                return f.read()
    return "{}"


def load_rules():
    path = os.path.join(BASE_DIR, "rules.md")
    if os.path.exists(path):
        with open(path) as f:
            return f.read()
    return ""


DEFAULT_PROFILE_TEXT = load_profile_text()
RULES_TEXT = load_rules()

# ---------- agent wrapper ----------


class Callbacks:
    """on_log(text), on_step(summary, shot_name), on_done(how, summary)."""

    def __init__(self, on_log, on_step, on_done):
        self.on_log = on_log
        self.on_step = on_step
        self.on_done = on_done


class WebAgent(Agent):
    """Agent subclass wired for the web UI.

    Differences from the CLI Agent:
    - explicit run_id (uuid) instead of a timestamp, so concurrent runs
      never collide;
    - screenshots land in runs/<run_id>/shots/ next to log.jsonl;
    - auto_submit is forced off: there is no submit path in web mode;
    - password fields become need_human (no getpass prompt in a server);
    - every log line / step / completion is pushed through callbacks.
    """

    def __init__(self, run_id, url, profile, rules, max_steps, callbacks,
                 control_queue, paused, llm_client=None):
        super().__init__(url, profile, rules, headed=False,
                         auto_submit=False, max_steps=max_steps,
                         llm_client=llm_client)
        self.run_id = run_id
        self.control_queue = control_queue  # user ops, agent thread executes
        self.paused = paused                # threading.Event: human has control
        self.run_dir = os.path.join(RUNS_DIR, run_id)
        self.shots_dir = os.path.join(self.run_dir, "shots")
        os.makedirs(self.shots_dir, exist_ok=True)
        self.log_path = os.path.join(self.run_dir, "log.jsonl")
        self.callbacks = callbacks

    # -- logging --

    def log(self, event, **fields):
        super().log(event, **fields)
        short = {k: (str(v)[:120]) for k, v in fields.items()
                 if k in ("url", "reason", "error", "how", "name", "action")}
        self.callbacks.on_log("%s %s" % (event, short))

    # -- screenshots (per-run dir, reported to the UI) --

    def screenshot(self):
        self.steps_executed += 1
        fname = "step_%03d.png" % self.steps_executed
        path = os.path.join(self.shots_dir, fname)
        try:
            self.page.screenshot(path=path)
            self.callbacks.on_step("step %d done" % self.steps_executed,
                                   fname)
        except Exception as exc:  # noqa: BLE001
            self.log("screenshot_failed", error=str(exc)[:200])
        self.refresh_live()

    def refresh_live(self):
        """Overwrite runs/<run_id>/live.png with the current viewport."""
        path = os.path.join(self.run_dir, "live.png")
        try:
            if self.page is not None:
                self.page.screenshot(path=path)
        except Exception as exc:  # noqa: BLE001
            self.log("live_failed", error=str(exc)[:200])

    def _apply_control_op(self, op):
        """Execute one user control op. Called in the agent thread only,
        because Playwright's sync API is not thread-safe."""
        kind = op.get("op")
        try:
            if kind == "click":
                self.page.mouse.click(float(op["x"]), float(op["y"]))
            elif kind == "type":
                self.page.keyboard.type(op.get("text", ""))
            elif kind == "press":
                self.page.keyboard.press(op.get("key", "Enter"))
            elif kind == "scroll":
                self.page.mouse.wheel(float(op.get("dx", 0)),
                                      float(op.get("dy", 0)))
            else:
                self.log("control_rejected", op=str(kind)[:40])
                return
        except Exception as exc:  # noqa: BLE001
            self.log("control_failed", op=str(kind)[:40],
                     error=str(exc)[:200])
            return
        self.log("control", op=kind)
        self.refresh_live()

    def _drain_control_queue(self):
        while True:
            try:
                op = self.control_queue.get_nowait()
            except queue.Empty:
                return
            self._apply_control_op(op)

    # -- loop with web-mode guards --

    def fill_loop(self):
        while self.steps_executed < self.max_steps:
            if self.paused.is_set():
                # Human has the wheel: run their queued control ops, keep
                # the live view fresh, and do NOT call the LLM while paused.
                self._drain_control_queue()
                self.refresh_live()
                time.sleep(0.3)
                continue
            items, by_id = self.snapshot()
            self.callbacks.on_log("snapshot: %d elements on %s"
                                  % (len(items), self.page.url))
            steps = self.decide(items)
            if steps is None:
                return "llm_error"
            for step in steps:
                action = step["action"]
                if action == "done":
                    self.log("done", reason=step.get("reason", ""))
                    return "done"
                if action == "need_human":
                    self.log("need_human", reason=step.get("reason", ""))
                    return "need_human:" + step.get("reason", "")
                item = by_id.get(step.get("id"), {}) if step.get("id") else {}
                if action == "fill" and item.get("type") == "password":
                    self.log("need_human",
                             reason="password field in web mode (no prompt)")
                    return ("need_human:the page asks for a password; web "
                            "runs cannot prompt for one")
                result = self.execute_step(step, by_id)
                self.screenshot()
                if result in ("submit_blocked", "submit_requested"):
                    return result
                if self.steps_executed >= self.max_steps:
                    break
        return "max_steps"

    # -- summary + run --

    def build_summary(self, how):
        lines = ["RESULT: %s" % how]
        lines.append("")
        lines.append("Filled fields (%d):" % len(self.filled))
        for label, value in self.filled:
            shown = value if len(str(value)) < 60 else str(value)[:57] + "..."
            lines.append("  %s = %s" % (label[:40], shown))
        if not self.filled:
            lines.append("  (nothing filled)")
        lines.append("")
        lines.append("Log: runs/%s/log.jsonl" % self.run_id)
        lines.append("Screenshots: %d in runs/%s/shots/"
                     % (self.steps_executed, self.run_id))
        if how == "stopped_before_submit":
            lines.append("Stopped before the final Submit/Apply button "
                         "(web runs never submit). Review the fields above.")
        elif how.startswith("need_human"):
            lines.append("Needs a human: %s" % how.split(":", 1)[-1])
        elif how == "max_steps":
            lines.append("Step budget (%d) exhausted." % self.max_steps)
        elif how == "llm_error":
            lines.append("The model call failed or returned invalid output. "
                         "Check LLM_* env vars.")
        return "\n".join(lines)

    def run(self):
        self.callbacks.on_log("Starting: opening %s" % self.url)
        how = "error"
        try:
            self.start()
            self.refresh_live()  # so the live panel shows something at once
            try:
                how = self.fill_loop()
                self.log("finished", how=how)
            finally:
                pass
        except Exception as exc:  # noqa: BLE001 - surface, don't crash thread
            how = "error: %s" % str(exc)[:200]
            self.callbacks.on_log("run error: %s" % traceback.format_exc()[-500:])
        finally:
            try:
                self.close()
            except Exception:
                pass
        summary = self.build_summary(how)
        self.callbacks.on_done(how, summary)


def run_application(run_id, url, profile, rules, max_steps, callbacks,
                    control_queue, paused, llm_client=None):
    """Entry point used by the web UI: build a WebAgent and run it."""
    agent = WebAgent(run_id, url, profile, rules,
                     max_steps=max_steps, callbacks=callbacks,
                     control_queue=control_queue, paused=paused,
                     llm_client=llm_client)
    agent.run()
    return run_id


# ---------- background run plumbing ----------

def _record(run_id):
    with runs_lock:
        return runs.get(run_id)


def make_callbacks(run_id):
    def on_log(text):
        rec = _record(run_id)
        if rec is None:
            return
        with runs_lock:
            rec["log"].append("[%s] %s" % (_now(), text))
            rec["log"] = rec["log"][-500:]

    def on_step(summary, shot_name):
        rec = _record(run_id)
        if rec is None:
            return
        with runs_lock:
            rec["steps_done"] += 1
            rec["shots"].append("/runs/%s/shots/%s" % (run_id, shot_name))
            rec["log"].append("[%s] %s" % (_now(), summary))

    def on_done(how, summary):
        rec = _record(run_id)
        if rec is None:
            return
        with runs_lock:
            rec["state"] = "done"
            rec["summary"] = {"result": how, "text": summary}
            rec["log"].append("[%s] finished: %s" % (_now(), how))

    return Callbacks(on_log, on_step, on_done)


def _run_in_thread(run_id, url, profile, rules_text, max_steps,
                   control_queue, paused, llm_client=None):
    try:
        run_application(run_id, url, profile, rules_text, max_steps,
                        make_callbacks(run_id), control_queue, paused,
                        llm_client=llm_client)
    except Exception as exc:  # noqa: BLE001 - never leave a run hanging
        rec = _record(run_id)
        if rec is not None:
            with runs_lock:
                rec["state"] = "error"
                rec["log"].append("[%s] thread error: %s" % (_now(), exc))


# ---------- routes ----------

@app.route("/")
def index():
    # The textarea holds plain HTML text, not a JS string literal, so only
    # HTML-escape it. Escaping $, backticks or backslashes here corrupts the
    # JSON (e.g. $80k became \$80k, which JSON.parse rejects in the browser).
    safe = html.escape(DEFAULT_PROFILE_TEXT)
    return INDEX_HTML.replace("__PROFILE_JSON__", safe)


@app.route("/api/runs", methods=["POST"])
def api_start_run():
    data = request.get_json(force=True, silent=True) or {}
    url = (data.get("url") or "").strip()
    if not url or not (url.startswith("http://") or url.startswith("https://")):
        return jsonify({"error": "a valid http(s) url is required"}), 400
    raw_profile = data.get("profile", "")
    try:
        profile = json.loads(raw_profile) if isinstance(raw_profile, str) \
            else raw_profile
    except (ValueError, TypeError):
        return jsonify({"error": "profile is not valid JSON"}), 400
    if not isinstance(profile, dict):
        return jsonify({"error": "profile must be a JSON object"}), 400
    try:
        max_steps = int(data.get("max_steps", 25))
    except (ValueError, TypeError):
        return jsonify({"error": "max_steps must be a number"}), 400
    max_steps = max(1, min(100, max_steps))

    # Optional per-run LLM override (for when the default Pollinations
    # endpoint is degraded). Empty means "use server defaults". The API key
    # is held in memory for this run only: it is never stored in the run
    # record, never logged, and never returned by the status endpoint.
    llm_base_url = (data.get("llm_base_url") or "").strip() or None
    llm_model = (data.get("llm_model") or "").strip() or None
    llm_api_key = (data.get("llm_api_key") or "").strip() or None
    llm_client = None
    if llm_base_url or llm_model or llm_api_key:
        llm_client = LLMClient(base_url=llm_base_url, model=llm_model,
                               api_key=llm_api_key)

    run_id = uuid.uuid4().hex[:12]
    control_queue = queue.Queue()
    paused = threading.Event()
    with runs_lock:
        runs[run_id] = {
            "run_id": run_id,
            "state": "starting",
            "paused": False,
            "url": url,
            "max_steps": max_steps,
            "steps_done": 0,
            "log": [],
            "shots": [],
            "summary": None,
            "started": datetime.datetime.now().isoformat(timespec="seconds"),
            "control_queue": control_queue,
            "paused_event": paused,
        }
        if llm_client is not None:
            # Redacted: base_url and model only, never the key.
            runs[run_id]["log"].append(
                "[%s] llm override: base_url=%s model=%s (api key %s)"
                % (_now(), llm_client.base_url, llm_client.model,
                   "supplied" if llm_api_key else "not supplied"))
    t = threading.Thread(target=_run_in_thread,
                         args=(run_id, url, profile, RULES_TEXT, max_steps,
                               control_queue, paused, llm_client),
                         daemon=True)
    with runs_lock:
        runs[run_id]["state"] = "running"
    t.start()
    return jsonify({"run_id": run_id})


@app.route("/api/runs/<run_id>/status")
def api_run_status(run_id):
    rec = _record(run_id)
    if rec is None:
        return jsonify({"error": "unknown run_id"}), 404
    with runs_lock:
        return jsonify({
            "run_id": rec["run_id"],
            "state": rec["state"],
            "paused": rec.get("paused", False),
            "url": rec["url"],
            "steps_done": rec["steps_done"],
            "log_tail": rec["log"][-80:],
            "shots": rec["shots"],
            "summary": rec["summary"],
        })


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


@app.route("/api/runs/<run_id>/pause", methods=["POST"])
def api_pause(run_id):
    rec = _record(run_id)
    if rec is None:
        return jsonify({"error": "unknown run_id"}), 404
    with runs_lock:
        if rec["state"] != "running":
            return jsonify({"error": "run is not active"}), 409
        rec["paused_event"].set()
        rec["paused"] = True
    return jsonify({"paused": True})


@app.route("/api/runs/<run_id>/resume", methods=["POST"])
def api_resume(run_id):
    rec = _record(run_id)
    if rec is None:
        return jsonify({"error": "unknown run_id"}), 404
    with runs_lock:
        if rec["state"] != "running":
            return jsonify({"error": "run is not active"}), 409
        rec["paused_event"].clear()
        rec["paused"] = False
    return jsonify({"paused": False})


@app.route("/api/runs/<run_id>/control", methods=["POST"])
def api_control(run_id):
    """Queue one user control op for the agent thread to execute.

    Ops: click {x,y}, type {text}, press {key}, scroll {dx,dy}.
    Coordinates are CSS pixels in the page viewport.
    """
    rec = _record(run_id)
    if rec is None:
        return jsonify({"error": "unknown run_id"}), 404
    data = request.get_json(force=True, silent=True) or {}
    op = data.get("op")
    if op == "click":
        x, y = data.get("x"), data.get("y")
        if not _is_num(x) or not _is_num(y):
            return jsonify({"error": "click needs numeric x and y"}), 400
        action = {"op": "click", "x": float(x), "y": float(y)}
    elif op == "type":
        text = data.get("text", "")
        if not isinstance(text, str) or not text:
            return jsonify({"error": "type needs a non-empty text"}), 400
        action = {"op": "type", "text": text[:2000]}
    elif op == "press":
        key = data.get("key", "")
        if not isinstance(key, str) or not key:
            return jsonify({"error": "press needs a key"}), 400
        action = {"op": "press", "key": key[:40]}
    elif op == "scroll":
        dx, dy = data.get("dx", 0), data.get("dy", 0)
        if not _is_num(dx) or not _is_num(dy):
            return jsonify({"error": "scroll needs numeric dx/dy"}), 400
        action = {"op": "scroll", "dx": float(dx), "dy": float(dy)}
    else:
        return jsonify({"error": "unknown op (click/type/press/scroll)"}), 400
    with runs_lock:
        if rec["state"] != "running":
            return jsonify({"error": "run is not active"}), 409
        # A control op means the human wants the wheel: pause the agent so
        # the op is picked up on the next loop pass, even if they skipped
        # the Take control button.
        rec["paused_event"].set()
        rec["paused"] = True
        rec["control_queue"].put(action)
    return jsonify({"queued": op, "paused": True})


@app.route("/runs/<run_id>/live.png")
def serve_live(run_id):
    if not RUN_ID_RE.match(run_id):
        return jsonify({"error": "not found"}), 404
    directory = os.path.join(RUNS_DIR, run_id)
    if not os.path.exists(os.path.join(directory, "live.png")):
        return jsonify({"error": "no live image yet"}), 404
    return send_from_directory(directory, "live.png")


@app.route("/runs/<run_id>/shots/<fname>")
def serve_shot(run_id, fname):
    if not RUN_ID_RE.match(run_id) or not re.fullmatch(r"step_\d+\.png", fname):
        return jsonify({"error": "not found"}), 404
    return send_from_directory(os.path.join(RUNS_DIR, run_id, "shots"), fname)


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Job Apply Agent</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 1100px; margin: 0 auto;
         padding: 20px; background: #f6f7f9; color: #222; }
  h1 { font-size: 1.4em; }
  .card { background: #fff; border: 1px solid #ddd; border-radius: 8px;
          padding: 16px; margin-bottom: 16px; }
  label { font-weight: 600; display: block; margin: 10px 0 4px; }
  input[type=text], input[type=url], input[type=number], textarea {
    width: 100%; box-sizing: border-box; padding: 8px; border: 1px solid #ccc;
    border-radius: 6px; font-size: 14px; }
  textarea { font-family: monospace; font-size: 12px; }
  button { background: #1a73e8; color: #fff; border: 0; border-radius: 6px;
           padding: 10px 18px; font-size: 15px; cursor: pointer; }
  button:disabled { background: #999; cursor: default; }
  #log { background: #111; color: #cfc; font-family: monospace; font-size: 12px;
         height: 300px; overflow-y: auto; padding: 10px; border-radius: 6px;
         white-space: pre-wrap; }
  #shots img { max-width: 320px; margin: 6px; border: 1px solid #ccc;
               border-radius: 4px; }
  #summary { white-space: pre-wrap; font-family: monospace; font-size: 13px;
             background: #f0f4ff; padding: 12px; border-radius: 6px; }
  .note { color: #555; font-size: 13px; }
  #liveWrap { text-align: center; }
  #live { max-width: 100%; border: 1px solid #ccc; border-radius: 4px; }
  #controlBanner { display: none; background: #fff3cd; border: 1px solid #e0c36a;
                   padding: 10px; border-radius: 6px; margin-bottom: 10px;
                   font-weight: 600; }
  #controlBar { display: none; margin-top: 10px; }
  #controlBar button { margin: 2px; padding: 6px 10px; font-size: 13px; }
  #controlBar input[type=text] { width: 260px; display: inline-block; }
</style>
</head>
<body>
<h1>Job Apply Agent</h1>
<p class="note">Paste a job posting URL, check the test profile, hit Start.
The agent opens a sandboxed Chromium, fills the form, and always stops
before the final Submit button. Watch the log and screenshots below.</p>

<div class="card">
  <label for="url">Job posting URL</label>
  <input type="url" id="url" placeholder="https://example.com/jobs/123">
  <label for="maxsteps">Max steps</label>
  <input type="number" id="maxsteps" value="25" min="1" max="100" style="width:120px">
  <label for="profile">Profile JSON (test data prefilled)</label>
  <textarea id="profile" rows="12">__PROFILE_JSON__</textarea>
  <label>LLM override (optional, for when Pollinations is down)</label>
  <input type="text" id="llmBaseUrl" placeholder="Base URL, e.g. https://generativelanguage.googleapis.com/v1beta/openai (blank = default)">
  <input type="text" id="llmModel" placeholder="Model, e.g. gemini-2.0-flash (blank = default)" style="margin-top:6px">
  <input type="password" id="llmApiKey" placeholder="API key (blank = default)" style="margin-top:6px" autocomplete="off">
  <p class="note">A key entered here stays in memory for this run only. It is never saved, never logged, and never appears in the status output.</p>
  <p><button id="start" onclick="startRun()">Start run</button></p>
</div>

<div class="card">
  <h3>Live browser</h3>
  <div id="controlBanner">PAUSED: you have control of the browser. Click the page
  below, or use the toolbar, then hit "Resume agent" to hand it back.</div>
  <div id="liveWrap"><img id="live" alt="live browser view" style="display:none"></div>
  <p>
    <button id="takeControl" onclick="takeControl()" disabled>Take control</button>
    <button id="resumeAgent" onclick="resumeAgent()" style="display:none">Resume agent</button>
  </p>
  <div id="controlBar">
    <input type="text" id="typeText" placeholder="text to type into the focused field">
    <button onclick="sendType()">Type</button>
    <button onclick="sendKey('Tab')">Tab</button>
    <button onclick="sendKey('Enter')">Enter</button>
    <button onclick="sendKey('Escape')">Esc</button>
    <button onclick="sendKey('ArrowDown')">Down</button>
    <button onclick="sendKey('ArrowUp')">Up</button>
    <button onclick="sendScroll(-400)">Scroll up</button>
    <button onclick="sendScroll(400)">Scroll down</button>
  </div>
  <p class="note">While you have control, clicking the live image clicks that spot
  in the real browser. The agent never submits on its own; clicks you make yourself
  are your own action.</p>
</div>

<div class="card">
  <h3>Live log</h3>
  <div id="log">(no run yet)</div>
</div>

<div class="card">
  <h3>Screenshots</h3>
  <div id="shots"><span class="note">Screenshots appear here as the agent works.</span></div>
</div>

<div class="card">
  <h3>Summary</h3>
  <div id="summary">(appears when the run finishes)</div>
</div>

<script>
let runId = null, timer = null, logSeen = 0, shotsSeen = 0;
let controlling = false;

async function takeControl() {
  if (!runId) return;
  const r = await fetch('/api/runs/' + runId + '/pause', {method: 'POST'});
  if (r.ok) { controlling = true; updateControlUI(); }
}

async function resumeAgent() {
  if (!runId) return;
  const r = await fetch('/api/runs/' + runId + '/resume', {method: 'POST'});
  if (r.ok) { controlling = false; updateControlUI(); }
}

function updateControlUI() {
  document.getElementById('takeControl').style.display = controlling ? 'none' : '';
  document.getElementById('resumeAgent').style.display = controlling ? '' : 'none';
  document.getElementById('controlBanner').style.display = controlling ? '' : 'none';
  document.getElementById('controlBar').style.display = controlling ? '' : 'none';
  document.getElementById('live').style.cursor = controlling ? 'crosshair' : 'default';
}

async function sendControl(payload) {
  if (!runId) return;
  await fetch('/api/runs/' + runId + '/control', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload)});
}

function sendType() {
  const t = document.getElementById('typeText').value;
  if (t) sendControl({op: 'type', text: t});
}
function sendKey(key) { sendControl({op: 'press', key: key}); }
function sendScroll(dy) { sendControl({op: 'scroll', dx: 0, dy: dy}); }

document.getElementById('live').addEventListener('click', (e) => {
  if (!controlling || !runId) return;
  const img = e.currentTarget;
  if (!img.naturalWidth || !img.clientWidth) return;
  const x = e.offsetX * (img.naturalWidth / img.clientWidth);
  const y = e.offsetY * (img.naturalHeight / img.clientHeight);
  sendControl({op: 'click', x: x, y: y});
});
document.getElementById('live').addEventListener('error', function() {
  this.style.display = 'none';
});
document.getElementById('live').addEventListener('load', function() {
  this.style.display = '';
});

async function startRun() {
  const url = document.getElementById('url').value.trim();
  const profile = document.getElementById('profile').value;
  const max_steps = parseInt(document.getElementById('maxsteps').value, 10) || 25;
  const llm_base_url = document.getElementById('llmBaseUrl').value.trim();
  const llm_model = document.getElementById('llmModel').value.trim();
  const llm_api_key = document.getElementById('llmApiKey').value;
  if (!url) { alert('Enter a job posting URL first.'); return; }
  try { JSON.parse(profile); } catch (e) { alert('Profile is not valid JSON.'); return; }
  const btn = document.getElementById('start');
  btn.disabled = true; btn.textContent = 'Running...';
  document.getElementById('log').textContent = '';
  document.getElementById('shots').innerHTML = '';
  document.getElementById('summary').textContent = '(running...)';
  logSeen = 0; shotsSeen = 0;
  const r = await fetch('/api/runs', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({url, profile, max_steps,
                          llm_base_url, llm_model, llm_api_key})
  });
  // Don't leave the key sitting in the form.
  document.getElementById('llmApiKey').value = '';
  const data = await r.json();
  if (!r.ok) { alert(data.error || 'Failed to start'); btn.disabled = false; btn.textContent = 'Start run'; return; }
  runId = data.run_id;
  controlling = false; updateControlUI();
  document.getElementById('takeControl').disabled = true;
  timer = setInterval(poll, 1500);
  poll();
}

async function poll() {
  if (!runId) return;
  const r = await fetch('/api/runs/' + runId + '/status');
  const s = await r.json();
  const live = document.getElementById('live');
  if (s.state === 'running' || s.state === 'starting') {
    live.src = '/runs/' + runId + '/live.png?t=' + Date.now();
    document.getElementById('takeControl').disabled = false;
  }
  if (typeof s.paused === 'boolean' && s.paused !== controlling) {
    controlling = s.paused;
    updateControlUI();
  }
  const logEl = document.getElementById('log');
  const lines = s.log_tail || [];
  for (let i = logSeen; i < lines.length; i++) logEl.textContent += lines[i] + '\\n';
  logSeen = lines.length;
  logEl.scrollTop = logEl.scrollHeight;
  const shotsEl = document.getElementById('shots');
  const shots = s.shots || [];
  for (let i = shotsSeen; i < shots.length; i++) {
    const img = document.createElement('img');
    img.src = shots[i]; img.loading = 'lazy';
    shotsEl.appendChild(img);
  }
  shotsSeen = shots.length;
  if (s.state === 'done' || s.state === 'error') {
    clearInterval(timer); timer = null;
    document.getElementById('summary').textContent =
      (s.summary && s.summary.text) || ('state: ' + s.state);
    const btn = document.getElementById('start');
    btn.disabled = false; btn.textContent = 'Start run';
    document.getElementById('takeControl').disabled = true;
    controlling = false; updateControlUI();
  }
}
</script>
</body>
</html>
"""

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, threaded=True)
