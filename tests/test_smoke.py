#!/usr/bin/env python3
"""Smoke tests for the job-apply agent.

Covers: every .py file compiles, both profiles are valid JSON, the
DOM snapshot JS extracts a fake form correctly, the LLM action schema
parses (and rejects garbage), and the Flask app's routes respond.

Run:  python tests/test_smoke.py
"""

import json
import os
import py_compile
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "webapp"))

FAKE_FORM = """<html><body>
<form>
  <label>First name <input type="text" name="first" placeholder="First"></label>
  <label>Email <input type="email" name="email"></label>
  <label>Country <select name="country">
    <option value="">Pick one</option>
    <option value="us">United States</option>
    <option value="ca">Canada</option>
  </select></label>
  <label><input type="checkbox" name="ok"> I agree</label>
  <input type="hidden" name="token" value="abc">
  <button type="submit">Submit Application</button>
  <button type="button">Next</button>
</form>
</body></html>"""


def test_py_compile():
    files = ["agent.py", "llm_client.py", "run.py",
             os.path.join("webapp", "app.py"), __file__]
    for rel in files:
        py_compile.compile(os.path.join(BASE_DIR, rel), doraise=True)
    print("PASS py_compile (%d files)" % len(files))


def test_profiles_valid_json():
    # profile.json is gitignored, so fresh clones (CI) only have the example.
    checked = []
    for name in ("profile.json", "profile.example.json"):
        path = os.path.join(BASE_DIR, name)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        assert isinstance(data, dict), name
        assert data["name"]["first"], name
        assert "@" in data["email"], name
        checked.append(name)
    assert "profile.example.json" in checked, "example profile missing"
    print("PASS profiles are valid JSON with expected keys (%s)"
          % ", ".join(checked))


def test_parse_steps():
    from agent import Agent
    ok = Agent.parse_steps('[{"action":"fill","id":"e3","text":"John"}]')
    assert ok == [{"action": "fill", "id": "e3", "text": "John"}], ok
    fenced = Agent.parse_steps(
        '```json\n[{"action":"click","id":"e5"}]\n```')
    assert fenced == [{"action": "click", "id": "e5"}], fenced
    single = Agent.parse_steps('{"action":"done","reason":"x"}')
    assert single == [{"action": "done", "reason": "x"}], single
    assert Agent.parse_steps("not json at all") is None
    assert Agent.parse_steps('[{"action":"teleport","id":"e1"}]') is None
    assert Agent.parse_steps('{"action":"nope"}') is None
    print("PASS parse_steps accepts valid, rejects garbage/unknown actions")


def test_snapshot_js():
    from playwright.sync_api import sync_playwright
    from agent import SNAPSHOT_JS
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page()
        page.set_content(FAKE_FORM)
        items = page.evaluate(SNAPSHOT_JS)
        browser.close()
    by_name = {it.get("name"): it for it in items}
    first = by_name.get("First name")
    assert first and first["id"].startswith("e"), items
    assert first["tag"] == "input" and first["type"] == "text", first
    email = by_name.get("Email")
    assert email and email["type"] == "email", email
    country = next((it for it in items if it.get("type") == "select"), None)
    assert country and country["name"].startswith("Country"), country
    values = [o["value"] for o in country["options"]]
    assert "us" in values and "ca" in values, country
    agree = by_name.get("I agree")
    assert agree and agree["type"] == "checkbox" \
        and agree["checked"] is False, agree
    submit = by_name.get("Submit Application")
    assert submit and submit["tag"] == "button", submit
    nxt = by_name.get("Next")
    assert nxt, "Next button missing"
    # hidden input must be excluded
    assert not any(it.get("name") == "abc" for it in items), items
    print("PASS snapshot JS extracts %d elements correctly" % len(items))


def test_submit_detection():
    from agent import Agent
    a = Agent.__new__(Agent)  # no browser needed for the regex check
    assert a.looks_like_submit({"tag": "button",
                                "name": "Submit Application"})
    assert a.looks_like_submit({"tag": "input", "type": "submit",
                                "name": "Apply"})
    assert not a.looks_like_submit({"tag": "button", "name": "Next"})
    assert not a.looks_like_submit({"tag": "button", "name": "Continue"})
    print("PASS submit-button detection (blocks final, allows Next)")


def test_index_profile_is_valid_json():
    """Regression: the profile prefilled into the index page textarea must
    still parse as JSON after HTML escaping. A past bug escaped $ as \\$,
    which made the browser's JSON.parse reject the profile."""
    import html as htmlmod
    import re
    from app import app
    client = app.test_client()
    r = client.get("/")
    assert r.status_code == 200, r.status_code
    m = re.search(rb'<textarea id="profile" rows="12">(.*?)</textarea>',
                  r.data, re.S)
    assert m, "profile textarea missing from index page"
    profile = json.loads(htmlmod.unescape(m.group(1).decode()))
    assert isinstance(profile, dict), type(profile)
    assert profile["name"]["first"], profile.get("name")
    assert "$80k" in json.dumps(profile), "dollar amounts must survive"
    print("PASS index page profile textarea parses as JSON")


def test_llm_retry_then_success():
    import llm_client
    from unittest.mock import patch

    calls = []

    class R500:
        status_code = 500
        text = '{"error":"ENOSPC: no space left on device, write"}'

    class R200:
        status_code = 200
        text = ""

        def json(self):
            return {"choices": [{"message": {"content": "hello"}}]}

    def fake_post(*a, **k):
        calls.append(1)
        return R500() if len(calls) < 3 else R200()

    with patch.object(llm_client.requests, "post", side_effect=fake_post), \
         patch.object(llm_client.time, "sleep",
                      return_value=None) as sleep_mock:
        text = llm_client.chat([{"role": "user", "content": "hi"}])
    assert text == "hello", text
    assert len(calls) == 3, calls
    assert sleep_mock.call_count == 2, sleep_mock.call_count
    assert sleep_mock.call_args_list[0][0][0] == 2
    assert sleep_mock.call_args_list[1][0][0] == 5
    print("PASS llm retry: 2x500 then success (3 attempts, backoff 2s/5s)")


def test_llm_400_fails_fast():
    import llm_client
    from unittest.mock import patch

    calls = []

    class R400:
        status_code = 400
        text = '{"error":"bad request"}'

    def fake_post(*a, **k):
        calls.append(1)
        return R400()

    with patch.object(llm_client.requests, "post", side_effect=fake_post), \
         patch.object(llm_client.time, "sleep",
                      return_value=None) as sleep_mock:
        try:
            llm_client.chat([{"role": "user", "content": "hi"}])
        except llm_client.LLMError as exc:
            assert "400" in str(exc), exc
        else:
            raise AssertionError("expected LLMError on 400")
    assert len(calls) == 1, calls
    assert sleep_mock.call_count == 0
    print("PASS llm 400 fails fast (1 attempt, no retry)")


def test_llm_connection_error_retries():
    import llm_client
    import requests as req_mod
    from unittest.mock import patch

    calls = []

    class R200:
        status_code = 200
        text = ""

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

    def fake_post(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            raise req_mod.ConnectionError("boom")
        return R200()

    with patch.object(llm_client.requests, "post", side_effect=fake_post), \
         patch.object(llm_client.time, "sleep", return_value=None):
        text = llm_client.chat([{"role": "user", "content": "hi"}])
    assert text == "ok", text
    assert len(calls) == 2, calls
    print("PASS llm connection error retried, then success")


def test_control_endpoints():
    import queue as queue_mod
    import threading
    from app import app, runs, runs_lock

    run_id = "aa11bb22cc33"
    q = queue_mod.Queue()
    ev = threading.Event()
    rec = {"run_id": run_id, "state": "running", "paused": False,
           "url": "https://example.com", "max_steps": 5, "steps_done": 0,
           "log": [], "shots": [], "summary": None, "started": "now",
           "control_queue": q, "paused_event": ev}
    with runs_lock:
        runs[run_id] = rec
    try:
        client = app.test_client()
        # pause sets the flag
        r = client.post("/api/runs/%s/pause" % run_id)
        assert r.status_code == 200, r.status_code
        assert ev.is_set()
        assert r.get_json() == {"paused": True}
        # unknown op -> 400
        r = client.post("/api/runs/%s/control" % run_id,
                        json={"op": "teleport"})
        assert r.status_code == 400, r.status_code
        # non-numeric coords -> 400
        r = client.post("/api/runs/%s/control" % run_id,
                        json={"op": "click", "x": "left", "y": 5})
        assert r.status_code == 400, r.status_code
        # valid click -> queued (auto-pauses too)
        r = client.post("/api/runs/%s/control" % run_id,
                        json={"op": "click", "x": 100, "y": 200})
        assert r.status_code == 200, r.status_code
        assert q.get_nowait() == {"op": "click", "x": 100.0, "y": 200.0}
        # type / press / scroll
        r = client.post("/api/runs/%s/control" % run_id,
                        json={"op": "type", "text": "hello"})
        assert r.status_code == 200, r.status_code
        assert q.get_nowait()["op"] == "type"
        r = client.post("/api/runs/%s/control" % run_id,
                        json={"op": "press", "key": "Tab"})
        assert r.status_code == 200, r.status_code
        assert q.get_nowait() == {"op": "press", "key": "Tab"}
        r = client.post("/api/runs/%s/control" % run_id,
                        json={"op": "scroll", "dx": 0, "dy": 300})
        assert r.status_code == 200, r.status_code
        assert q.get_nowait() == {"op": "scroll", "dx": 0.0, "dy": 300.0}
        # resume clears the flag
        r = client.post("/api/runs/%s/resume" % run_id)
        assert r.status_code == 200, r.status_code
        assert not ev.is_set()
        assert r.get_json() == {"paused": False}
        # status reflects paused flag
        r = client.get("/api/runs/%s/status" % run_id)
        assert r.status_code == 200, r.status_code
        assert r.get_json()["paused"] is False
        # unknown run -> 404
        r = client.post("/api/runs/ffffffffffff/control",
                        json={"op": "click", "x": 1, "y": 2})
        assert r.status_code == 404, r.status_code
        r = client.post("/api/runs/ffffffffffff/pause")
        assert r.status_code == 404, r.status_code
        r = client.get("/runs/zzz-not-hex/live.png")
        assert r.status_code == 404, r.status_code
        r = client.get("/runs/000000000000/live.png")
        assert r.status_code == 404, r.status_code
        # inactive run -> 409
        with runs_lock:
            rec["state"] = "done"
        r = client.post("/api/runs/%s/control" % run_id,
                        json={"op": "click", "x": 1, "y": 2})
        assert r.status_code == 409, r.status_code
        r = client.post("/api/runs/%s/pause" % run_id)
        assert r.status_code == 409, r.status_code
        r = client.post("/api/runs/%s/resume" % run_id)
        assert r.status_code == 409, r.status_code
    finally:
        with runs_lock:
            runs.pop(run_id, None)
    print("PASS control endpoints (pause/resume/control/live.png, "
          "400/404/409)")


def test_llm_client_explicit_args():
    import llm_client
    from unittest.mock import patch

    # Explicit args win over env defaults.
    c = llm_client.LLMClient(base_url="https://example-llm.test/v1/",
                             model="test-model", api_key="sk-test-123")
    assert c.base_url == "https://example-llm.test/v1", c.base_url
    assert c.model == "test-model", c.model
    assert c.api_key == "sk-test-123", c.api_key

    # None falls back to env defaults.
    d = llm_client.LLMClient()
    assert d.base_url == llm_client.BASE_URL, d.base_url
    assert d.model == llm_client.MODEL, d.model
    assert d.api_key == llm_client.API_KEY, d.api_key

    # The override actually drives the HTTP call (url, header, body model).
    seen = {}

    class R200:
        status_code = 200
        text = ""

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

    def fake_post(url, headers=None, data=None, timeout=None):
        seen["url"] = url
        seen["headers"] = headers
        seen["body"] = json.loads(data)
        return R200()

    with patch.object(llm_client.requests, "post", side_effect=fake_post):
        text = c.chat([{"role": "user", "content": "hi"}])
    assert text == "ok", text
    assert seen["url"] == "https://example-llm.test/v1/chat/completions", \
        seen["url"]
    assert seen["headers"].get("Authorization") == "Bearer sk-test-123", \
        seen["headers"]
    assert seen["body"]["model"] == "test-model", seen["body"]
    print("PASS llm client honors explicit args over env (url/header/model)")


def test_api_runs_llm_overrides():
    import app as app_mod
    from unittest.mock import patch

    captured = {}

    def fake_run_in_thread(run_id, url, profile, rules_text, max_steps,
                           control_queue, paused, llm_client=None):
        captured["run_id"] = run_id
        captured["llm_client"] = llm_client

    client = app_mod.app.test_client()
    with patch.object(app_mod, "_run_in_thread",
                      side_effect=fake_run_in_thread):
        # With overrides: client built, key held, never leaked.
        r = client.post("/api/runs", json={
            "url": "https://example.com/jobs/1",
            "profile": {"name": {"first": "John"}},
            "llm_base_url": "https://example-llm.test/v1",
            "llm_model": "test-model",
            "llm_api_key": "sk-secret-xyz",
        })
        assert r.status_code == 200, r.status_code
        run_id = r.get_json()["run_id"]
        lc = captured["llm_client"]
        assert lc is not None, "expected an LLMClient with overrides"
        assert lc.base_url == "https://example-llm.test/v1", lc.base_url
        assert lc.model == "test-model", lc.model
        assert lc.api_key == "sk-secret-xyz", lc.api_key

        # The status endpoint must never leak the key.
        s = client.get("/api/runs/%s/status" % run_id).get_json()
        blob = json.dumps(s)
        assert "sk-secret-xyz" not in blob, blob
        assert "llm_api_key" not in blob, blob
        # The redacted override note may mention base_url/model, not the key.
        assert not any("sk-secret-xyz" in line for line in s["log_tail"])

        # Without overrides: no client, server defaults in effect.
        captured.clear()
        r = client.post("/api/runs", json={
            "url": "https://example.com/jobs/2",
            "profile": {"name": {"first": "John"}},
        })
        assert r.status_code == 200, r.status_code
        assert captured["llm_client"] is None, captured["llm_client"]
    print("PASS /api/runs accepts llm overrides, status never leaks the key")


def test_flask_routes():
    from app import app
    client = app.test_client()
    r = client.get("/")
    assert r.status_code == 200, r.status_code
    assert b"Job Apply Agent" in r.data
    r = client.post("/api/runs", json={})
    assert r.status_code == 400, r.status_code
    r = client.post("/api/runs", json={"url": "not-a-url",
                                       "profile": "{}"})
    assert r.status_code == 400, r.status_code
    r = client.post("/api/runs", json={"url": "https://example.com/j",
                                       "profile": "{bad json"})
    assert r.status_code == 400, r.status_code
    r = client.get("/api/runs/doesnotexist/status")
    assert r.status_code == 404, r.status_code
    r = client.get("/runs/doesnotexist/shots/step_001.png")
    assert r.status_code == 404, r.status_code
    print("PASS flask routes (index, validation, 404s) - no browser launched")


def main():
    test_py_compile()
    test_profiles_valid_json()
    test_parse_steps()
    test_snapshot_js()
    test_submit_detection()
    test_index_profile_is_valid_json()
    test_llm_retry_then_success()
    test_llm_400_fails_fast()
    test_llm_connection_error_retries()
    test_control_endpoints()
    test_llm_client_explicit_args()
    test_api_runs_llm_overrides()
    test_flask_routes()
    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
