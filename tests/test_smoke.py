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
    for name in ("profile.json", "profile.example.json"):
        with open(os.path.join(BASE_DIR, name)) as f:
            data = json.load(f)
        assert isinstance(data, dict), name
        assert data["name"]["first"], name
        assert "@" in data["email"], name
    print("PASS profiles are valid JSON with expected keys")


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
    test_flask_routes()
    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
