#!/usr/bin/env python3
"""Core observe/decide/act loop for the job-apply browser agent.

The agent works in a tight loop:

    1. OBSERVE  - inject JS into the page, snapshot every visible
                  interactive element (each gets a data-agent-id).
    2. DECIDE    - hand the snapshot + profile + rules to the LLM, which
                  replies with a JSON array of steps.
    3. ACT       - execute each step with Playwright, screenshot it, log it.

Repeat until the LLM says "done" (form filled), "need_human" (stuck),
or we run out of steps.
"""

import datetime
import getpass
import json
import os
import re
import sys

from playwright.sync_api import sync_playwright

from llm_client import chat, LLMError

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
BROWSER_PROFILE_DIR = os.path.join(PROJECT_DIR, "browser-profile")
SHOTS_DIR = os.path.join(PROJECT_DIR, "shots")
RUNS_DIR = os.path.join(PROJECT_DIR, "runs")

# Buttons that look like the FINAL submit of an application. "Next",
# "Continue", "Save" etc. are deliberately NOT in this list.
SUBMIT_RE = re.compile(
    r"submit(\s+(my|this))?\s+application"
    r"|complete\s+application"
    r"|send\s+application"
    r"|apply\s+for\s+this\s+job"
    r"|finish(\s+application)?$",
    re.IGNORECASE,
)

# JS that walks the DOM and returns every visible interactive element.
# Each element is tagged with a data-agent-id like "e3" so the LLM can
# refer to it and Playwright can locate it again.
SNAPSHOT_JS = """
() => {
  const out = [];
  let n = 0;
  const seen = new Set();
  const isVisible = (el) => {
    const r = el.getBoundingClientRect();
    if (!r || (r.width === 0 && r.height === 0)) return false;
    const cs = window.getComputedStyle(el);
    if (cs.display === "none" || cs.visibility === "hidden") return false;
    if (parseFloat(cs.opacity || "1") === 0) return false;
    return true;
  };
  const labelOf = (el) => {
    const aria = el.getAttribute("aria-label");
    if (aria && aria.trim()) return aria.trim().slice(0, 100);
    if (el.labels && el.labels.length) {
      const t = Array.from(el.labels).map((l) => (l.innerText || "").trim())
        .filter(Boolean).join(" ");
      if (t) return t.slice(0, 100);
    }
    const lb = el.getAttribute("aria-labelledby");
    if (lb) {
      const t = lb.split(/\\s+/).map((id) => {
        const t2 = document.getElementById(id);
        return t2 ? (t2.innerText || "").trim() : "";
      }).filter(Boolean).join(" ");
      if (t) return t.slice(0, 100);
    }
    const tag = el.tagName.toLowerCase();
    const itype = (el.type || "").toLowerCase();
    if (tag === "input" && ["submit", "button", "reset"].includes(itype)) {
      return (el.value || "").trim().slice(0, 100);
    }
    if (el.placeholder) return el.placeholder.trim().slice(0, 100);
    if (el.title) return el.title.trim().slice(0, 100);
    const txt = (el.innerText || "").trim().replace(/\\s+/g, " ");
    if (txt) return txt.slice(0, 100);
    return "";
  };
  const q = document.querySelectorAll(
    'input, textarea, select, button, a, [role="button"], [role="link"], ' +
    '[role="checkbox"], [role="radio"], [role="combobox"], [role="listbox"], ' +
    '[contenteditable="true"]'
  );
  for (const el of q) {
    if (seen.has(el)) continue;
    seen.add(el);
    const tag = el.tagName.toLowerCase();
    const itype = (el.type || "").toLowerCase();
    if (tag === "input" && itype === "hidden") continue;
    if (el.disabled) continue;
    if (!isVisible(el)) continue;
    n += 1;
    const id = "e" + n;
    el.setAttribute("data-agent-id", id);
    const item = { id: id, tag: tag, name: labelOf(el) };
    if (tag === "input") {
      item.type = itype || "text";
      if (item.type === "checkbox" || item.type === "radio") {
        item.checked = !!el.checked;
        item.value = el.value || "on";
      } else if (item.type === "password") {
        item.value = el.value ? "(has value)" : "";
      } else if (item.type !== "file") {
        item.value = el.value || "";
      }
      if (el.placeholder) item.placeholder = el.placeholder;
      if (el.required) item.required = true;
    } else if (tag === "textarea") {
      item.type = "textarea";
      item.value = el.value || "";
      if (el.placeholder) item.placeholder = el.placeholder;
      if (el.required) item.required = true;
    } else if (tag === "select") {
      item.type = "select";
      item.options = Array.from(el.options).map((o) => ({
        value: o.value,
        text: (o.text || "").trim().slice(0, 80),
      })).slice(0, 60);
      item.value = el.value;
      if (el.required) item.required = true;
    } else if (tag === "a") {
      item.type = "link";
      const href = el.getAttribute("href") || "";
      if (href) item.href = href.slice(0, 120);
    } else if (el.getAttribute("contenteditable") === "true") {
      item.type = "contenteditable";
      item.value = (el.innerText || "").trim().slice(0, 200);
    } else {
      item.type = el.getAttribute("role") || (tag === "button" ? "button" : tag);
    }
    out.push(item);
  }
  return out;
}
"""

ALLOWED_ACTIONS = {
    "click", "fill", "select", "check", "uncheck",
    "press", "wait", "scroll", "done", "need_human",
}


class Agent:
    def __init__(self, url, profile, rules, headed=False,
                 auto_submit=False, max_steps=40, llm_client=None):
        self.url = url
        self.profile = profile
        self.rules = rules
        self.headed = headed
        self.auto_submit = auto_submit
        self.max_steps = max_steps
        # Optional LLMClient with per-run overrides (base_url/model/api_key).
        # None means "use the module-level env-configured chat()".
        self.llm_client = llm_client

        self.run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(RUNS_DIR, self.run_id)
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(SHOTS_DIR, exist_ok=True)
        self.log_path = os.path.join(self.run_dir, "log.jsonl")

        self.filled = []            # summary of filled fields for the report
        self.history = []           # compact action history for the LLM
        self.steps_executed = 0
        self.submit_confirmed = False

        self.pw = None
        self.context = None
        self.page = None

    # ---------- logging ----------

    def log(self, event, **fields):
        rec = {"t": datetime.datetime.now().isoformat(timespec="seconds"),
               "event": event}
        rec.update(fields)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    # ---------- browser ----------

    def start(self):
        self.pw = sync_playwright().start()
        # Persistent context: anything the human logs into by hand in the
        # visible browser survives between runs.
        self.context = self.pw.chromium.launch_persistent_context(
            BROWSER_PROFILE_DIR,
            headless=not self.headed,
            viewport={"width": 1366, "height": 900},
        )
        self.context.set_default_timeout(10000)
        self.page = self.context.pages[0] if self.context.pages \
            else self.context.new_page()
        print("Opening %s" % self.url)
        self.page.goto(self.url, wait_until="domcontentloaded")
        self.log("start", url=self.url, headed=self.headed,
                 auto_submit=self.auto_submit)

    def close(self):
        try:
            if self.context:
                self.context.close()
        finally:
            if self.pw:
                self.pw.stop()

    # ---------- observe ----------

    def snapshot(self):
        items = self.page.evaluate(SNAPSHOT_JS)
        by_id = {it["id"]: it for it in items}
        self.log("snapshot", url=self.page.url, count=len(items))
        return items, by_id

    def looks_like_submit(self, item):
        """True if this element looks like the FINAL submit button."""
        name = (item.get("name") or "")
        if item.get("tag") == "input" and item.get("type") == "submit":
            return True
        return bool(SUBMIT_RE.search(name))

    # ---------- decide ----------

    def system_prompt(self):
        name = self.profile.get("name", {})
        full_name = ("%s %s" % (name.get("first", ""),
                                name.get("last", ""))).strip() or "the applicant"
        return (
            "You are a browser automation agent that fills out job application "
            "forms for " + full_name + ".\n\n"
            "GOAL: Fill the current job application page completely and "
            "accurately using the profile below. Explore the page, fill every "
            "required field, upload the resume where asked, and work through "
            "multi-step forms (Next/Continue buttons are fine to click).\n\n"
            "PROFILE (use exactly these values):\n"
            + json.dumps(self.profile, indent=2) + "\n\n"
            "RULES (follow exactly):\n" + self.rules + "\n\n"
            "HOW YOU SEE THE PAGE: after every action you get a fresh "
            "snapshot: a JSON list of visible interactive elements. Each has "
            "an id like \"e3\", tag, type, name/label, placeholder, current "
            "value, checked state, and for selects the list of options.\n\n"
            "YOUR OUTPUT: a JSON array of steps, and NOTHING else. No prose, "
            "no markdown fences. Example:\n"
            "[{\"action\":\"fill\",\"id\":\"e3\",\"text\":\"John\","
            "\"reason\":\"first name field\"}]\n\n"
            "ALLOWED ACTIONS:\n"
            "- {\"action\":\"click\",\"id\":\"e5\",\"reason\":\"...\"}\n"
            "- {\"action\":\"fill\",\"id\":\"e3\",\"text\":\"...\",\"reason\":\"...\"} "
            "(for file-upload inputs put the resume file path in \"text\")\n"
            "- {\"action\":\"select\",\"id\":\"e7\",\"value\":\"...\",\"reason\":\"...\"} "
            "(use the exact option \"value\" from the snapshot)\n"
            "- {\"action\":\"check\",\"id\":\"e9\",\"reason\":\"...\"} / "
            "{\"action\":\"uncheck\",\"id\":\"e9\",\"reason\":\"...\"}\n"
            "- {\"action\":\"press\",\"key\":\"Enter\",\"reason\":\"...\"}\n"
            "- {\"action\":\"wait\",\"ms\":1500,\"reason\":\"...\"}\n"
            "- {\"action\":\"scroll\",\"id\":\"e5\",\"reason\":\"...\"}\n"
            "- {\"action\":\"done\",\"reason\":\"...\"} (form fully filled; "
            "do NOT click the final submit button yourself)\n"
            "- {\"action\":\"need_human\",\"reason\":\"...\"} (you are stuck)\n\n"
            "RULES OF ENGAGEMENT:\n"
            "- Keep each reply to 1-5 steps, one coherent chunk per turn.\n"
            "- NEVER invent or guess an answer. A field with no answer in the "
            "profile and no safe default stays blank; name it in your \"done\" "
            "reason so it gets logged.\n"
            "- Answer Yes/True to qualification, experience, and skill "
            "questions. Use the standing answers for work authorization, "
            "relocation, travel, etc.\n"
            "- Voluntary EEO fields: Male / South Asian (or Asian) / "
            "Heterosexual / veteran No / disability No. Never decline.\n"
            "- Phone type is mobile/cell, never home. Use the applicant's "
            "name exactly as given in profile.json.\n"
            "- For resume uploads, \"fill\" with the resume_path from the "
            "profile; the agent attaches the file itself.\n"
            "- If a password field appears, fill it normally; the agent "
            "securely prompts the human for the password at runtime "
            "(never stored, never written to disk).\n"
            "- CAPTCHA, login wall you cannot pass, video submission, or a "
            "question with no safe answer: output need_human with a clear "
            "reason. Do not guess through a blocker.\n"
            "- NEVER click the final Submit/Apply button. When everything is "
            "filled, output a single done step. The human reviews and submits.\n"
            "- Sponsorship gate: if the page or form says the employer will "
            "not sponsor (no H-1B, citizenship required, unable to sponsor), "
            "output need_human with reason \"sponsorship refused\" and stop."
        )

    def decide(self, items, extra_instruction=""):
        user_msg = (
            "Page URL: %s\nPage title: %s\n\n"
            "Interactive elements:\n%s\n\n"
            "Actions taken so far:\n%s\n\n"
            "%s"
            "Reply with the next JSON array of steps. JSON array only."
            % (self.page.url, self.page.title(),
               json.dumps(items, indent=1)[:12000],
               "\n".join(self.history[-15:]) or "(none)",
               (extra_instruction + "\n\n") if extra_instruction else "")
        )
        messages = [
            {"role": "system", "content": self.system_prompt()},
            {"role": "user", "content": user_msg},
        ]
        for attempt in (1, 2):
            try:
                chat_fn = (self.llm_client.chat if self.llm_client else chat)
                reply = chat_fn(messages)
            except LLMError as exc:
                self.log("llm_error", error=str(exc))
                return None
            self.log("llm_reply", attempt=attempt, reply=reply[:4000])
            steps = self.parse_steps(reply)
            if steps is not None:
                return steps
            messages.append({"role": "assistant", "content": reply})
            messages.append({
                "role": "user",
                "content": "That was not a valid JSON array of steps. "
                           "Reply again with ONLY the JSON array.",
            })
        self.log("llm_error", error="invalid JSON twice")
        return None

    @staticmethod
    def parse_steps(reply):
        text = reply.strip()
        # Tolerate markdown fences even though the prompt forbids them.
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
        try:
            steps = json.loads(text)
        except (ValueError, TypeError):
            return None
        if isinstance(steps, dict):
            steps = [steps]
        if not isinstance(steps, list):
            return None
        clean = []
        for s in steps:
            if not isinstance(s, dict):
                return None
            if s.get("action") not in ALLOWED_ACTIONS:
                return None
            clean.append(s)
        return clean

    # ---------- act ----------

    def loc(self, step_id):
        return self.page.locator('[data-agent-id="%s"]' % step_id)

    def execute_step(self, step, by_id):
        """Run one step. Returns ok | failed | submit_blocked | submit_requested."""
        action = step["action"]
        reason = step.get("reason", "")
        step_id = step.get("id")
        item = by_id.get(step_id, {}) if step_id else {}
        name = item.get("name", step_id or "")

        try:
            if action == "click":
                if self.looks_like_submit(item) and self.filled:
                    if self.auto_submit and self.submit_confirmed:
                        pass  # allowed: human already confirmed
                    elif self.auto_submit:
                        self.log("submit_requested", id=step_id, name=name)
                        return "submit_requested"
                    else:
                        self.log("submit_blocked", id=step_id, name=name)
                        return "submit_blocked"
                self.loc(step_id).click()

            elif action == "fill":
                text = step.get("text", "")
                if item.get("type") == "file":
                    path = os.path.expanduser(
                        text or self.profile.get("resume_path", ""))
                    if not path or not os.path.exists(path):
                        raise RuntimeError("upload file not found: %r" % path)
                    self.loc(step_id).set_input_files(path)
                    self.filled.append((name or "resume upload", path))
                elif item.get("type") == "password":
                    # Never from the LLM, never stored: ask at runtime.
                    pw = getpass.getpass(
                        "Password needed for '%s': " % (name or "login"))
                    self.loc(step_id).fill(pw)
                    self.filled.append((name or "password", "[entered, not stored]"))
                else:
                    try:
                        self.loc(step_id).fill(text)
                    except Exception:
                        # Custom widgets sometimes reject fill(); click + type.
                        self.loc(step_id).click()
                        self.page.keyboard.type(text)
                    self.filled.append((name or step_id, text))

            elif action == "select":
                value = step.get("value", "")
                try:
                    self.loc(step_id).select_option(value=value)
                except Exception:
                    self.loc(step_id).select_option(label=value)
                self.filled.append((name or step_id, value))

            elif action == "check":
                self.loc(step_id).check()
                self.filled.append((name or step_id, "checked"))

            elif action == "uncheck":
                self.loc(step_id).uncheck()
                self.filled.append((name or step_id, "unchecked"))

            elif action == "press":
                self.page.keyboard.press(step.get("key", "Enter"))

            elif action == "wait":
                self.page.wait_for_timeout(int(step.get("ms", 1000)))

            elif action == "scroll":
                self.loc(step_id).scroll_into_view_if_needed()

            else:
                raise RuntimeError("unhandled action: %s" % action)

        except Exception as exc:  # noqa: BLE001 - report, don't crash the run
            self.log("action_failed", action=action, id=step_id,
                     reason=reason, error=str(exc)[:300])
            self.history.append("- %s %s (%s) -> FAILED: %s"
                                % (action, step_id, name, str(exc)[:80]))
            return "failed"

        self.log("action", action=action, id=step_id, name=name, reason=reason)
        self.history.append("- %s %s (%s) -> ok" % (action, step_id, name))
        return "ok"

    def screenshot(self):
        self.steps_executed += 1
        path = os.path.join(
            SHOTS_DIR, "%s_step_%03d.png" % (self.run_id, self.steps_executed))
        try:
            self.page.screenshot(path=path)
        except Exception as exc:  # noqa: BLE001
            self.log("screenshot_failed", error=str(exc)[:200])

    # ---------- loop ----------

    def fill_loop(self):
        """Observe/decide/act until done / need_human / blocked / budget."""
        while self.steps_executed < self.max_steps:
            items, by_id = self.snapshot()
            steps = self.decide(items)
            if steps is None:
                return "llm_error"
            terminal = None
            for step in steps:
                action = step["action"]
                if action == "done":
                    self.log("done", reason=step.get("reason", ""))
                    return "done"
                if action == "need_human":
                    self.log("need_human", reason=step.get("reason", ""))
                    return "need_human:" + step.get("reason", "")
                result = self.execute_step(step, by_id)
                self.screenshot()
                if result in ("submit_blocked", "submit_requested"):
                    return result
                if self.steps_executed >= self.max_steps:
                    break
            if terminal:
                return terminal
        return "max_steps"

    def submit_round(self):
        """After the human confirms: let the LLM click the final submit."""
        for _ in range(5):
            items, by_id = self.snapshot()
            steps = self.decide(
                items,
                extra_instruction="The human has reviewed the filled form and "
                "confirmed submission. Click the final submit/apply button "
                "now, then output a done step.")
            if not steps:
                return "llm_error"
            for step in steps:
                if step["action"] == "done":
                    return "submitted"
                if step["action"] == "need_human":
                    return "need_human:" + step.get("reason", "")
                result = self.execute_step(step, by_id)
                self.screenshot()
                if result == "failed":
                    return "submit_failed"
        return "submit_failed"

    # ---------- reporting ----------

    def print_summary(self):
        print("\n===== FILLED FIELDS (%d) =====" % len(self.filled))
        for label, value in self.filled:
            shown = value if len(str(value)) < 60 else str(value)[:57] + "..."
            print("  %-40s %s" % (label[:40], shown))
        if not self.filled:
            print("  (nothing filled)")
        print("Log: %s" % self.log_path)
        print("Shots: %s" % SHOTS_DIR)

    def report(self, how):
        print("\n===== RESULT: %s =====" % how)
        if how == "stopped_before_submit":
            print("Stopped before the final Submit/Apply button (default "
                  "safety). Review the summary above; re-run with "
                  "--auto-submit to proceed.")
        elif how == "need_human" or how.startswith("need_human:"):
            print("The agent needs you: %s" % how.split(":", 1)[-1])
            print("The browser is left open in headed mode; handle it by hand.")
        elif how == "max_steps":
            print("Step budget (%d) exhausted. Re-run with --max-steps N."
                  % self.max_steps)
        elif how == "llm_error":
            print("The model call failed or returned invalid output twice. "
                  "Check the log and your LLM_* env vars.")
        elif how == "submitted":
            print("Submit clicked. Verify the confirmation page yourself.")

    def ask_submit_confirmation(self):
        answer = input("\nType SUBMIT to let the agent click the final "
                       "submit button (anything else aborts): ").strip()
        return answer == "SUBMIT"

    def run(self):
        self.start()
        try:
            how = self.fill_loop()
            self.print_summary()
            if how == "done" and self.auto_submit:
                if self.ask_submit_confirmation():
                    self.submit_confirmed = True
                    how = self.submit_round()
                    self.print_summary()
                else:
                    how = "submit_aborted_by_human"
            self.log("finished", how=how)
            self.report(how)
        finally:
            self.close()


def main():  # pragma: no cover - exercised via run.py
    print("Use run.py instead.", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
