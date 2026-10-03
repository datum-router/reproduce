# Test notes

## What was tested (2026-10-03)

1. **Syntax + data validity** — `python -m py_compile` on `llm_client.py`,
   `agent.py`, `run.py`: all pass. `profile.json` and `profile.example.json`
   both parse as JSON with identical key structure.

2. **DOM snapshot + executor** — launched real Chromium (headless shell)
   against a local fake application form (`/tmp/job-apply-test/form.html`,
   not shipped) with text inputs, email, select, radios, checkboxes,
   textarea, file upload, hidden/display:none traps, and Next/Submit
   buttons. 29/29 checks passed:
   - Snapshot found all 14 visible interactive elements with stable
     `e1..e14` ids; hidden and `display:none` inputs excluded.
   - Labels resolved from `<label for>`, wrapped labels, and button text.
   - Select reported all 3 options; checkbox reported `checked=True`.
   - Canned JSON action list (no LLM) filled every field; re-snapshot
     confirmed DOM values changed; `set_input_files` attached the resume.
   - `looks_like_submit` matched "Submit Application" but not "Next".
   - Default mode blocked the final submit click (`submit_blocked`);
     `--auto-submit` mode deferred to confirmation (`submit_requested`).
   - `parse_steps` accepted plain JSON and fenced JSON; rejected garbage
     and unknown actions.

3. **LLM connectivity** — `llm_client.health_check()` against the default
   Pollinations endpoint returned `ok`. Anonymous, no key needed.

4. **Browser install** — `python -m playwright install chromium --only-shell`
   worked (~60s). Full `playwright install chromium` was not needed.

5. **Public-safety review** — no real personal data anywhere in the project
   (verified by grep for name/email/phone/address/handle). `profile.json`
   holds an obvious "John Doe" test profile, `profile.example.json` is the
   copy-me template, and `profile.json` is gitignored. The real profile
   gets wired in later, after testing.

## Known gaps for v2

- Main frame only: forms inside cross-origin iframes are not snapshotted.
  (Many ATS forms, e.g. Taleo, render in iframes.)
- No CAPTCHA handling: the agent stops with `need_human`.
- The model sees the element list, not a screenshot; canvas-heavy or
  heavily custom widgets may confuse it. A vision pass (screenshot +
  element bounding boxes) would help.
- `fill` falls back to click+type on failure, but some React-controlled
  inputs may need native value-setter tricks.
- No retry/backoff on LLM rate limits; a 429 surfaces as `llm_error`.
- No per-site adapters: date pickers, autocomplete comboboxes, and
  multi-file uploads are handled generically and may need site-specific
  steps.
- `--headed` needs a display; on a headless server use xvfb or X forwarding.
