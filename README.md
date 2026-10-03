# Job Apply Agent

A self-serve browser agent that fills out job application forms for you.
You give it a job posting URL; it opens a real Chromium browser, reads the
page, and uses your own LLM to decide what to click and what to type,
field by field. Your profile data is preloaded from `profile.json` and your
standing rules from `rules.md`.

## Setup

```bash
cd ~/workspace/job-apply-agent
pip install -r requirements.txt
python -m playwright install chromium   # downloads the browser (~170 MB)
```

The `--only-shell` variant (`python -m playwright install chromium --only-shell`)
is enough for headless runs.

## Configuration

Environment variables (all optional):

| Var            | Default                              | Notes                              |
|----------------|--------------------------------------|------------------------------------|
| `LLM_BASE_URL` | `https://text.pollinations.ai/openai`| Any OpenAI-compatible endpoint     |
| `LLM_MODEL`    | `openai`                             | Model name for the endpoint        |
| `LLM_API_KEY`  | _(empty)_                             | Needed for most providers; Pollinations needs none |

## Profile setup

`profile.json` holds your personal data and is **gitignored**, so it never
lands in the repo. Start from the example:

```bash
cp profile.example.json profile.json
# then edit profile.json with your real details
```

It carries your name, contact, work authorization, EEO answers, education,
jobs, links, and standing form answers. `rules.md` describes how the agent
uses them. Edit either file; the agent reads them fresh on every run.

Passwords are never stored: if a login form appears, the agent prompts for
the password at runtime with `getpass` and never writes it to disk.

## Usage

```bash
# Dry run: fills the form, stops before the final Submit button.
python run.py --url https://example.com/jobs/123

# Visible browser: watch it work, log in by hand if a site needs it.
# Logins persist between runs in ./browser-profile.
python run.py --url https://example.com/jobs/123 --headed

# Full run: after filling, prints a summary and asks you to type SUBMIT
# before it clicks the final submit button.
python run.py --url https://example.com/jobs/123 --headed --auto-submit

# Bigger step budget for long multi-page forms.
python run.py --url https://example.com/jobs/123 --max-steps 80
```

## How it works: observe / decide / act

1. **Observe.** `agent.py` injects JavaScript into the page that walks the
   DOM and collects every *visible* interactive element (`input`,
   `textarea`, `select`, `button`, links, ARIA buttons/checkboxes/radios).
   Each element is tagged with a `data-agent-id` (`e1`, `e2`, ...) and
   reported back as a compact list: id, tag, type, label, placeholder,
   current value, checked state, and dropdown options. The snapshot is
   re-taken after every action, so the agent always sees the current page.

2. **Decide.** The snapshot, your profile, and your rules are sent to the
   LLM with a strict system prompt: reply with a JSON array of steps and
   nothing else. The allowed actions are `click`, `fill`, `select`,
   `check`, `uncheck`, `press`, `wait`, `scroll`, `done`, and
   `need_human`. The prompt forbids guessing: unknown fields stay blank
   and get logged; CAPTCHAs, login walls, and video submissions return
   `need_human`.

3. **Act.** Each step is executed with Playwright against
   `[data-agent-id="..."]`. Special cases: file inputs use
   `set_input_files` with your resume path; password fields trigger a
   runtime `getpass` prompt; `select` matches the option value (falling
   back to visible text). A screenshot lands in `shots/` after every
   step, and every observation, model reply, and action is appended to
   `runs/<timestamp>/log.jsonl`.

## Safety model

- **Default mode never submits.** When the form is fully filled the agent
  prints a summary of every filled field and exits *before* the final
  Submit/Apply button.
- **`--auto-submit` still asks.** It prints the same summary first and
  only proceeds when you literally type `SUBMIT`.
- The agent also refuses to click anything that looks like a final submit
  button mid-run unless submission was confirmed.
- Unknown fields are left blank and logged, never invented.
- One application per company and the other rules in `rules.md` are part
  of the model's instructions.

## Limitations (v1)

- Main frame only: forms rendered inside cross-origin iframes are not
  snapshotted yet.
- No CAPTCHA solving: the agent stops and hands control to you.
- The model only sees the element list, not a screenshot; heavily
  visual or canvas-based widgets may confuse it.
- Multi-page "Next" flows work, but very long forms may need a larger
  `--max-steps`.
