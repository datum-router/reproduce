#!/usr/bin/env python3
"""CLI for the self-serve job application browser agent.

Usage:
    python run.py --url <job posting URL> [--headed] [--auto-submit]
                  [--max-steps 40]

Examples:
    # Headless dry run: fills the form, stops before Submit.
    python run.py --url https://example.com/jobs/123

    # Visible browser so you can watch (and log in by hand if needed).
    python run.py --url https://example.com/jobs/123 --headed

    # Full run: after filling, prints a summary and asks you to type
    # SUBMIT before it clicks the final submit button.
    python run.py --url https://example.com/jobs/123 --headed --auto-submit
"""

import argparse
import json
import os
import sys

from agent import Agent

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(
        description="Self-serve job application browser agent")
    ap.add_argument("--url", required=True, help="Job posting URL to open")
    ap.add_argument("--headed", action="store_true",
                    help="Show a visible browser window")
    ap.add_argument("--auto-submit", action="store_true",
                    help="After filling, ask for SUBMIT confirmation and then "
                         "click the final submit button. Default: stop before it.")
    ap.add_argument("--max-steps", type=int, default=40,
                    help="Max agent steps per run (default 40)")
    args = ap.parse_args()

    profile_path = os.path.join(BASE_DIR, "profile.json")
    if not os.path.exists(profile_path):
        print("profile.json not found.")
        print("Copy the example and fill in your details first:")
        print("  cp profile.example.json profile.json")
        sys.exit(1)
    with open(profile_path) as f:
        profile = json.load(f)
    with open(os.path.join(BASE_DIR, "rules.md")) as f:
        rules = f.read()

    agent = Agent(url=args.url, profile=profile, rules=rules,
                  headed=args.headed, auto_submit=args.auto_submit,
                  max_steps=args.max_steps)
    agent.run()


if __name__ == "__main__":
    sys.exit(main())
