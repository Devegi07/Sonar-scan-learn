import os
import requests
import zipfile
import io
import re
from transformers import pipeline

# ================= CONFIG =================
OWNER = "Devegi07"
REPO = "Monitoring_workflow"
BRANCH = "main"

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise RuntimeError("❌ GITHUB_TOKEN not set")

HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json"
}

# ================= LLM =================
llm = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    device=-1
)

# ================= AGENT =================
class GitHubMonitorAgent:

    # ---------- GitHub API Calls ----------

    def github_get(self, url, raw=False):
        r = requests.get(url, headers=HEADERS)
        if r.status_code != 200:
            print(f"⚠️ GitHub API request failed: {r.status_code} - {r.text}")
            return None
        return r.content if raw else r.json()

    def list_workflows(self):
        url = f"https://api.github.com/repos/{OWNER}/{REPO}/actions/workflows"
        data = self.github_get(url)
        return data.get("workflows", []) if data else []

    def latest_run(self, workflow_id):
        url = (
            f"https://api.github.com/repos/{OWNER}/{REPO}/actions/workflows/"
            f"{workflow_id}/runs?branch={BRANCH}&per_page=1"
        )
        data = self.github_get(url)
        runs = data.get("workflow_runs", []) if data else []
        return runs[0] if runs else None

    def download_logs(self, run_id):
        url = f"https://api.github.com/repos/{OWNER}/{REPO}/actions/runs/{run_id}/logs"
        logs = self.github_get(url, raw=True)
        if not logs:
            print(f"⚠️ Failed to download logs for run ID {run_id}")
        return logs

    # ---------- ERROR EXTRACTION ----------

    def extract_all_errors(self, zip_bytes):
        z = zipfile.ZipFile(io.BytesIO(zip_bytes))
        seen_errors = set()
        errors = []

        for name in z.namelist():
            text = z.read(name).decode(errors="ignore")
            lines = text.splitlines()

            for idx, line in enumerate(lines, start=1):

                # Ignore noise / metadata lines
                if re.search(
                    r"(Process completed with exit code|Job defined at:)",
                    line,
                    re.IGNORECASE
                ):
                    continue

                # Detect error lines
                if re.search(
                    r"(##\[error\]|ERROR|Exception|ZeroDivisionError|ModuleNotFoundError|No such file)",
                    line,
                    re.IGNORECASE
                ):
                    # Normalize error line: strip timestamps and "##[error]"
                    normalized = re.sub(
                        r"^\d{4}-\d{2}-\d{2}T.*?Z\s*##\[error\]\s*",
                        "",
                        line
                    ).strip()

                    # Deduplicate case-insensitive
                    if normalized.lower() in seen_errors:
                        continue

                    seen_errors.add(normalized.lower())

                    errors.append({
                        "job_step": name.replace(".txt", ""),
                        "line_no": idx,
                        "error_line": normalized
                    })

        return errors

    # ---------- RULE-BASED FIXES ----------

    def rule_based_fix(self, error_line):
        error_lower = error_line.lower()
       
        if "modulenotfounderror" in error_lower:
            return (
                "Cause: A required Python dependency is missing.\n"
                "Fix: Add the missing package to requirements.txt and ensure dependencies are installed in the workflow."
            )
        return None

    # ---------- AI-BASED FIX ----------

    def ai_fix(self, error_line):
        rule_fix = self.rule_based_fix(error_line)
        if rule_fix:
            return rule_fix

        prompt = (
            "You are a senior DevOps engineer.\n"
            "Explain the root cause and fix for this CI failure.\n\n"
            f"{error_line}\n\n"
            "Respond exactly in this format:\n"
            "Cause: <one sentence>\n"
            "Fix: <one sentence>"
        )

        response = llm(prompt, max_new_tokens=80)
        # response is a list of dicts, get generated text from first
        text = response[0].get("generated_text") or response[0].get("text") or ""
        text = text.strip()

        # fallback if output is unexpected
        if "Cause:" not in text or "Fix:" not in text:
            return (
                "Cause: The workflow failed due to a configuration or dependency issue.\n"
                "Fix: Review the error message and update the workflow or project files accordingly."
            )

        return text

    # ---------- MAIN MONITOR FUNCTION ----------

    def monitor(self):
        print("\n🤖 GitHub Monitor AI Agent Started\n")

        workflows = self.list_workflows()
        passed = 0
        failed = 0

        all_results = []

        for wf in workflows:
            run = self.latest_run(wf["id"])
            if not run:
                continue

            if run["conclusion"] == "success":
                passed += 1
                print(f"✅ {wf['name']} → Passed")
                continue

            if run["conclusion"] == "failure":
                failed += 1
                errors_info = []
                logs = self.download_logs(run["id"])
                if not logs:
                    print(f"⚠️ Could not download logs for run {run['id']}")
                    continue

                errors = self.extract_all_errors(logs)

                for idx, err in enumerate(errors, start=1):
                    fix = self.ai_fix(err['error_line'])
                    errors_info.append({
                        "error_number": idx,
                        "job_step": err['job_step'],
                        "line_no": err['line_no'],
                        "error_line": err['error_line'],
                        "suggestion": fix
                    })

                all_results.append({
                    "workflow_name": wf['name'],
                    "run_id": run['id'],
                    "errors": errors_info
                })

        # Print all failures with all errors grouped per workflow run
        for result in all_results:
            print(f"\n❌ Workflow Failed: {result['workflow_name']}")
            print(f"Run ID: {result['run_id']}\n")
            for err in result['errors']:
                print(f"🔴 Error #{err['error_number']}")
                print(f"Job/Step: {err['job_step']}")
                print(f"Log Line: {err['line_no']}")
                print(f"Error: {err['error_line']}")
                print(f"Suggestion:\n{err['suggestion']}\n")

        print("\n================ SUMMARY ================")
        print(f"Total Workflows: {len(workflows)}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print("========================================\n")

# ================= RUN =================
if __name__ == "__main__":
    GitHubMonitorAgent().monitor()
