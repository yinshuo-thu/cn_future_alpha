"""
Shared real-time journaling utility.
Both Plan A and Plan B agents append their attempts/thinking/results/analysis
to markdown docs under /root/feature_model/reports/journal/ for live review.
"""
import os
import datetime

JOURNAL_DIR = "/root/feature_model/reports/journal"


def _now():
    # Server local time
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(plan, section, content, level="INFO"):
    """
    Append a timestamped entry to the plan's journal markdown.
    plan: 'plan_a' | 'plan_b' | 'shared'
    section: short heading e.g. 'Phase 6 Baseline'
    content: markdown body (attempt / thinking / result / analysis)
    """
    os.makedirs(JOURNAL_DIR, exist_ok=True)
    path = os.path.join(JOURNAL_DIR, f"{plan}.md")
    entry = f"\n### [{_now()}] {section}  `{level}`\n\n{content}\n"
    with open(path, "a") as f:
        f.write(entry)
    # Also echo to stdout so tmux log captures it
    print(f"[JOURNAL:{plan}] {section} ({level})")


def log_attempt(plan, what, why):
    log(plan, f"尝试: {what}", f"**思路 (Why):** {why}", level="ATTEMPT")


def log_result(plan, what, metrics: dict, analysis=""):
    body = "**结果 (Result):**\n\n"
    for k, v in metrics.items():
        body += f"- `{k}` = {v}\n"
    if analysis:
        body += f"\n**分析 (Analysis):** {analysis}\n"
    log(plan, f"结果: {what}", body, level="RESULT")


def log_shared_conclusion(title, content):
    """Cross-plan shared conclusions (minimal facts both agents can reuse)."""
    log("shared", title, content, level="CONCLUSION")


def init_journal(plan, title):
    os.makedirs(JOURNAL_DIR, exist_ok=True)
    path = os.path.join(JOURNAL_DIR, f"{plan}.md")
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(f"# {title}\n\n> 实时记录：尝试 / 思考 / 结果 / 分析\n")
    log(plan, "Journal 初始化", f"开始记录 {title}", level="INIT")
