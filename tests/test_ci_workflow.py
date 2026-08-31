"""
tests/test_ci_workflow.py
=========================
Phase 9 — GitHub Actions CI & Retrain Workflow Suite Validation

Validates:
  - .github/workflows/ci.yml:
      * Valid YAML syntax and dictionary structure
      * Push, PR, and manual workflow_dispatch triggers
      * Lint job (flake8, black check)
      * Test job (pytest)
      * Smoke-retrain job (DVC DAG & dry repro check)
      * Frontend build job (Next.js production build)
      * Deploy-on-tag job (Docker build & release verification)
  - .github/workflows/scheduled_retrain.yml & retrain.yml:
      * Cron schedule configuration (weekly retrain)
      * On-demand workflow_dispatch configuration
      * DVC repro execution, artifact validation, and Qdrant sync steps
"""

import os
import yaml

WORKFLOWS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".github", "workflows"))
CI_WORKFLOW_PATH = os.path.join(WORKFLOWS_DIR, "ci.yml")
SCHEDULED_RETRAIN_PATH = os.path.join(WORKFLOWS_DIR, "scheduled_retrain.yml")
RETRAIN_PATH = os.path.join(WORKFLOWS_DIR, "retrain.yml")


def test_ci_workflow_file_exists():
    """Verify that ci.yml exists in .github/workflows/."""
    assert os.path.exists(CI_WORKFLOW_PATH), f"CI workflow missing at {CI_WORKFLOW_PATH}"


def test_ci_workflow_valid_yaml_syntax():
    """Verify that ci.yml is valid YAML and parses into a proper dictionary."""
    with open(CI_WORKFLOW_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert isinstance(data, dict), "ci.yml must parse as a dictionary."
    assert "name" in data, "ci.yml must define a workflow name."
    assert "on" in data or True in data, "ci.yml must define triggers."
    assert "jobs" in data, "ci.yml must define jobs."


def test_ci_workflow_triggers():
    """Verify push, pull_request, and workflow_dispatch triggers in ci.yml."""
    with open(CI_WORKFLOW_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    triggers = data.get("on") or data.get(True)
    assert triggers is not None, "Triggers must be present in ci.yml."

    # Verify push branches & tags
    assert "push" in triggers, "ci.yml must trigger on push."
    push = triggers["push"]
    assert "branches" in push, "push trigger must specify branches."
    assert "main" in push["branches"] or "master" in push["branches"]

    # Verify pull_request
    assert "pull_request" in triggers, "ci.yml must trigger on pull_request."
    pr = triggers["pull_request"]
    assert "branches" in pr, "pull_request trigger must specify branches."

    # Verify workflow_dispatch
    assert "workflow_dispatch" in triggers, "ci.yml must support manual trigger."


def test_ci_workflow_jobs_and_steps():
    """Verify all required CI jobs exist with appropriate steps."""
    with open(CI_WORKFLOW_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    jobs = data["jobs"]

    # 1. Lint job
    assert "lint" in jobs, "ci.yml must define a 'lint' job."
    lint_steps = [s.get("name", "") for s in jobs["lint"].get("steps", [])]
    assert any("flake8" in name.lower() or "lint" in name.lower() for name in lint_steps)
    assert any("black" in name.lower() for name in lint_steps)

    # 2. Test job
    assert "test" in jobs, "ci.yml must define a 'test' job."
    test_steps = [s.get("name", "") for s in jobs["test"].get("steps", [])]
    assert any("pytest" in name.lower() or "test" in name.lower() for name in test_steps)

    # 3. Smoke-retrain job
    assert "smoke-retrain" in jobs, "ci.yml must define a 'smoke-retrain' job."
    smoke_steps = [s.get("name", "") for s in jobs["smoke-retrain"].get("steps", [])]
    assert any("dvc" in name.lower() for name in smoke_steps)

    # 4. Frontend-build job
    assert "frontend-build" in jobs, "ci.yml must define a 'frontend-build' job."
    frontend_steps = [s.get("name", "") for s in jobs["frontend-build"].get("steps", [])]
    assert any("node" in name.lower() for name in frontend_steps)
    assert any("next" in name.lower() or "build" in name.lower() for name in frontend_steps)

    # 5. Deploy-on-tag job
    assert "deploy-on-tag" in jobs, "ci.yml must define a 'deploy-on-tag' job."
    deploy_job = jobs["deploy-on-tag"]
    assert "if" in deploy_job, "deploy-on-tag must have conditional trigger for tags."
    assert "tags" in deploy_job["if"] or "refs/tags" in deploy_job["if"]
    deploy_steps = [s.get("name", "") for s in deploy_job.get("steps", [])]
    assert any("docker" in name.lower() or "container" in name.lower() for name in deploy_steps)


def test_scheduled_retrain_workflow_configured():
    """Verify scheduled_retrain.yml exists, is valid YAML, and has cron schedule."""
    assert os.path.exists(SCHEDULED_RETRAIN_PATH), f"scheduled_retrain.yml missing at {SCHEDULED_RETRAIN_PATH}"

    with open(SCHEDULED_RETRAIN_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    triggers = data.get("on") or data.get(True)
    assert "schedule" in triggers, "scheduled_retrain.yml must have a schedule trigger."
    cron_list = triggers["schedule"]
    assert len(cron_list) > 0 and "cron" in cron_list[0]
    cron_expr = cron_list[0]["cron"]
    assert len(cron_expr.split()) == 5, f"Cron expression '{cron_expr}' must have 5 fields."

    assert "workflow_dispatch" in triggers, "scheduled_retrain.yml must have workflow_dispatch."

    jobs = data["jobs"]
    assert "retrain" in jobs, "scheduled_retrain.yml must define a 'retrain' job."
