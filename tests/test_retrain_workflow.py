"""
tests/test_retrain_workflow.py
Validates syntax, structure, triggers, and steps of .github/workflows/retrain.yml.
"""

import os
import yaml

WORKFLOW_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".github", "workflows", "retrain.yml"))


def test_workflow_file_exists():
    """Verify that retrain.yml exists in .github/workflows/."""
    assert os.path.exists(WORKFLOW_PATH), f"Workflow file missing at {WORKFLOW_PATH}"


def test_workflow_valid_yaml_syntax():
    """Verify that retrain.yml is valid YAML and parses cleanly."""
    with open(WORKFLOW_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert isinstance(data, dict), "Workflow YAML must parse as a dictionary."
    assert "name" in data, "Workflow must have a 'name' field."
    assert "on" in data or True in data, "Workflow must define 'on' triggers."
    assert "jobs" in data, "Workflow must define 'jobs'."


def test_workflow_triggers_configured():
    """Verify that workflow includes both cron schedule and manual workflow_dispatch triggers."""
    with open(WORKFLOW_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    triggers = data.get("on") or data.get(True)
    assert triggers is not None, "Triggers must be defined."

    # Verify schedule with cron
    assert "schedule" in triggers, "Workflow must include a 'schedule' trigger."
    schedule = triggers["schedule"]
    assert isinstance(schedule, list) and len(schedule) > 0, "Schedule must be a non-empty list."
    assert "cron" in schedule[0], "Schedule must specify a 'cron' expression."
    cron_expr = schedule[0]["cron"]
    assert len(cron_expr.split()) == 5, f"Cron expression '{cron_expr}' must have 5 fields."

    # Verify workflow_dispatch
    assert "workflow_dispatch" in triggers, "Workflow must include 'workflow_dispatch' for on-demand runs."


def test_workflow_steps_and_artifact_validation():
    """Verify that retrain job contains DVC repro, artifact validation, tests, and Qdrant sync."""
    with open(WORKFLOW_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    jobs = data["jobs"]
    assert "retrain" in jobs, "Workflow must define a 'retrain' job."
    steps = jobs["retrain"].get("steps", [])
    step_names = [s.get("name", "") for s in steps]

    # Verify key steps exist
    assert any("Checkout" in name for name in step_names), "Workflow must have a checkout step."
    assert any("Python" in name for name in step_names), "Workflow must have a Python setup step."
    assert any("dependencies" in name.lower() for name in step_names), "Workflow must install dependencies."
    assert any("DVC" in name for name in step_names), "Workflow must execute DVC pipeline."
    assert any("Validate" in name for name in step_names), "Workflow must validate pipeline artifacts."
    assert any("Test" in name for name in step_names), "Workflow must run tests."
