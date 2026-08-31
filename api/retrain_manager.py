"""
api/retrain_manager.py
Orchestrates background execution of DVC retrain pipeline via subprocess.
Provides thread-safe job management, single-flight concurrency lock, and log streaming.
"""

import os
import sys
import uuid
import threading
import subprocess
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

from config import BASE_DIR, RETRAIN_LOG_PATH


class RetrainManager:
    """
    Manages background execution of 'dvc repro' via subprocess.
    Ensures single-flight execution (no concurrent runs), tracks status,
    and logs stdout/stderr to a dedicated log file.
    """

    def __init__(self, log_path: str = RETRAIN_LOG_PATH, base_dir: str = BASE_DIR):
        self.log_path = log_path
        self.base_dir = base_dir
        self.lock = threading.Lock()
        self.current_process: Optional[subprocess.Popen] = None
        self.job_id: Optional[str] = None
        self.started_at: Optional[str] = None
        self.finished_at: Optional[str] = None
        self.return_code: Optional[int] = None
        self.status: str = "idle"  # idle, running, completed, failed
        self._thread: Optional[threading.Thread] = None

    def is_running(self) -> bool:
        """Check whether a retrain job is currently actively executing."""
        with self.lock:
            if self.current_process is not None:
                poll = self.current_process.poll()
                if poll is None:
                    return True
                else:
                    self._update_finished_state(poll)
            return False

    def _update_finished_state(self, return_code: int):
        """Internal helper to finalize job state upon completion."""
        self.return_code = return_code
        self.status = "completed" if return_code == 0 else "failed"
        self.finished_at = datetime.now(timezone.utc).isoformat()
        self.current_process = None

    def trigger(
        self,
        force: bool = False,
        targets: Optional[List[str]] = None,
        custom_cmd: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Trigger a background retrain job.

        Args:
            force: If True, passes --force to dvc repro.
            targets: Optional list of specific DVC stage targets to run.
            custom_cmd: Optional custom command override (useful for testing).

        Returns:
            Dict containing success status, job_id, and message.
        """
        with self.lock:
            # Check if an existing process is still running
            if self.current_process is not None and self.current_process.poll() is None:
                return {
                    "success": False,
                    "status": "in_progress",
                    "message": "A retrain job is already in progress.",
                    "job_id": self.job_id,
                    "started_at": self.started_at,
                }

            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            self.job_id = str(uuid.uuid4())
            self.started_at = datetime.now(timezone.utc).isoformat()
            self.finished_at = None
            self.return_code = None
            self.status = "running"

            if custom_cmd is not None:
                cmd = custom_cmd
            else:
                cmd = [sys.executable, "-m", "dvc", "repro"]
                if force:
                    cmd.append("--force")
                if targets:
                    cmd.extend(targets)

            log_file = open(self.log_path, "w", encoding="utf-8")
            log_file.write(f"=== Retrain Job {self.job_id} Started at {self.started_at} ===\n")
            log_file.write(f"Command: {' '.join(cmd)}\n\n")
            log_file.flush()

            self.current_process = subprocess.Popen(
                cmd,
                cwd=self.base_dir,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )

            # Spawn monitor thread to track completion and safely close log file
            def _monitor(proc: subprocess.Popen, lf):
                proc.wait()
                rc = proc.returncode
                with self.lock:
                    self._update_finished_state(rc)
                try:
                    lf.write(
                        f"\n=== Retrain Job {self.job_id} Finished at {self.finished_at} with exit code {rc} ===\n"
                    )
                    lf.close()
                except Exception:
                    pass

            self._thread = threading.Thread(target=_monitor, args=(self.current_process, log_file), daemon=True)
            self._thread.start()

            return {
                "success": True,
                "status": "triggered",
                "message": "DVC retrain pipeline started successfully.",
                "job_id": self.job_id,
                "started_at": self.started_at,
            }

    def get_status(self, max_log_lines: int = 50) -> Dict[str, Any]:
        """
        Get current job execution status and tail of retrain logs.
        """
        with self.lock:
            if self.current_process is not None:
                poll = self.current_process.poll()
                if poll is not None:
                    self._update_finished_state(poll)

            log_tail = ""
            if os.path.exists(self.log_path):
                try:
                    with open(self.log_path, "r", encoding="utf-8", errors="replace") as f:
                        lines = f.readlines()
                        log_tail = "".join(lines[-max_log_lines:])
                except Exception as e:
                    log_tail = f"<error reading log: {e}>"

            return {
                "status": self.status,
                "job_id": self.job_id,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "return_code": self.return_code,
                "log_tail": log_tail,
            }
