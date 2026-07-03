import os
import uuid
import subprocess
from datetime import datetime


class DeploymentContext:
    def __init__(self):
        now = datetime.now()
        self.deployment_id = f"v6_{now.strftime('%Y%m%d_%H%M')}"
        self.session_id = uuid.uuid4().hex[:8]
        self.start_time = now.isoformat()
        self.git_commit = self._get_git_commit()

    def _get_git_commit(self) -> str:
        try:
            repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, cwd=repo, timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return "unknown"

    def to_dict(self) -> dict:
        return {
            "deployment_id": self.deployment_id,
            "session_id": self.session_id,
            "start_time": self.start_time,
            "git_commit": self.git_commit,
        }

    def headers(self) -> str:
        return (f"Deployment: {self.deployment_id} | Session: {self.session_id} | "
                f"Start: {self.start_time} | Commit: {self.git_commit}")
