"""Enforce the user's10-hour GPU allocation cap and release after persistence."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from neurochess.autostop import request_unassign,find_runtime_address
from neurochess.pawn_data import write_json


def main(plan_path):
    plan=json.loads(Path(plan_path).read_text());work=Path(plan['work']);save=Path(plan['save'])
    work.mkdir(parents=True,exist_ok=True);save.mkdir(parents=True,exist_ok=True)
    success=False
    try:
        timeout=max(1,plan['allocated_at']+plan['gpu_budget_seconds']-time.time()-60)
        subprocess.run([sys.executable,'-u','-m','scripts.train_pawn_cloud',plan_path],check=True,timeout=timeout)
        success=True
    finally:
        try:
            if not os.path.ismount('/content/drive'):raise OSError('Drive unavailable; do not claim results persisted')
            if (work/'console.log').exists():shutil.copyfile(work/'console.log',save/'console.log')
            write_json(save/'shutdown-request.json',dict(success=success,requested_at=time.time(),gpu_budget_seconds=plan['gpu_budget_seconds']))
        finally:
            # The user's allocation-time cap still applies if persistence fails.
            request_unassign(find_runtime_address())


if __name__=='__main__':main(sys.argv[1])
