"""Launch the deep R&F search detached, with persistent workspace logs."""
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parents[1]
out=root/'outputs/experiments/resonate_and_fire_deep_gain_search';out.mkdir(parents=True,exist_ok=True)
stdout=(out/'deep_search.stdout.log').open('a',buffering=1)
stderr=(out/'deep_search.stderr.log').open('a',buffering=1)
flags=getattr(subprocess,'CREATE_NO_WINDOW',0)|getattr(subprocess,'DETACHED_PROCESS',0)|getattr(subprocess,'CREATE_NEW_PROCESS_GROUP',0)
process=subprocess.Popen([sys.executable,str(root/'scripts/run_resonate_and_fire_deep_gain_search.py')],cwd=root,stdout=stdout,stderr=stderr,stdin=subprocess.DEVNULL,creationflags=flags,close_fds=True)
(out/'deep_search.pid').write_text(str(process.pid),encoding='ascii')
print(process.pid)
