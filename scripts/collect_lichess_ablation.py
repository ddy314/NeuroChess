"""Fetch the authorized one-off experiment report using existing CLI credentials."""
import argparse,hashlib,json,shutil,subprocess,time
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--session',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    remote='/content/lichess-experiment-4000';deadline=time.time()+21600
    colab=shutil.which('colab') or str(Path.home()/'.local/bin/colab')
    def write(name,d):
        path=out/name;temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(d,indent=2));temp.replace(path)
    def download(name):
        rel=Path(name)
        if rel.is_absolute() or '..' in rel.parts:raise ValueError('Invalid evidence path')
        dest=out/rel;dest.parent.mkdir(parents=True,exist_ok=True);temp=dest.with_name(dest.name+'.download.tmp')
        r=subprocess.run([colab,'download','-s',a.session,remote+'/'+name,str(temp)],capture_output=True,text=True,timeout=40)
        if r.returncode:return False
        temp.replace(dest);return True
    missing=0
    while time.time()<deadline:
        try:
            name='status.json';have=download(name)
            if not have:name='watch-status.json';have=download(name)
            if have:
                missing=0;status=json.loads((out/name).read_text())
                write('collector-status.json',dict(phase='monitoring',remote_phase=status['phase'],checked_at=time.time()))
                if status['phase']=='failed':
                    download('failure.json');write('collector-failure.json',dict(remote_status=status,checked_at=time.time()));return 2
            else:
                missing+=1;write('collector-status.json',dict(phase='temporarily-unreachable',consecutive_failures=missing,checked_at=time.time()))
            if have and status['phase'] in ['awaiting-report-copy','complete'] and download('completion.json'):
                proof=json.loads((out/'completion.json').read_text())
                for name,want in proof['files_sha256'].items():
                    if Path(name).suffix not in ['.json','.jsonl','.md','.npz','.py']:continue
                    if not download(name):raise OSError('Evidence download failed: '+name)
                    if hashlib.sha256((out/name).read_bytes()).hexdigest()!=want:raise OSError('Evidence hash differs: '+name)
                if hashlib.sha256((out/'results.json').read_bytes()).hexdigest()!=proof['report_sha256']:raise OSError('Report hash mismatch')
                ack=dict(report_sha256=proof['report_sha256'],verified_at=time.time());write('local-report-ack.json',ack)
                r=subprocess.run([colab,'upload','-s',a.session,str(out/'local-report-ack.json'),remote+'/local-report-ack.json'],capture_output=True,text=True,timeout=40)
                write('collector-complete.json',dict(completed=True,verified_at=time.time(),report_sha256=proof['report_sha256'],ack_uploaded=r.returncode==0,all_three_completed_500=proof['all_three_completed_500']))
                return 0
        except Exception as e:
            write('collector-status.json',dict(phase='retrying',error=str(e),checked_at=time.time()))
        time.sleep(30)
    write('collector-failure.json',dict(reason='Report collection exceeded six-hour window',checked_at=time.time()));return 2


if __name__=='__main__':raise SystemExit(main())
