#!/usr/bin/env python3
import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def pyname():
    return os.path.basename(sys.executable)


def child_env(base=None):
    e = dict(os.environ if base is None else base)
    e["PATH"] = os.pathsep.join(filter(None, [os.path.dirname(sys.executable), e.get("PATH")]))
    return e


def reexec(env):
    os.execve(sys.executable, [pyname(), str(Path(__file__).resolve()), *sys.argv[1:]], child_env(env))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs/server_r1.yaml"))
    ap.add_argument("--no-install", action="store_true", help="패키지를 이미 설치했으면 pip 단계를 건너뜀")
    ap.add_argument("--torch-index", help="드라이버가 오래돼 CUDA를 못 쓸 때 torch 휠 주소 (예: https://download.pytorch.org/whl/cu126)")
    ap.add_argument("--skip-preflight", action="store_true", help="이미 점검을 통과했고 재시작만 할 때")
    ap.add_argument("--foreground", action="store_true", help="백그라운드로 보내지 않고 이 터미널에서 실행")
    a = ap.parse_args()

    if os.environ.get("PYTHONHASHSEED") != "0":
        reexec({**os.environ, "PYTHONHASHSEED": "0",
                "TOKENIZERS_PARALLELISM": "true", "PYTHONUNBUFFERED": "1"})

    print(f"파이썬: {pyname()} ({sys.version.split()[0]})", flush=True)
    if not a.no_install:
        pip = [pyname(), "-m", "pip", "install", "-q"]
        if a.torch_index:
            subprocess.run(pip + ["torch==2.14.1", "--index-url", a.torch_index], check=True,
                           executable=sys.executable, env=child_env())
        r = subprocess.run(pip + ["-r", str(ROOT / "requirements.txt")], executable=sys.executable, env=child_env())
        if r.returncode:
            sys.exit("pip 설치 실패. 시스템 파이썬이 설치를 막으면(externally-managed) "
                     "`pip install --user -r requirements.txt`로 직접 설치한 뒤 --no-install로 실행하세요.")

    config = str(Path(a.config).expanduser().resolve())
    import yaml
    cfg = yaml.safe_load(open(config, encoding="utf-8"))
    if "sources" in cfg and not os.environ.get("HF_TOKEN"):
        sys.exit("HF_TOKEN 환경 변수가 필요합니다 (bigcode/the-stack-dedup 약관에 동의한 계정의 토큰)")
    env = child_env({**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(ROOT), os.environ.get("PYTHONPATH")]))})
    if not a.skip_preflight:
        r = subprocess.run([pyname(), str(ROOT / "scripts/preflight.py"), "--config", config],
                           cwd=ROOT, env=env, executable=sys.executable)
        if r.returncode:
            sys.exit(r.returncode)

    entry = cfg.get("entry", "arbor.train.r1")
    cmd = [pyname(), "-m", entry, "--config", config]
    if a.foreground:
        os.chdir(ROOT)
        os.execve(sys.executable, cmd, env)
    root = Path(os.path.expanduser(cfg["data_root"]))
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    tag = entry.rsplit(".", 1)[-1]
    out = open(logs / f"{tag}.out", "a")
    p = subprocess.Popen(cmd, cwd=ROOT, env=env, executable=sys.executable,
                         stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    print(f"\n시작했습니다. PID {p.pid}")
    print(f"진행: tail -f {logs / (tag + '.log')}")
    print(f"결과: {root / 'runs' / tag / 'summary.md'}")
    args = sys.argv[1:] + ([] if a.skip_preflight else ["--skip-preflight"])
    print(f"재시작(점검 생략): HF_TOKEN=... python3 {shlex.join([sys.argv[0], *args])}")


if __name__ == "__main__":
    main()
