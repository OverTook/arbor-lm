import argparse
import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

fails, warns = [], []


def check(ok, msg, fatal=True):
    print(("OK   " if ok else ("FAIL " if fatal else "WARN ")) + msg, flush=True)
    if not ok:
        (fails if fatal else warns).append(msg)


def gpu_kernel_checks(torch):
    import torch.nn.functional as F
    try:
        from fla.ops.gla import chunk_gla
        sys.path.insert(0, str(ROOT))
        from arbor.layers.mixer import gla_reference
        q, k, v = (torch.randn(1, 64, 2, 32, device="cuda") for _ in range(3))
        g = F.logsigmoid(torch.randn(1, 64, 2, 32, device="cuda")) / 16
        o = chunk_gla(q, k, v, g, scale=32 ** -0.5)[0]
        ref = gla_reference(q, k, v, g, 32 ** -0.5)
        err = ((o - ref).norm() / ref.norm()).item()
        check(o.is_cuda and err < 5e-3, f"fla Triton 커널 GPU 실행 (상대 오차 {err:.1e})")
    except Exception as e:
        check(False, f"fla Triton 커널 GPU 실행 실패: {type(e).__name__}: {str(e)[:200]}")
    try:
        f = torch.compile(lambda x: torch.nn.functional.silu(x) * 2)
        x = torch.randn(1024, device="cuda")
        check(torch.allclose(f(x), torch.nn.functional.silu(x) * 2), "torch.compile GPU 실행")
    except Exception as e:
        check(False, f"torch.compile GPU 실행 실패: {type(e).__name__}: {str(e)[:200]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--min-disk-gb", type=float, default=120)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    root = Path(os.path.expanduser(cfg["data_root"]))
    root.mkdir(parents=True, exist_ok=True)

    check(sys.version_info >= (3, 10), f"Python {sys.version.split()[0]} (3.10 이상)")
    for mod in ("torch", "fla", "tokenizers", "pyarrow", "pypdf", "yaml", "huggingface_hub", "numpy"):
        try:
            importlib.import_module(mod)
            check(True, f"import {mod}")
        except Exception as e:
            check(False, f"import {mod}: {e}")
    import torch
    check(torch.cuda.is_available(), "CUDA 사용 가능")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        check(p.total_memory > 40e9, f"GPU {p.name} {p.total_memory / 1e9:.0f}GB", fatal=False)
    import sysconfig
    inc = Path(sysconfig.get_paths()["include"]) / "Python.h"
    check(inc.exists(), f"Python 개발 헤더 {inc} (Triton·torch.compile에 필요. 없으면: sudo apt install "
                        f"python{sys.version_info.major}.{sys.version_info.minor}-dev)")
    if torch.cuda.is_available():
        gpu_kernel_checks(torch)


    free = shutil.disk_usage(root).free / 1e9
    check(free >= a.min_disk_gb, f"디스크 여유 {free:.0f}GB (필요 {a.min_disk_gb:.0f}GB 이상, {root})")
    mem = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    check(mem >= 64, f"RAM {mem:.0f}GB (64GB 이상)")

    from huggingface_hub import HfApi
    api = HfApi()
    if "data_config" in cfg:
        done = root / "state/tokenize.done"
        check(done.exists(), f"R1 토큰 데이터 {done}")
    for name, s in cfg.get("sources", {}).items():
        try:
            api.auth_check(s["repo"], repo_type="dataset")
            n = sum(1 for f in api.list_repo_files(s["repo"], repo_type="dataset")
                    if f.startswith(s["prefix"]) and f.endswith(".parquet"))
            check(n > 0, f"데이터 {name}: {s['repo']}/{s['prefix']} 조각 {n}개")
        except Exception as e:
            check(False, f"데이터 {name}: {s['repo']} 접근 불가 ({type(e).__name__}). 게이트 데이터면 HF 약관 동의 + HF_TOKEN")
    for name, s in cfg.get("benchmarks", {}).items():
        try:
            api.auth_check(s["repo"], repo_type="dataset")
            check(True, f"벤치마크 {name}", fatal=False)
        except Exception as e:
            check(False, f"벤치마크 {name}: {s['repo']} 접근 불가 ({type(e).__name__}) → 이 벤치마크 누설 제거는 빠짐", fatal=False)
    if "shareable_dir" in cfg:
        sh = ROOT / os.path.expanduser(cfg["shareable_dir"])
        check(sh.exists() and any(sh.glob("*.pdf")), f"공유 가능 문서 {sh} (KCS PDF, 누설 제거에 사용)", fatal=False)

    r = subprocess.run([os.path.basename(sys.executable), "-m", "pytest", "-q", str(ROOT / "tests")],
                       capture_output=True, text=True, executable=sys.executable, cwd=ROOT,
                       env={**os.environ,
                            "PYTHONPATH": os.pathsep.join(filter(None, [str(ROOT), os.environ.get("PYTHONPATH")])),
                            "PATH": os.pathsep.join(filter(None, [os.path.dirname(sys.executable), os.environ.get("PATH")]))})
    check(r.returncode == 0, "테스트: " + (r.stdout.strip().splitlines() or ["(출력 없음)"])[-1])
    if r.returncode:
        print(r.stdout[-3000:])

    print(f"\n경고 {len(warns)}개, 실패 {len(fails)}개")
    if fails:
        print("실패 항목을 고친 뒤 다시 실행하세요. 본 실행은 시작하지 않았습니다.")
        sys.exit(1)


if __name__ == "__main__":
    main()
