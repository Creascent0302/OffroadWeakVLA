"""Verify the runtime can actually execute Blackwell (sm_120) kernels.

`torch.cuda.is_available()` is not enough: a torch built for an older toolkit
imports fine, reports the GPU, and only fails when a kernel is launched.  So we
launch one.
"""

from __future__ import annotations

import argparse
import sys

import torch

EXPECTED_CAPABILITY = (12, 0)  # RTX 5090 / Blackwell


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--allow-cpu",
        action="store_true",
        help="do not fail when no GPU is visible (build time, CI, laptop dev)",
    )
    args = ap.parse_args()

    print(f"torch                : {torch.__version__}")
    print(f"torch CUDA build     : {torch.version.cuda}")
    print(f"cuda available       : {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("no CUDA device visible")
        return 0 if args.allow_cpu else 1

    print(f"device count         : {torch.cuda.device_count()}")
    ok = True
    for i in range(torch.cuda.device_count()):
        cap = torch.cuda.get_device_capability(i)
        name = torch.cuda.get_device_name(i)
        flag = "OK" if cap >= EXPECTED_CAPABILITY else "WARN"
        print(f"  [{i}] {name:<32} sm_{cap[0]}{cap[1]}  {flag}")
        if cap != EXPECTED_CAPABILITY:
            ok = False

    # Launch a real kernel. This is what catches a torch that was not built for
    # sm_120 ("no kernel image is available for execution on the device").
    try:
        a = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
        b = a @ a
        torch.cuda.synchronize()
        print(f"bf16 matmul          : OK (checksum {b.float().sum().item():.3e})")
    except RuntimeError as exc:
        print(f"bf16 matmul          : FAILED -> {exc}")
        return 1

    sdpa = torch.nn.functional.scaled_dot_product_attention
    q = torch.randn(1, 4, 64, 32, device="cuda", dtype=torch.bfloat16)
    sdpa(q, q, q)
    torch.cuda.synchronize()
    print("scaled_dot_product_attention: OK")

    if not ok:
        print(f"\nNOTE: expected sm_{EXPECTED_CAPABILITY[0]}{EXPECTED_CAPABILITY[1]} (RTX 5090).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
