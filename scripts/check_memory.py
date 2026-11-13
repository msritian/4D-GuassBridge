"""C3 acceptance: does a training step of the full MotionDiT at batch 256 fit a 40 GB GPU in bf16? (TASKS.md C3)

Builds a random batch with the real shapes (no data needed), runs a few full training steps (forward, loss,
backward, AdamW step) under bf16 autocast, and reports peak GPU memory and time per step. Also runs fp32 for
comparison unless --skip-fp32. Writes the numbers to --out (JSON) and prints them.

usage (CHTC GPU job, chtc/check_memory.sub; or a CPU smoke test with --size tiny --batch 4):
    python scripts/check_memory.py --out memory.json
"""
import argparse
import json
import time

import torch

from gaussbridge.model.dit import MotionDiT
from gaussbridge.model.flow import flow_loss
from gaussbridge.trajectory import K_MAX, N_STEPS, NUM_FEATURES, known_mask

SIZES = {"full": {}, "tiny": {"d_model": 64, "depth": 2, "heads": 4}}
BUDGET_GB = 40.0


def random_batch(batch, device):
    """Random tensors with the shapes and dtypes of a MotionDataset batch (StackCube: robot + 2 objects)."""
    E = 1 + K_MAX
    ent_mask = torch.tensor([True, True, True, False], device=device).expand(batch, E)
    known = known_mask(ent_mask, "full")
    x0 = torch.randn(batch, N_STEPS, E, NUM_FEATURES, device=device)
    return {"x0": x0, "x_known": x0 * known[..., None], "known": known, "ent_mask": ent_mask,
            "log_dur": torch.randn(batch, device=device), "shapes": torch.randn(batch, K_MAX, 512, 6, device=device)}


def measure(model, batch_size, device, bf16, steps):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    batch = random_batch(batch_size, device)
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(steps):
        t0 = time.time()
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=bf16):
            loss, _ = flow_loss(model, batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if device == "cuda":
            torch.cuda.synchronize()
        times.append(time.time() - t0)
    peak = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else float("nan")
    del opt
    return {"bf16": bf16, "batch": batch_size, "peak_gb": round(peak, 2),
            "sec_per_step": round(sum(times[1:]) / max(1, len(times) - 1), 4), "loss": round(loss.item(), 4)}


def safe_measure(model, batch_size, device, bf16, steps):
    """measure(), but a run that does not fit this GPU is recorded as {"oom": True} instead of crashing the job
    (the job may land on a smaller card than the 40 GB budget; peak memory itself does not depend on the card)."""
    if device == "cuda":
        torch.cuda.empty_cache()
    try:
        return measure(model, batch_size, device, bf16, steps)
    except torch.cuda.OutOfMemoryError:
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        return {"bf16": bf16, "batch": batch_size, "oom": True,
                "note": "does not fit this GPU (see gpu_total_gb); rerun on a 40 GB card for the budget check"}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", default="memory.json")
    p.add_argument("--size", choices=sorted(SIZES), default="full")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--skip-fp32", action="store_true")
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    info = {"device": torch.cuda.get_device_name() if device == "cuda" else "cpu", "torch": torch.__version__}
    if device == "cuda":
        info["gpu_total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
    model = MotionDiT(**SIZES[args.size]).to(device)
    info["parameters_m"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 2)
    runs = [safe_measure(model, args.batch, device, True, args.steps)]
    if not args.skip_fp32:
        runs.append(safe_measure(model, args.batch, device, False, args.steps))
    bf16 = runs[0]
    report = {**info, "budget_gb": BUDGET_GB, "runs": runs,
              "fits_budget_bf16": None if device != "cuda" or bf16.get("oom") else bf16["peak_gb"] < BUDGET_GB}
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1), flush=True)


if __name__ == "__main__":
    main()
