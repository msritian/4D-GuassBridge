"""C4: train the motion model (recipe and resume behaviour: src/gaussbridge/model/train.py).

usage:
    python scripts/train_motion.py configs/train/stackcube_v1.yaml --run runs/stackcube_v1 \
        [--set optim.steps=1000 data.val=null ...] [--max-minutes 120]

Writes into --run: config.yaml (resolved), log.jsonl (training/validation lines), ckpt/ckpt_XXXXXXX.pt (last
log.keep_ckpts kept), final_ema.pt (EMA weights + dur_stats + configs, what sampling uses).
Running the same command again resumes from the newest checkpoint.

Exit codes: 0 = finished, 85 = paused by --max-minutes with a checkpoint (HTCondor self-checkpointing: the job is
restarted and resumes), anything else = error.
"""
import argparse
import sys

from gaussbridge.model.train import EXIT_PAUSED, Trainer, load_config


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("config")
    p.add_argument("--run", required=True, help="run directory (created; resumed if it has checkpoints)")
    # action="extend": repeated --set flags accumulate (chtc/train.sh passes data paths in one, the run's overrides in
    # another; with the default action the second would silently replace the first)
    p.add_argument("--set", nargs="*", action="extend", default=[], metavar="KEY=VALUE",
                   help="config overrides, e.g. optim.lr=1e-4")
    p.add_argument("--max-minutes", type=float, default=None, help="checkpoint and exit 85 after this long")
    p.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")
    args = p.parse_args()
    cfg = load_config(args.config, args.set)
    status = Trainer(cfg, args.run, args.device).fit(max_minutes=args.max_minutes, log=lambda s: print(s, flush=True))
    sys.exit(EXIT_PAUSED if status == "paused" else 0)


if __name__ == "__main__":
    main()
