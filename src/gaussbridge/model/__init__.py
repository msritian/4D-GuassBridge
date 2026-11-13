"""Workstream C: the goal-conditioned motion model (CLAUDE.md §8).

dataset.py   Episodes -> training samples (motion grid, known mask, conditioning copy, shape points)
encoders.py  object shape points -> context tokens                                  (next step)
dit.py       MotionDiT, the transformer that predicts the flow velocity             (next step)
flow.py      flow-matching loss and the inpainting sampler                          (next step)
"""
