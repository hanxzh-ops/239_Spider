#!/usr/bin/env python3
"""
eval.py  —  watch / measure a trained policy
============================================

Loads rl/ppo_hexapod.zip and rolls it out. With a display it opens the MuJoCo
viewer; headless it just prints distance walked and mean forward speed.

Usage:
    python rl/eval.py                    # viewer if a display is available
    python rl/eval.py --episodes 5 --no-render
"""
import os
import argparse

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from hexapod_env import HexapodEnv

_HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(_HERE, "ppo_hexapod.zip"))
    ap.add_argument("--vecnorm", default=os.path.join(_HERE, "vecnormalize.pkl"))
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()

    render_mode = None if args.no_render else "human"

    # rebuild the same normalization wrapper used in training (stats frozen)
    def _mk():
        return HexapodEnv(render_mode=render_mode)
    venv = DummyVecEnv([_mk])
    if os.path.exists(args.vecnorm):
        venv = VecNormalize.load(args.vecnorm, venv)
        venv.training = False
        venv.norm_reward = False

    model = PPO.load(args.model, device="cpu")

    for ep in range(args.episodes):
        obs = venv.reset()
        done = np.array([False])
        ret, steps, x0 = 0.0, 0, None
        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = venv.step(action)
            ret += float(reward[0])
            steps += 1
            inner = info[0]
            if x0 is None:
                x0 = inner.get("x", 0.0)
            last = inner
        dist = last.get("x", 0.0) - (x0 or 0.0)
        print(f"episode {ep}: steps={steps}  return={ret:.2f}  "
              f"distance={dist:.3f} m  final_z={last.get('z', 0):.3f}")

    venv.close()


if __name__ == "__main__":
    main()
