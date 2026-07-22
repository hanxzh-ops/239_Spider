#!/usr/bin/env python3
"""
train.py  —  train a walking policy with PPO (Stable-Baselines3)
================================================================

Learns to walk the hexapod forward from scratch by optimizing the reward in
hexapod_env.py. Runs headless (no viewer) across several parallel envs for speed.

Usage:
    python rl/train.py                       # default 1M steps, 4 envs
    python rl/train.py --timesteps 3000000 --n-envs 8
    python rl/train.py --timesteps 5000      # tiny smoke test

Outputs:
    rl/ppo_hexapod.zip        the trained policy (load in eval.py / policy_node)
    rl/logs/                  TensorBoard logs  ->  tensorboard --logdir rl/logs
"""
import os
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

from hexapod_env import HexapodEnv

_HERE = os.path.dirname(os.path.abspath(__file__))


def build_env(n_envs):
    # SubprocVecEnv-style parallelism for throughput; obs normalization helps PPO.
    venv = make_vec_env(HexapodEnv, n_envs=n_envs)
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)
    return venv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=1_000_000)
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(_HERE, "ppo_hexapod"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    venv = build_env(args.n_envs)

    model = PPO(
        "MlpPolicy",
        venv,
        seed=args.seed,
        n_steps=2048,
        batch_size=256,
        gae_lambda=0.95,
        gamma=0.99,
        n_epochs=10,
        ent_coef=0.0,
        learning_rate=3e-4,
        clip_range=0.2,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        tensorboard_log=os.path.join(_HERE, "logs"),
        device="cpu",
    )

    ckpt = CheckpointCallback(save_freq=max(50_000 // args.n_envs, 1),
                              save_path=os.path.join(_HERE, "checkpoints"),
                              name_prefix="ppo_hexapod")

    model.learn(total_timesteps=args.timesteps, callback=ckpt, progress_bar=False)

    model.save(args.out)
    # VecNormalize stats are needed to run the policy later — save them too.
    venv.save(os.path.join(_HERE, "vecnormalize.pkl"))
    print(f"\nsaved policy -> {args.out}.zip")
    print(f"saved obs-norm stats -> {os.path.join(_HERE, 'vecnormalize.pkl')}")


if __name__ == "__main__":
    main()
