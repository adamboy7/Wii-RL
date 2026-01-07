from __future__ import annotations

import argparse
import atexit
import multiprocessing as mp
import os
import re
import signal
import time
from pathlib import Path

import numpy as np
import psutil
import torch

from BTR import Agent, DolphinEnv


def parse_checkpoint_base(checkpoint_base: str) -> str:
    base = checkpoint_base
    if base.endswith(".model"):
        base = base[: -len(".model")]
    return base


def find_latest_checkpoint(resume_dir: Path) -> str:
    model_files = sorted(resume_dir.glob("*.model"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not model_files:
        raise FileNotFoundError(f"No .model files found in {resume_dir}")
    return model_files[0].with_suffix("").name


def derive_agent_name(checkpoint_base: str) -> str:
    return re.sub(r"_[0-9]+M$", "", checkpoint_base)


def kill_dolphin_processes():
    for proc in psutil.process_iter():
        if proc.name() == "Dolphin.exe":
            proc.kill()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-dir", type=str, required=True)
    parser.add_argument("--checkpoint-base", type=str, default=None)
    parser.add_argument("--envs", "--instances", type=int, default=4)

    parser.add_argument("--game", type=str, default="MarioKart")
    parser.add_argument("--frames", type=int, default=2000000000)
    parser.add_argument("--bs", type=int, default=256)
    parser.add_argument("--framestack", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--nstep", type=int, default=3)
    parser.add_argument("--maxpool_size", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--testing", type=bool, default=False)
    parser.add_argument("--munch_alpha", type=float, default=0.9)
    parser.add_argument("--grad_clip", type=int, default=10)
    parser.add_argument("--spectral", type=int, default=1)
    parser.add_argument("--discount", type=float, default=0.997)
    parser.add_argument("--taus", type=int, default=8)
    parser.add_argument("--c", type=int, default=500)
    parser.add_argument("--linear_size", type=int, default=512)
    parser.add_argument("--model_size", type=float, default=2)
    parser.add_argument("--ncos", type=int, default=64)
    parser.add_argument("--per_alpha", type=float, default=0.2)
    parser.add_argument("--layer_norm", type=int, default=0)
    parser.add_argument("--eps_steps", type=int, default=2000000)
    parser.add_argument("--eps_disable", type=int, default=1)

    args = parser.parse_args()

    resume_dir = Path(args.resume_dir).expanduser().resolve()
    if not resume_dir.exists():
        raise FileNotFoundError(f"Resume directory not found: {resume_dir}")

    checkpoint_base = args.checkpoint_base
    if checkpoint_base:
        checkpoint_base = parse_checkpoint_base(checkpoint_base)
    else:
        checkpoint_base = find_latest_checkpoint(resume_dir)

    agent_name = derive_agent_name(checkpoint_base)
    checkpoint_model = resume_dir / f"{checkpoint_base}.model"
    checkpoint_state = resume_dir / f"{checkpoint_base}.state.pt"

    if not checkpoint_model.exists():
        raise FileNotFoundError(f"Checkpoint model not found: {checkpoint_model}")

    os.chdir(resume_dir)

    envs = args.envs
    bs = args.bs
    c = args.c
    lr = args.lr
    framestack = args.framestack
    device_name = args.device
    nstep = args.nstep
    maxpool_size = args.maxpool_size
    munch_alpha = args.munch_alpha
    grad_clip = args.grad_clip
    spectral = args.spectral
    discount = args.discount
    linear_size = args.linear_size
    taus = args.taus
    model_size = args.model_size
    frames = args.frames // 4
    ncos = args.ncos
    per_alpha = args.per_alpha
    eps_steps = args.eps_steps
    eps_disable = args.eps_disable
    layer_norm = args.layer_norm
    testing = args.testing

    if device_name is None:
        gpu = "0"
        device = torch.device("cuda:" + gpu if torch.cuda.is_available() else "cpu")
        print(f"\nDevice: {device}. If this does not say (cuda), you should be worried."
              f" Running this on CPU is extremely slow\n")
    else:
        device = torch.device(device_name)
        print(f"\nDevice: {device}")

    env = DolphinEnv(envs)
    print(env.observation_space)
    print(env.action_space[0])

    replay_period = 64 / envs
    spi = 1

    agent = Agent(n_actions=env.action_space[0].n, input_dims=[framestack, 75, 140], device=device, num_envs=envs,
                  agent_name=agent_name, total_frames=frames, testing=testing, batch_size=bs, lr=lr,
                  maxpool_size=maxpool_size, target_replace=c, spectral=spectral, discount=discount, taus=taus,
                  model_size=model_size, linear_size=linear_size, ncos=ncos, replay_period=replay_period,
                  framestack=framestack, per_alpha=per_alpha, layer_norm=layer_norm, eps_steps=eps_steps,
                  eps_disable=eps_disable, n=nstep, munch_alpha=munch_alpha, grad_clip=grad_clip, imagex=140,
                  imagey=75, spi=spi, loading_checkpoint=True)

    agent.load_models(str(checkpoint_model.with_suffix("")))
    if checkpoint_state.exists():
        agent.load_training_state(str(checkpoint_state))
    else:
        print(f"Warning: training state not found at {checkpoint_state}, resuming with fresh state.")

    scores_temp = []
    steps = agent.env_steps
    last_steps = steps
    last_time = time.time()
    episodes = 0
    current_eval = 0
    scores_count = [0 for _ in range(envs)]
    scores = []
    observation, info = env.reset()
    processes = []
    eval_every = 250000
    next_eval = ((steps // eval_every) + 1) * eval_every

    def handle_shutdown_signal(signum, frame):
        print(f"Shutdown signal ({signum}) received. Killing Dolphin instances.")
        if hasattr(signal, "SIGBREAK") and signum == signal.SIGBREAK:
            checkpoint_name = agent.checkpoint_name()
            print(f"Saving checkpoint due to SIGBREAK: {checkpoint_name}")
            agent.save_model()
            agent.save_training_state(checkpoint_name)
        kill_dolphin_processes()
        raise KeyboardInterrupt

    atexit.register(kill_dolphin_processes)
    signal.signal(signal.SIGINT, handle_shutdown_signal)
    signal.signal(signal.SIGTERM, handle_shutdown_signal)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, handle_shutdown_signal)

    try:
        while steps < frames:
            steps += envs
            try:
                action = agent.choose_action(observation)
            except Exception as e:
                print(f"Error: {e}")
                print(f"Observation: {observation}")
                raise Exception("Stop! Error Occurred")

            env.step_async(action)
            agent.learn()
            observation_, reward, done_, trun_, info = env.step_wait()

            for i in range(envs):
                scores_count[i] += reward[i]
                if done_[i] or trun_[i]:
                    episodes += 1
                    scores.append([scores_count[i], steps])
                    scores_temp.append(scores_count[i])
                    scores_count[i] = 0

            for stream in range(envs):
                if info["Ignore"][stream]:
                    continue

                if info["First"][stream]:
                    observation[stream] = observation_[stream]

                next_obs = observation_[stream] if not trun_[stream] else np.array(info["final_observation"][stream])
                agent.store_transition(observation[stream], action[stream], reward[stream], next_obs,
                                       done_[stream], trun_[stream], stream=stream)

            observation = observation_

            if steps % 600 == 0 and len(scores) > 0:
                avg_score = np.mean(scores_temp[-50:])
                if episodes % 1 == 0:
                    print('{} avg score {:.2f} total_timesteps {:.0f} fps {:.2f} games {}'
                          .format(agent_name, avg_score, steps,
                                  (steps - last_steps) / (time.time() - last_time), episodes), flush=True)
                    last_steps = steps
                    last_time = time.time()

            if steps >= next_eval or steps >= frames:
                print("Saving Model...")
                agent.save_model()

                if not testing:
                    fname = agent_name + "Experiment.npy"
                    np.save(fname, np.array(scores))

                current_eval += 1
                next_eval += eval_every
    except KeyboardInterrupt:
        print("Keyboard interrupt received. Killing Dolphin instances.")
        kill_dolphin_processes()
        raise

    for process in processes:
        process.join()

    print("Evaluations finished, job completed successfully!")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
