from __future__ import annotations

import argparse
import atexit
import multiprocessing as mp
import os
import random
import re
import signal
import time

import matplotlib.pyplot as plt
import numpy as np
import psutil
import torch

from BTR import Agent, format_arguments, non_default_args
from DolphinEnv import DolphinEnv


CHECKPOINT_EXTENSIONS = (".pt", ".model", ".pth")


def strip_checkpoint_extension(checkpoint_name: str) -> str:
    for extension in CHECKPOINT_EXTENSIONS:
        if checkpoint_name.endswith(extension):
            return checkpoint_name[: -len(extension)]
    return checkpoint_name


def parse_checkpoint_step(checkpoint_name: str) -> int | None:
    base_name = strip_checkpoint_extension(checkpoint_name)
    match = re.match(r"^.+_(?P<step>\d+)M$", base_name)
    if match:
        return int(match.group("step"))
    return None


def resolve_latest_checkpoint(run_dir: str) -> str:
    candidates = []
    for entry in os.listdir(run_dir):
        if not entry.endswith(CHECKPOINT_EXTENSIONS):
            continue
        path = os.path.join(run_dir, entry)
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            continue
        step = parse_checkpoint_step(entry)
        candidates.append((step, mtime, entry))

    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint artifacts found in {run_dir} with extensions {CHECKPOINT_EXTENSIONS}."
        )

    steps_available = [candidate for candidate in candidates if candidate[0] is not None]
    if steps_available:
        max_step = max(candidate[0] for candidate in steps_available)
        step_candidates = [candidate for candidate in steps_available if candidate[0] == max_step]
        return max(step_candidates, key=lambda candidate: candidate[1])[2]

    return max(candidates, key=lambda candidate: candidate[1])[2]


def derive_agent_name(checkpoint_name: str) -> str:
    base_name = strip_checkpoint_extension(os.path.basename(checkpoint_name))
    match = re.match(r"^(?P<agent>.+)_\d+M$", base_name)
    if match:
        return match.group("agent")
    return base_name


def load_existing_scores(agent_name: str) -> list:
    scores_file = f"{agent_name}Experiment.npy"
    if os.path.exists(scores_file):
        return np.load(scores_file, allow_pickle=True).tolist()
    return []


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str)
    parser.add_argument(
        "--latest",
        action="store_true",
        help=(
            "Resolve the most recent checkpoint in run_dir based on step suffix "
            "(e.g., *_123M) or file mtime as a fallback."
        ),
    )
    parser.add_argument("--run_dir", type=str, default=".")

    parser.add_argument("--game", type=str, default="MarioKart")
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--frames", type=int, default=2000000000)
    parser.add_argument("--eval_envs", type=int, default=10)
    parser.add_argument("--bs", type=int, default=256)
    parser.add_argument("--repeat", type=int, default=0)
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
    parser.add_argument("--per_beta_anneal", type=int, default=0)
    parser.add_argument("--layer_norm", type=int, default=0)
    parser.add_argument("--eps_steps", type=int, default=2000000)
    parser.add_argument("--eps_disable", type=int, default=1)
    parser.add_argument(
        "--save_state_on_eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save training state alongside model checkpoints on eval intervals.",
    )
    parser.add_argument(
        "--resume_without_state",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Allow resuming without a saved training state. "
            "When the .state.pt file is missing, reset counters and continue with "
            "weights-only loading."
        ),
    )

    args = parser.parse_args()

    arg_string = non_default_args(args, parser)
    formatted_string = format_arguments(arg_string)
    print(formatted_string)

    if args.run_dir:
        os.chdir(args.run_dir)

    if args.latest and args.checkpoint:
        parser.error("--latest cannot be used together with --checkpoint.")

    if args.latest or not args.checkpoint:
        checkpoint_name = resolve_latest_checkpoint(".")
        print(f"Resolved latest checkpoint: {checkpoint_name}")
    else:
        checkpoint_name = args.checkpoint
    agent_name = derive_agent_name(checkpoint_name)

    game = args.game
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

    replay_period = 64 / envs
    spi = 1

    if args.testing:
        envs = 2
        num_envs = 2
        eval_every = 11580000
        n_steps = 11560000
        bs = 32
    else:
        num_envs = envs
        n_steps = frames
        eval_every = 250000
    next_eval = eval_every

    print("Currently Playing Game: " + str(game))

    if device_name is None:
        gpu = "0"
        device = torch.device("cuda:" + gpu if torch.cuda.is_available() else "cpu")
        print(
            f"\nDevice: {device}. If this does not say (cuda), you should be worried."
            f" Running this on CPU is extremely slow\n"
        )
    else:
        device = torch.device(device_name)
        print(f"\nDevice: {device}")

    env = DolphinEnv(envs)
    print(env.observation_space)
    print(env.action_space[0])

    agent = Agent(
        n_actions=env.action_space[0].n,
        input_dims=[framestack, 75, 140],
        device=device,
        num_envs=num_envs,
        agent_name=agent_name,
        total_frames=n_steps,
        testing=args.testing,
        batch_size=bs,
        lr=lr,
        maxpool_size=maxpool_size,
        target_replace=c,
        spectral=spectral,
        discount=discount,
        taus=taus,
        model_size=model_size,
        linear_size=linear_size,
        ncos=ncos,
        replay_period=replay_period,
        framestack=framestack,
        per_alpha=per_alpha,
        layer_norm=layer_norm,
        eps_steps=eps_steps,
        eps_disable=eps_disable,
        n=nstep,
        munch_alpha=munch_alpha,
        grad_clip=grad_clip,
        imagex=140,
        imagey=75,
        spi=spi,
    )

    agent.load_models(checkpoint_name)
    state_path = f"{checkpoint_name}.state.pt"
    try:
        training_state = agent.load_training_state(checkpoint_name)
    except FileNotFoundError:
        guidance = (
            f"Training state not found at {state_path}. "
            "Generate it by enabling --save_state_on_eval or by sending SIGBREAK "
            "(Ctrl+Break) to dump a state checkpoint."
        )
        if not args.resume_without_state:
            print(guidance)
            raise
        print(f"{guidance} Proceeding with weights-only resume.")
        training_state = {}
        agent.env_steps = 0
        agent.grad_steps = 0
        agent.replay_ratio_cnt = 0
        agent.eval_every = None
        agent.next_eval = None

    loaded_eval_every = training_state.get("eval_every")
    loaded_next_eval = training_state.get("next_eval")

    if loaded_eval_every is not None:
        eval_every = loaded_eval_every
    else:
        print(
            "Training state missing eval_every; using current value "
            f"({eval_every})."
        )

    if loaded_next_eval is not None:
        next_eval = loaded_next_eval
    else:
        print(
            "Training state missing next_eval; deriving fallback based on "
            f"env_steps={agent.env_steps} and eval_every={eval_every}."
        )
        next_eval = ((agent.env_steps // eval_every) + 1) * eval_every

    agent.eval_every = eval_every
    agent.next_eval = next_eval

    scores = load_existing_scores(agent_name)
    scores_temp = [score[0] for score in scores]
    steps = agent.env_steps
    last_steps = steps
    last_time = time.time()
    episodes = len(scores)
    current_eval = 0
    scores_count = [0 for _ in range(num_envs)]
    observation, info = env.reset()
    processes = []

    def kill_dolphin_processes():
        for proc in psutil.process_iter():
            if proc.name() == "Dolphin.exe":
                proc.kill()

    def handle_shutdown_signal(signum, frame):
        print(f"Shutdown signal ({signum}) received. Killing Dolphin instances.")
        if hasattr(signal, "SIGBREAK") and signum == signal.SIGBREAK:
            print(f"Saving checkpoint due to SIGBREAK: {checkpoint_name}")
            agent.save_model()
            agent.save_training_state(
                checkpoint_name, eval_every=eval_every, next_eval=next_eval
            )
        kill_dolphin_processes()
        raise KeyboardInterrupt

    atexit.register(kill_dolphin_processes)
    signal.signal(signal.SIGINT, handle_shutdown_signal)
    signal.signal(signal.SIGTERM, handle_shutdown_signal)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, handle_shutdown_signal)

    try:
        while steps < n_steps:
            steps += num_envs
            try:
                action = agent.choose_action(observation)
            except Exception as e:
                print(f"Error: {e}")
                print(f"Observation: {observation}")
                raise Exception("Stop! Error Occurred")

            env.step_async(action)
            agent.learn()
            observation_, reward, done_, trun_, info = env.step_wait()

            for i in range(num_envs):
                scores_count[i] += reward[i]
                if done_[i] or trun_[i]:
                    episodes += 1
                    scores.append([scores_count[i], steps])
                    scores_temp.append(scores_count[i])
                    scores_count[i] = 0

            for stream in range(num_envs):
                if info["Ignore"][stream]:
                    continue

                if info["First"][stream]:
                    observation[stream] = observation_[stream]

                next_obs = (
                    observation_[stream]
                    if not trun_[stream]
                    else np.array(info["final_observation"][stream])
                )
                agent.store_transition(
                    observation[stream],
                    action[stream],
                    reward[stream],
                    next_obs,
                    done_[stream],
                    trun_[stream],
                    stream=stream,
                )

            observation = observation_

            if steps % 600 == 0 and len(scores) > 0:
                avg_score = np.mean(scores_temp[-50:])
                if episodes % 1 == 0:
                    print(
                        "{} avg score {:.2f} total_timesteps {:.0f} fps {:.2f} games {}".format(
                            agent_name,
                            avg_score,
                            steps,
                            (steps - last_steps) / (time.time() - last_time),
                            episodes,
                        ),
                        flush=True,
                    )
                    last_steps = steps
                    last_time = time.time()

            if steps >= next_eval or steps >= n_steps:
                print("Saving Model...")
                agent.save_model()
                if args.save_state_on_eval:
                    checkpoint_name = agent.checkpoint_name()
                    agent.save_training_state(
                        checkpoint_name, eval_every=eval_every, next_eval=next_eval
                    )

                if not args.testing:
                    np.save(f"{agent_name}Experiment.npy", np.array(scores))

                window = 200
                episode_scores = np.array([s[0] for s in scores])
                episode_steps = np.array([s[1] for s in scores])

                if len(scores) < window:
                    print(
                        f"Not enough episodes for a window size of {window}, reducing to {len(scores)}"
                    )
                    window = len(scores)

                cumsum = np.cumsum(np.insert(episode_scores, 0, 0))
                moving_avg = (cumsum[window:] - cumsum[:-window]) / window
                avg_steps = episode_steps[window - 1 :]

                plt.figure(figsize=(8, 5))
                plt.plot(avg_steps, moving_avg, label=f"{window}-episode moving average")
                plt.xlabel("Steps")
                plt.ylabel("Score (smoothed)")
                plt.title("Smoothed Episode Scores Over Time")
                plt.grid(True)
                plt.tight_layout()
                plt.legend()

                plt.savefig("scores_over_time_smoothed.png")
                plt.close()

                current_eval += 1
                next_eval += eval_every
                agent.next_eval = next_eval

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
