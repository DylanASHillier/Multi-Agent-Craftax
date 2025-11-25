#!/usr/bin/env python3
import pickle
from typing import Dict, Any, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import cv2

from craftax.craftax_classic.envs.craftax_state import EnvParams, StaticEnvParams
from craftax.craftax_classic.train.metacontroller_bc_aux import ClassicMetaController
from craftax.craftax_classic.renderer import render_craftax_pixels

# -------------------------------------------------------------------
# Achievement indices:
#   0: COLLECT_WOOD
#   2: EAT_COW
#   4: COLLECT_DRINK
# -------------------------------------------------------------------

def make_empty_weights():
    # Length should match EnvParams.achievement_weights (22 in your snippet)
    return jnp.zeros(22, dtype=jnp.float32)

WOOD_WEIGHTS  = make_empty_weights().at[0].set(10.0)  # COLLECT_WOOD
COW_WEIGHTS   = make_empty_weights().at[2].set(10.0)  # EAT_COW
DRINK_WEIGHTS = make_empty_weights().at[4].set(10.0)  # COLLECT_DRINK


def make_metacontroller_for_task(achievement_weights, num_players: int = 2) -> ClassicMetaController:
    """
    Create a ClassicMetaController configured for a specific task via achievement_weights.
    """
    env_params = EnvParams(
        max_timesteps=300,
        achievement_weights=achievement_weights,
    )

    static_params = StaticEnvParams(num_players=num_players)

    metacontroller = ClassicMetaController(
        env_params=env_params,
        static_parameters=static_params,
        num_envs=200,
        num_steps=500,
        num_iterations=400,
        num_minibatches=10,
        fixed_timesteps=False,
        learning_rate=2.5e-4,
        anneal_lr=False,
        update_epochs=5,
        max_grad_norm=1.0,
        wandb_project="MARLCraftax",
        aux_coef=0.1,
    )

    # Set RNG seed for reproducibility (optional)
    metacontroller.rng = jax.random.PRNGKey(4253)
    return metacontroller


def train_specialist(task_name: str, weights: jnp.ndarray) -> Tuple[ClassicMetaController, Any]:
    """
    Train one specialist metacontroller for a given task and return it with the trained params.
    """
    print(f"\n===== Training specialist for task: {task_name} =====")
    metacontroller = make_metacontroller_for_task(weights, num_players=2)

    # Train -> returns (agent_params, aux_params)
    (params, aux_params), opt_state, log = metacontroller.train()

    # Save just the policy params if you want
    with open(f"params_ippo_{task_name}.p", "wb") as f:
        pickle.dump(params, f)

    return metacontroller, params


def collect_episodes_for_specialist(
    agent_id: int,
    task_name: str,
    metacontroller: ClassicMetaController,
    params: Any,
    num_episodes: int,
) -> List[Dict[str, Any]]:
    """
    Run run_one_episode num_episodes times and return a list of demo dicts:
      {
        "agent_id": 0/1/2,
        "task_name": "...",
        "episode_idx": k,
        "states":  ...,
        "actions": ...,
        "logits":  ...,
        "rewards": ...
      }
    """
    demos = []
    for ep in range(num_episodes):
        print(f"[{task_name}] Running episode {ep+1}/{num_episodes}")
        states, actions, logits, rewards = metacontroller.run_one_episode(params)

        demo = {
            "agent_id": agent_id,
            "task_name": task_name,
            "episode_idx": ep,
            "states": states,
            "actions": actions,
            "logits": logits,
            "rewards": rewards,
        }
        demos.append(demo)

    return demos


def save_episode_videos(states, task_name: str, fps: float = 15.0):
    """
    Save states from one episode to one MP4 per player.

    states: list/array of EnvState over time (T steps)
    """
    if len(states) == 0:
        print(f"[{task_name}] No states to render, skipping video.")
        return

    # Determine num_players from the state
    first_state = states[0]
    num_players = first_state.player_position.shape[0]

    tile_size = 64  # adjust if your renderer uses a different tile size

    # Render a single frame to get frame size
    first_img = render_craftax_pixels(first_state, tile_size, num_players, 0)
    first_frame = np.asarray(first_img, dtype=np.uint8)[..., ::-1]  # RGB -> BGR
    H, W = first_frame.shape[:2]
    frame_size = (W, H)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    for player_idx in range(num_players):
        video_name = f"{task_name}_agent{player_idx}_last_episode.mp4"
        print(f"[{task_name}] Writing video: {video_name}")

        out = cv2.VideoWriter(video_name, fourcc, fps, frame_size)

        for s in states:
            img = render_craftax_pixels(s, tile_size, num_players, player_idx)
            frame = np.asarray(img, dtype=np.uint8)[..., ::-1]  # RGB -> BGR

            if frame.shape[:2] != (H, W):
                frame = cv2.resize(frame, frame_size, interpolation=cv2.INTER_NEAREST)

            out.write(frame)

        out.release()

    print(f"[{task_name}] Saved videos for {num_players} agents.")


def save_task_pickle(task_name: str, demos: List[Dict[str, Any]]) -> str:
    """
    Save demos for a single task to a separate pickle file.
    Format is the same list-of-dicts structure used in this script.
    """
    filename = f"demos_{task_name}.pickle"
    with open(filename, "wb") as f:
        pickle.dump(demos, f)
    print(f"[{task_name}] Saved {len(demos)} demos to {filename}")
    return filename


def combine_task_pickles_to_flat(
    pickle_paths: List[str],
    output_path: str = "combined_flat_dataset.pickle",
):
    """
    Load multiple per-task demo pickle files and combine them into a single list of
    [agent_id, state, action, logits, reward] records, then save to output_path.

    Assumes each pickle contains a list of demo dicts with keys:
      "agent_id", "states", "actions", "logits", "rewards"
    where each of states/actions/logits/rewards is time-major (T, ...).
    """
    combined: List[List[Any]] = []

    for path in pickle_paths:
        print(f"Loading demos from {path}")
        with open(path, "rb") as f:
            demos = pickle.load(f)

        for demo in demos:
            agent_id = demo["agent_id"]
            states = demo["states"]
            actions = demo["actions"]
            logits = demo["logits"]
            rewards = demo["rewards"]

            # Assume time dimension is the first axis
            T = len(states)
            for t in range(T):
                record = [
                    agent_id,
                    states[t],
                    actions[t],
                    logits[t],
                    rewards[t],
                ]
                combined.append(record)

    with open(output_path, "wb") as f:
        pickle.dump(combined, f)

    print(f"Saved {len(combined)} [agent, state, action, logits, reward] records to {output_path}")


if __name__ == "__main__":
    # How many demo episodes per specialist to generate
    NUM_EPISODES_PER_AGENT = 1  # <-- change this as you like

    # ----------------------------------------------------------------
    # 1) Train specialists
    #    agent_id mapping:
    #      0 -> wood specialist
    #      1 -> drink specialist
    #      2 -> cow specialist
    # ----------------------------------------------------------------
    wood_mc, wood_params   = train_specialist("collect_wood",  WOOD_WEIGHTS)
    drink_mc, drink_params = train_specialist("collect_drink", DRINK_WEIGHTS)
    cow_mc, cow_params     = train_specialist("eat_cow",       COW_WEIGHTS)

    # ----------------------------------------------------------------
    # 2) Collect demo episodes from each specialist
    # ----------------------------------------------------------------
    all_demos: List[Dict[str, Any]] = []
    task_pickle_paths: List[str] = []

    # agent 0: wood
    wood_demos = collect_episodes_for_specialist(
        agent_id=0,
        task_name="collect_wood",
        metacontroller=wood_mc,
        params=wood_params,
        num_episodes=NUM_EPISODES_PER_AGENT,
    )
    all_demos.extend(wood_demos)
    task_pickle_paths.append(save_task_pickle("collect_wood", wood_demos))
    # video for last episode of this task
    if len(wood_demos) > 0:
        save_episode_videos(wood_demos[-1]["states"], "collect_wood", fps=15.0)

    # agent 1: drink (water)
    drink_demos = collect_episodes_for_specialist(
        agent_id=1,
        task_name="collect_drink",
        metacontroller=drink_mc,
        params=drink_params,
        num_episodes=NUM_EPISODES_PER_AGENT,
    )
    all_demos.extend(drink_demos)
    task_pickle_paths.append(save_task_pickle("collect_drink", drink_demos))
    if len(drink_demos) > 0:
        save_episode_videos(drink_demos[-1]["states"], "collect_drink", fps=15.0)

    # agent 2: cow
    cow_demos = collect_episodes_for_specialist(
        agent_id=2,
        task_name="eat_cow",
        metacontroller=cow_mc,
        params=cow_params,
        num_episodes=NUM_EPISODES_PER_AGENT,
    )
    all_demos.extend(cow_demos)
    task_pickle_paths.append(save_task_pickle("eat_cow", cow_demos))
    if len(cow_demos) > 0:
        save_episode_videos(cow_demos[-1]["states"], "eat_cow", fps=15.0)

    # ----------------------------------------------------------------
    # 3) Optionally: save everything to a single pickle file
    #    (same list-of-episode-dicts format as before)
    # ----------------------------------------------------------------
    output_path = "irl_demonstrations_specialists_all_tasks.pickle"
    with open(output_path, "wb") as f:
        pickle.dump(all_demos, f)
    print(f"\nSaved {len(all_demos)} demos to {output_path}")
    print("Agent mapping: 0=wood, 1=drink, 2=cow")

    # ----------------------------------------------------------------
    # 4) (Optional) Combine per-task pickle files into flat transitions
    #    Uncomment this if you want to automatically build the flat dataset.
    # ----------------------------------------------------------------
    # combine_task_pickles_to_flat(task_pickle_paths, output_path="combined_flat_dataset.pickle")
