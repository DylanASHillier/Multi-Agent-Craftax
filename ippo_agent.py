# #!/usr/bin/env python3
# import pickle
# from typing import Dict, Any, List, Tuple

# import jax
# import jax.numpy as jnp
# import numpy as np
# import cv2

# from craftax.craftax_classic.envs.craftax_state import EnvParams, StaticEnvParams
# from craftax.craftax_classic.train.metacontroller_bc_aux import ClassicMetaController
# from craftax.craftax_classic.renderer import render_craftax_pixels


# def make_empty_weights():
#     return jnp.zeros(22, dtype=jnp.float32)


# WOOD_WEIGHTS  = make_empty_weights().at[0].set(50.0)
# COW_WEIGHTS   = make_empty_weights().at[2].set(50.0)
# DRINK_WEIGHTS = make_empty_weights().at[4].set(50.0)


# def make_metacontroller_for_task(
#     achievement_weights,
#     num_players: int = 2,
# ) -> ClassicMetaController:
#     """
#     Create a ClassicMetaController configured for a specific task via achievement_weights.
#     """
#     env_params = EnvParams(
#         max_timesteps=300,
#         achievement_weights=achievement_weights,
#     )

#     static_params = StaticEnvParams(num_players=num_players)

#     metacontroller = ClassicMetaController(
#         env_params=env_params,
#         static_parameters=static_params,
#         num_envs=200,
#         num_steps=50,
#         num_iterations=400,
#         num_minibatches=10,
#         fixed_timesteps=False,
#         learning_rate=2.5e-4,
#         anneal_lr=False,
#         update_epochs=10,
#         max_grad_norm=1.0,
#         ent_coef=0.02,
#         wandb_project="MARLCraftax",
#         aux_coef=0.0,
#     )

#     """
#     learning rate: adjust where it is learning but also not learning to fast 
#     - do a parameter sweep, varying the entropy coefficients
#     - number of iterations 
#     - number of minibatches maybe 
#     """

#     metacontroller.rng = jax.random.PRNGKey(4253)
#     return metacontroller


# def train_specialist(task_name: str, weights: jnp.ndarray) -> Tuple[ClassicMetaController, Any]:
#     """
#     Train one specialist metacontroller for a given task and return it with the trained params.
#     """
#     print(f"\n===== Training specialist for task: {task_name} =====")
#     metacontroller = make_metacontroller_for_task(weights, num_players=2)

#     # Train: returns (agent_params, aux_params)
#     (params, aux_params), opt_state, log = metacontroller.train()

#     # Save just the policy params if you want
#     with open(f"params_ippo_{task_name}.p", "wb") as f:
#         pickle.dump(params, f)
#     print(f"[{task_name}] Saved trained params to params_ippo_{task_name}.p")

#     return metacontroller, params


# def collect_episodes_for_specialist(
#     agent_id: int,
#     task_name: str,
#     metacontroller: ClassicMetaController,
#     params: Any,
#     num_episodes: int,
# ) -> List[Dict[str, Any]]:
#     demos = []
#     for ep in range(num_episodes):
#         print(f"[{task_name}] Running episode {ep+1}/{num_episodes}")
#         states, actions, logits, rewards = metacontroller.run_one_episode(params)

#         demo = {
#             "agent_id": agent_id,
#             "task_name": task_name,
#             "episode_idx": ep,
#             "states": states,
#             "actions": actions,
#             "logits": logits,
#             "rewards": rewards,
#         }
#         demos.append(demo)

#     print(f"[{task_name}] Collected {len(demos)} episodes.")
#     return demos


# def save_episode_videos(states, task_name: str, fps: float = 15.0):
#     """
#     Save states from one episode to one MP4 per player.
#     """
#     if len(states) == 0:
#         print(f"[{task_name}] No states to render, skipping video.")
#         return

#     # Determine num_players from the state
#     first_state = states[0]
#     num_players = first_state.player_position.shape[0]

#     tile_size = 64

#     # Render a single frame to get frame size
#     first_img = render_craftax_pixels(first_state, tile_size, num_players, 0)
#     first_frame = np.asarray(first_img, dtype=np.uint8)[..., ::-1]
#     H, W = first_frame.shape[:2]
#     frame_size = (W, H)

#     fourcc = cv2.VideoWriter_fourcc(*"mp4v")

#     for player_idx in range(num_players):
#         video_name = f"{task_name}_agent{player_idx}_last_episode.mp4"
#         print(f"[{task_name}] Writing video: {video_name}")

#         out = cv2.VideoWriter(video_name, fourcc, fps, frame_size)

#         for s in states:
#             img = render_craftax_pixels(s, tile_size, num_players, player_idx)
#             frame = np.asarray(img, dtype=np.uint8)[..., ::-1]

#             if frame.shape[:2] != (H, W):
#                 frame = cv2.resize(frame, frame_size, interpolation=cv2.INTER_NEAREST)

#             out.write(frame)

#         out.release()

#     print(f"[{task_name}] Saved videos for {num_players} agents.")


# def save_task_pickle(task_name: str, demos: List[Dict[str, Any]]) -> str:
#     """
#     Save the list of episode-level demos for one task into a pickle file.
#     """
#     filename = f"demos_{task_name}.pickle"
#     with open(filename, "wb") as f:
#         pickle.dump(demos, f)
#     print(f"[{task_name}] Saved {len(demos)} demos to {filename}")
#     return filename


# def combine_task_pickles_to_flat(
#     pickle_paths: List[str],
#     output_path: str = "combined_flat_dataset.pickle",
# ):
#     combined: List[List[Any]] = []

#     for path in pickle_paths:
#         print(f"Loading demos from {path}")
#         with open(path, "rb") as f:
#             demos = pickle.load(f)

#         for demo in demos:
#             agent_id = demo["agent_id"]
#             states = demo["states"]
#             actions = demo["actions"]
#             logits = demo["logits"]
#             rewards = demo["rewards"]

#             T = len(states)
#             for t in range(T):
#                 record = [
#                     agent_id,
#                     states[t],
#                     actions[t],
#                     logits[t],
#                     rewards[t],
#                 ]
#                 combined.append(record)

#     with open(output_path, "wb") as f:
#         pickle.dump(combined, f)

#     print(f"Saved {len(combined)} [agent, state, action, logits, reward] records to {output_path}")


# def train_and_save_for_task(
#     agent_id: int,
#     task_name: str,
#     weights: jnp.ndarray,
#     num_episodes: int,
# ) -> Tuple[List[Dict[str, Any]], str]:
#     """
#     Convenience wrapper:
#     - trains specialist for this task
#     - collects episodes
#     - saves per-task demos pickle
#     - saves video of last episode
#     Returns (demos, demos_pickle_path).
#     """
#     # 1) Train specialist
#     mc, params = train_specialist(task_name, weights)

#     # 2) Collect demos
#     demos = collect_episodes_for_specialist(
#         agent_id=agent_id,
#         task_name=task_name,
#         metacontroller=mc,
#         params=params,
#         num_episodes=num_episodes,
#     )

#     # 3) Save per-task pickle
#     demos_path = save_task_pickle(task_name, demos)

#     # 4) Save video from last episode (if any)
#     if len(demos) > 0:
#         last_states = demos[-1]["states"]
#         save_episode_videos(last_states, task_name, fps=15.0)

#     return demos, demos_path


# if __name__ == "__main__":
#     # How many demo episodes per specialist to generate
#     NUM_EPISODES_PER_AGENT = 100

#     all_demos: List[Dict[str, Any]] = []
#     task_pickle_paths: List[str] = []

#     # agent 0: wood
#     wood_demos, wood_path = train_and_save_for_task(
#         agent_id=0,
#         task_name="collect_wood",
#         weights=WOOD_WEIGHTS,
#         num_episodes=NUM_EPISODES_PER_AGENT,
#     )
#     all_demos.extend(wood_demos)
#     task_pickle_paths.append(wood_path)

#     # agent 1: drink (water)
#     drink_demos, drink_path = train_and_save_for_task(
#         agent_id=1,
#         task_name="collect_drink",
#         weights=DRINK_WEIGHTS,
#         num_episodes=NUM_EPISODES_PER_AGENT,
#     )
#     all_demos.extend(drink_demos)
#     task_pickle_paths.append(drink_path)

#     # agent 2: cow
#     cow_demos, cow_path = train_and_save_for_task(
#         agent_id=2,
#         task_name="eat_cow",
#         weights=COW_WEIGHTS,
#         num_episodes=NUM_EPISODES_PER_AGENT,
#     )
#     all_demos.extend(cow_demos)
#     task_pickle_paths.append(cow_path)

#     # 3) Save one big pickle with *all* episode-level demos
#     output_path = "irl_demonstrations_specialists_all_tasks.pickle"
#     with open(output_path, "wb") as f:
#         pickle.dump(all_demos, f)
#     print(f"\nSaved {len(all_demos)} demos to {output_path}")
#     print("Agent mapping: 0=wood, 1=drink, 2=cow")

#     # # 4) Also create a flat transition dataset across all tasks
#     # combine_task_pickles_to_flat(task_pickle_paths, output_path="combined_flat_dataset.pickle")



"""
This is for parameter sweeping****************************************************************************
"""


# #!/usr/bin/env python3
# import pickle  # still here in case you later want to re-enable saving params
# from typing import Dict, Any, List, Tuple

# import jax
# import jax.numpy as jnp

# from craftax.craftax_classic.envs.craftax_state import EnvParams, StaticEnvParams
# from craftax.craftax_classic.train.metacontroller_bc_aux import ClassicMetaController


# def make_empty_weights():
#     return jnp.zeros(22, dtype=jnp.float32)


# WOOD_WEIGHTS  = make_empty_weights().at[0].set(50.0)
# COW_WEIGHTS   = make_empty_weights().at[2].set(50.0)
# DRINK_WEIGHTS = make_empty_weights().at[4].set(50.0)

# # Map base_task_name -> W&B project name
# AGENT_PROJECTS = {
#     "collect_drink": "water_gather_agent",
#     "collect_wood": "carpenter_agent",
#     "eat_cow": "hunter_agent",
# }


# def make_metacontroller_for_task(
#     achievement_weights,
#     num_players: int = 2,
#     num_iterations: int = 400,
#     ent_coef: float = 0.02,
#     learning_rate: float = 2.5e-4,
#     num_minibatches: int = 16,
#     wandb_project: str = "",
#     wandb_run_name: str | None = None,
# ) -> ClassicMetaController:
#     """
#     Create a ClassicMetaController configured for a specific task via achievement_weights,
#     with hyperparameters exposed for sweeps + W&B project/run naming.
#     """
#     env_params = EnvParams(
#         max_timesteps=300,
#         achievement_weights=achievement_weights,
#     )

#     static_params = StaticEnvParams(num_players=num_players)

#     print("PROJECT: ", wandb_project)

#     metacontroller = ClassicMetaController(
#         env_params=env_params,
#         static_parameters=static_params,
#         num_envs=200,
#         num_steps=50,
#         num_iterations=num_iterations,
#         num_minibatches=num_minibatches,
#         fixed_timesteps=False,
#         learning_rate=learning_rate,
#         anneal_lr=False,
#         update_epochs=10,
#         max_grad_norm=1.0,
#         ent_coef=ent_coef,
#         wandb_project=wandb_project,
#         wandb_run_name=wandb_run_name,
#         aux_coef=0.0,
#     )

#     metacontroller.rng = jax.random.PRNGKey(4253)
#     return metacontroller


# def _make_cfg_suffix(
#     num_iterations: int,
#     ent_coef: float,
#     learning_rate: float,
#     num_minibatches: int,
# ) -> str:
#     """
#     Create a filesystem-friendly suffix representing the hyperparameters.
#     Example: it400_ent0p02_lr2p5e-04_mb16
#     """
#     def f(x: float) -> str:
#         return str(x).replace(".", "p")

#     return (
#         f"it{num_iterations}_"
#         f"ent{f(ent_coef)}_"
#         f"lr{f(learning_rate)}_"
#         f"mb{num_minibatches}"
#     )


# def train_specialist(
#     task_name: str,
#     weights: jnp.ndarray,
#     num_iterations: int,
#     ent_coef: float,
#     learning_rate: float,
#     num_minibatches: int = 16,
#     cfg_suffix: str = "",
#     wandb_project: str = "",
# ) -> Tuple[ClassicMetaController, Any]:
#     """
#     Train one specialist metacontroller for a given task and return it with the trained params.
#     Does NOT save trajectories or videos.
#     """
#     full_task_name = f"{task_name}_{cfg_suffix}" if cfg_suffix else task_name
#     print(f"\n===== Training specialist for task: {full_task_name} =====")
#     print(
#         f"    num_iterations={num_iterations}, ent_coef={ent_coef}, "
#         f"learning_rate={learning_rate}, num_minibatches={num_minibatches}"
#     )

#     # Use full_task_name as W&B run name
#     wandb_run_name = full_task_name

#     metacontroller = make_metacontroller_for_task(
#         achievement_weights=weights,
#         num_players=2,
#         num_iterations=num_iterations,
#         ent_coef=ent_coef,
#         learning_rate=learning_rate,
#         num_minibatches=num_minibatches,
#         wandb_project=wandb_project,
#         wandb_run_name=wandb_run_name,
#     )

#     # Train: returns (agent_params, aux_params)
#     (params, aux_params), opt_state, log = metacontroller.train()

#     # Optionally save just the policy params (still allowed, but no trajectory/video)
#     params_fname = f"params_ippo_{full_task_name}.p"
#     with open(params_fname, "wb") as f:
#         pickle.dump(params, f)
#     print(f"[{full_task_name}] Saved trained params to {params_fname}")

#     return metacontroller, params


# def train_and_log_for_task(
#     agent_id: int,
#     base_task_name: str,
#     weights: jnp.ndarray,
#     num_episodes: int,   # kept for API compatibility, but unused now
#     num_iterations: int,
#     ent_coef: float,
#     learning_rate: float,
#     num_minibatches: int = 16,
# ) -> None:
#     """
#     Convenience wrapper:
#     - trains specialist for this task with given hyperparams
#     - logs to W&B
#     - DOES NOT save trajectories or videos.
#     """
#     cfg_suffix = _make_cfg_suffix(
#         num_iterations,
#         ent_coef,
#         learning_rate,
#         num_minibatches,
#     )
#     full_task_name = f"{base_task_name}_{cfg_suffix}"

#     # Choose W&B project based on the task
#     wandb_project = AGENT_PROJECTS.get(base_task_name, "MARLCraftax")

#     print("wandb_project: ", wandb_project)

#     # Train specialist (no demo collection)
#     _mc, _params = train_specialist(
#         task_name=base_task_name,
#         weights=weights,
#         num_iterations=num_iterations,
#         ent_coef=ent_coef,
#         learning_rate=learning_rate,
#         num_minibatches=num_minibatches,
#         cfg_suffix=cfg_suffix,
#         wandb_project=wandb_project,
#     )

#     print(f"[{full_task_name}] Training complete (no trajectories/videos saved).")


# if __name__ == "__main__":
#     # How many demo episodes per specialist to generate
#     # (kept for API symmetry, but currently unused)
#     NUM_EPISODES_PER_AGENT = 100

#     # -----------------------
#     # Hyperparameter sweep
#     # -----------------------
#     SWEEP_NUM_ITERATIONS  = [200, 400]
#     SWEEP_ENT_COEFS       = [0.01, 0.02]
#     SWEEP_LEARNING_RATES  = [2.5e-4, 5e-4]
#     SWEEP_NUM_MINIBATCHES = [10, 16]

#     # Quadruple nested sweep
#     for num_iterations in SWEEP_NUM_ITERATIONS:
#         for ent_coef in SWEEP_ENT_COEFS:
#             for learning_rate in SWEEP_LEARNING_RATES:
#                 for num_minibatches in SWEEP_NUM_MINIBATCHES:
#                     print("\n==============================================")
#                     print("Starting sweep config:")
#                     print(f"  num_iterations={num_iterations}")
#                     print(f"  ent_coef={ent_coef}")
#                     print(f"  learning_rate={learning_rate}")
#                     print(f"  num_minibatches={num_minibatches}")
#                     print("==============================================\n")

#                     # agent 0: wood -> carpenter_agent project
#                     train_and_log_for_task(
#                         agent_id=0,
#                         base_task_name="collect_wood",
#                         weights=WOOD_WEIGHTS,
#                         num_episodes=NUM_EPISODES_PER_AGENT,
#                         num_iterations=num_iterations,
#                         ent_coef=ent_coef,
#                         learning_rate=learning_rate,
#                         num_minibatches=num_minibatches,
#                     )

#                     # agent 1: drink (water) -> water_gather_agent project
#                     train_and_log_for_task(
#                         agent_id=1,
#                         base_task_name="collect_drink",
#                         weights=DRINK_WEIGHTS,
#                         num_episodes=NUM_EPISODES_PER_AGENT,
#                         num_iterations=num_iterations,
#                         ent_coef=ent_coef,
#                         learning_rate=learning_rate,
#                         num_minibatches=num_minibatches,
#                     )

#                     # agent 2: cow -> hunter_agent project
#                     train_and_log_for_task(
#                         agent_id=2,
#                         base_task_name="eat_cow",
#                         weights=COW_WEIGHTS,
#                         num_episodes=NUM_EPISODES_PER_AGENT,
#                         num_iterations=num_iterations,
#                         ent_coef=ent_coef,
#                         learning_rate=learning_rate,
#                         num_minibatches=num_minibatches,
#                     )

#     print("\nAll sweep configs finished (no trajectories/videos saved).")


"""
Gathering trajectories 
"""
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


def make_empty_weights():
    return jnp.zeros(22, dtype=jnp.float32)


WOOD_WEIGHTS  = make_empty_weights().at[0].set(50.0)
COW_WEIGHTS   = make_empty_weights().at[2].set(50.0)
DRINK_WEIGHTS = make_empty_weights().at[4].set(50.0)

# Map base_task_name -> W&B project name
AGENT_PROJECTS = {
    "collect_drink": "water_gather_agent",
    "collect_wood": "carpenter_agent",
    "eat_cow": "hunter_agent",
}


def make_metacontroller_for_task(
    achievement_weights,
    num_players: int = 2,
    num_iterations: int = 400,
    ent_coef: float = 0.02,
    learning_rate: float = 2.5e-4,
    num_minibatches: int = 16,
    wandb_project: str = "",
    wandb_run_name: str | None = None,
) -> ClassicMetaController:
    """
    Create a ClassicMetaController configured for a specific task via achievement_weights.
    """
    env_params = EnvParams(
        max_timesteps=300,
        achievement_weights=achievement_weights,
    )

    static_params = StaticEnvParams(num_players=num_players)

    print("PROJECT: ", wandb_project)

    metacontroller = ClassicMetaController(
        env_params=env_params,
        static_parameters=static_params,
        num_envs=200,
        num_steps=50,
        num_iterations=num_iterations,
        num_minibatches=num_minibatches,
        fixed_timesteps=False,
        learning_rate=learning_rate,
        anneal_lr=False,
        update_epochs=10,
        max_grad_norm=1.0,
        ent_coef=ent_coef,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        aux_coef=0.0,
    )

    metacontroller.rng = jax.random.PRNGKey(4253)
    return metacontroller


def _make_cfg_suffix(
    num_iterations: int,
    ent_coef: float,
    learning_rate: float,
    num_minibatches: int,
) -> str:
    """
    Create a filesystem-friendly suffix representing the hyperparameters.
    Example: it200_ent0p02_lr2p5e-04_mb16
    """
    def f(x: float) -> str:
        return str(x).replace(".", "p")

    return (
        f"it{num_iterations}_"
        f"ent{f(ent_coef)}_"
        f"lr{f(learning_rate)}_"
        f"mb{num_minibatches}"
    )


def train_specialist(
    task_name: str,
    weights: jnp.ndarray,
    num_iterations: int,
    ent_coef: float,
    learning_rate: float,
    num_minibatches: int = 16,
    cfg_suffix: str = "",
    wandb_project: str = "",
) -> Tuple[ClassicMetaController, Any]:
    """
    Train one specialist metacontroller for a given task and return it with the trained params.
    """
    full_task_name = f"{task_name}_{cfg_suffix}" if cfg_suffix else task_name
    print(f"\n===== Training specialist for task: {full_task_name} =====")
    print(
        f"    num_iterations={num_iterations}, ent_coef={ent_coef}, "
        f"learning_rate={learning_rate}, num_minibatches={num_minibatches}"
    )

    # Use full_task_name as W&B run name
    wandb_run_name = full_task_name

    metacontroller = make_metacontroller_for_task(
        achievement_weights=weights,
        num_players=2,
        num_iterations=num_iterations,
        ent_coef=ent_coef,
        learning_rate=learning_rate,
        num_minibatches=num_minibatches,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
    )

    # Train: returns (agent_params, aux_params)
    (params, aux_params), opt_state, log = metacontroller.train()

    # Save just the policy params
    params_fname = f"params_ippo_{full_task_name}.p"
    with open(params_fname, "wb") as f:
        pickle.dump(params, f)
    print(f"[{full_task_name}] Saved trained params to {params_fname}")

    return metacontroller, params


def collect_episodes_for_specialist(
    agent_id: int,
    task_name: str,
    metacontroller: ClassicMetaController,
    params: Any,
    num_episodes: int,
) -> List[Dict[str, Any]]:
    """
    Roll out num_episodes episodes with the trained specialist and
    return a list of trajectory dicts.
    """
    demos: List[Dict[str, Any]] = []
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

    print(f"[{task_name}] Collected {len(demos)} episodes.")
    return demos


def save_episode_videos(states, task_name: str, fps: float = 15.0):
    """
    Save states from one episode to one MP4 per player.
    """
    if len(states) == 0:
        print(f"[{task_name}] No states to render, skipping video.")
        return

    # Determine num_players from the state
    first_state = states[0]
    num_players = first_state.player_position.shape[0]

    tile_size = 64

    # Render a single frame to get frame size
    first_img = render_craftax_pixels(first_state, tile_size, num_players, 0)
    first_frame = np.asarray(first_img, dtype=np.uint8)[..., ::-1]
    H, W = first_frame.shape[:2]
    frame_size = (W, H)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    for player_idx in range(num_players):
        video_name = f"{task_name}_agent{player_idx}_last_episode.mp4"
        print(f"[{task_name}] Writing video: {video_name}")

        out = cv2.VideoWriter(video_name, fourcc, fps, frame_size)

        for s in states:
            img = render_craftax_pixels(s, tile_size, num_players, player_idx)
            frame = np.asarray(img, dtype=np.uint8)[..., ::-1]

            if frame.shape[:2] != (H, W):
                frame = cv2.resize(frame, frame_size, interpolation=cv2.INTER_NEAREST)

            out.write(frame)

        out.release()

    print(f"[{task_name}] Saved videos for {num_players} agents.")


def save_task_pickle(task_name: str, demos: List[Dict[str, Any]]) -> str:
    """
    Save the list of episode-level demos (trajectories) for one task into a pickle file.
    This is per-agent, per-hyperparam.
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
    Optional: take multiple per-task demo pickles and convert them into
    a single flat list of [agent_id, state, action, logits, reward].
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


def train_and_save_for_task(
    agent_id: int,
    base_task_name: str,
    weights: jnp.ndarray,
    num_episodes: int,
    num_iterations: int,
    ent_coef: float,
    learning_rate: float,
    num_minibatches: int = 16,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Train a specialist with the given hyperparams, collect num_episodes demos,
    save them in a per-agent pickle, and save a video of the last episode.
    """
    cfg_suffix = _make_cfg_suffix(
        num_iterations,
        ent_coef,
        learning_rate,
        num_minibatches,
    )
    full_task_name = f"{base_task_name}_{cfg_suffix}"

    # Choose W&B project based on the task
    wandb_project = AGENT_PROJECTS.get(base_task_name, "MARLCraftax")
    print("wandb_project: ", wandb_project)

    # 1) Train specialist
    mc, params = train_specialist(
        task_name=base_task_name,
        weights=weights,
        num_iterations=num_iterations,
        ent_coef=ent_coef,
        learning_rate=learning_rate,
        num_minibatches=num_minibatches,
        cfg_suffix=cfg_suffix,
        wandb_project=wandb_project,
    )

    # 2) Collect demos (these are the trajectories)
    demos = collect_episodes_for_specialist(
        agent_id=agent_id,
        task_name=full_task_name,
        metacontroller=mc,
        params=params,
        num_episodes=num_episodes,
    )

    # 3) Save per-task (per-agent) pickle of trajectories
    demos_path = save_task_pickle(full_task_name, demos)

    # 4) Save video from last episode (if any)
    if len(demos) > 0:
        last_states = demos[-1]["states"]
        save_episode_videos(last_states, full_task_name, fps=15.0)

    return demos, demos_path


if __name__ == "__main__":
    # Number of episodes per specialist to generate
    NUM_EPISODES_PER_AGENT = 100

    all_demos: List[Dict[str, Any]] = []
    task_pickle_paths: List[str] = []

    # -----------------------
    # Fixed hyperparams per agent
    # -----------------------

    # Wood / carpenter_agent
    wood_demos, wood_path = train_and_save_for_task(
        agent_id=0,
        base_task_name="collect_wood",
        weights=WOOD_WEIGHTS,
        num_episodes=NUM_EPISODES_PER_AGENT,
        num_iterations=200,
        ent_coef=0.02,
        learning_rate=2.5e-4,
        num_minibatches=16,
    )
    all_demos.extend(wood_demos)
    task_pickle_paths.append(wood_path)

    # Water gatherer / water_gather_agent
    drink_demos, drink_path = train_and_save_for_task(
        agent_id=1,
        base_task_name="collect_drink",
        weights=DRINK_WEIGHTS,
        num_episodes=NUM_EPISODES_PER_AGENT,
        num_iterations=200,
        ent_coef=0.01,
        learning_rate=1.25e-4,
        num_minibatches=10,
    )
    all_demos.extend(drink_demos)
    task_pickle_paths.append(drink_path)

    # Hunter / hunter_agent
    cow_demos, cow_path = train_and_save_for_task(
        agent_id=2,
        base_task_name="eat_cow",
        weights=COW_WEIGHTS,
        num_episodes=NUM_EPISODES_PER_AGENT,
        num_iterations=200,
        ent_coef=0.01,
        learning_rate=1.25e-4,
        num_minibatches=10,
    )
    all_demos.extend(cow_demos)
    task_pickle_paths.append(cow_path)

    # 5) Save one big pickle with *all* episode-level demos from all agents
    output_path = "irl_demonstrations_specialists_all_tasks_fixed.pickle"
    with open(output_path, "wb") as f:
        pickle.dump(all_demos, f)
    print(f"\nSaved {len(all_demos)} demos to {output_path}")
    print("Agent mapping: 0=wood, 1=drink, 2=cow")

    # 6) (Optional) also create a flat transition dataset across all tasks
    #     -> one big list of [agent_id, state, action, logits, reward]
    combine_task_pickles_to_flat(
        task_pickle_paths,
        output_path="combined_flat_dataset_fixed.pickle",
    )
