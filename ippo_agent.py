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
Gathering trajectories ****************************************************************************
"""
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
#     task_id: int = 0
# ) -> ClassicMetaController:
#     """
#     Create a ClassicMetaController configured for a specific task via achievement_weights.
#     """
#     env_params = EnvParams(
#         max_timesteps=300,
#         achievement_weights=achievement_weights,
#         task_id=task_id,
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
#         task_id=task_id,
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
#     Example: it200_ent0p02_lr2p5e-04_mb16
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
#     task_id: int = 0
# ) -> Tuple[ClassicMetaController, Any]:
#     """
#     Train one specialist metacontroller for a given task and return it with the trained params.
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
#         task_id= task_id
#     )

#     # Train: returns (agent_params, aux_params)
#     (params, aux_params), opt_state, log = metacontroller.train()

#     # Save just the policy params
#     params_fname = f"params_ippo_{full_task_name}.p"
#     with open(params_fname, "wb") as f:
#         pickle.dump(params, f)
#     print(f"[{full_task_name}] Saved trained params to {params_fname}")

#     return metacontroller, params


# def collect_episodes_for_specialist(
#     agent_id: int,
#     task_name: str,
#     metacontroller: ClassicMetaController,
#     params: Any,
#     num_episodes: int,
# ) -> List[Dict[str, Any]]:
#     """
#     Roll out num_episodes episodes with the trained specialist and
#     return a list of trajectory dicts.
#     """
#     demos: List[Dict[str, Any]] = []
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
#     Save the list of episode-level demos (trajectories) for one task into a pickle file.
#     This is per-agent, per-hyperparam.
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
#     """
#     Optional: take multiple per-task demo pickles and convert them into
#     a single flat list of [agent_id, state, action, logits, reward].
#     """
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
#     base_task_name: str,
#     weights: jnp.ndarray,
#     num_episodes: int,
#     num_iterations: int,
#     ent_coef: float,
#     learning_rate: float,
#     num_minibatches: int = 16,
# ) -> Tuple[List[Dict[str, Any]], str]:
#     """
#     Train a specialist with the given hyperparams, collect num_episodes demos,
#     save them in a per-agent pickle, and save a video of the last episode.
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

#     # 1) Train specialist
#     mc, params = train_specialist(
#         task_name=base_task_name,
#         weights=weights,
#         num_iterations=num_iterations,
#         ent_coef=ent_coef,
#         learning_rate=learning_rate,
#         num_minibatches=num_minibatches,
#         cfg_suffix=cfg_suffix,
#         wandb_project=wandb_project,
#         task_id=agent_id,
#     )

#     # 2) Collect demos (these are the trajectories)
#     demos = collect_episodes_for_specialist(
#         agent_id=agent_id,
#         task_name=full_task_name,
#         metacontroller=mc,
#         params=params,
#         num_episodes=num_episodes,
#     )

#     # 3) Save per-task (per-agent) pickle of trajectories
#     demos_path = save_task_pickle(full_task_name, demos)

#     # 4) Save video from last episode (if any)
#     if len(demos) > 0:
#         last_states = demos[-1]["states"]
#         save_episode_videos(last_states, full_task_name, fps=15.0)

#     return demos, demos_path


# if __name__ == "__main__":
#     # Number of episodes per specialist to generate
#     NUM_EPISODES_PER_AGENT = 100

#     all_demos: List[Dict[str, Any]] = []
#     task_pickle_paths: List[str] = []

#     # -----------------------
#     # Fixed hyperparams per agent
#     # -----------------------

#     # Wood / carpenter_agent
#     wood_demos, wood_path = train_and_save_for_task(
#         agent_id=0,
#         base_task_name="collect_wood",
#         weights=WOOD_WEIGHTS,
#         num_episodes=NUM_EPISODES_PER_AGENT,
#         num_iterations=400,
#         ent_coef=0.02,
#         learning_rate=2.5e-4,
#         num_minibatches=16,
#     )
#     all_demos.extend(wood_demos)
#     task_pickle_paths.append(wood_path)

#     # # Water gatherer / water_gather_agent
#     # drink_demos, drink_path = train_and_save_for_task(
#     #     agent_id=1,
#     #     base_task_name="collect_drink",
#     #     weights=DRINK_WEIGHTS,
#     #     num_episodes=NUM_EPISODES_PER_AGENT,
#     #     num_iterations=200,
#     #     ent_coef=0.01,
#     #     learning_rate=1.25e-4,
#     #     num_minibatches=10,
#     # )
#     # all_demos.extend(drink_demos)
#     # task_pickle_paths.append(drink_path)

#     # # Hunter / hunter_agent
#     # cow_demos, cow_path = train_and_save_for_task(
#     #     agent_id=2,
#     #     base_task_name="eat_cow",
#     #     weights=COW_WEIGHTS,
#     #     num_episodes=NUM_EPISODES_PER_AGENT,
#     #     num_iterations=200,
#     #     ent_coef=0.01,
#     #     learning_rate=1.25e-4,
#     #     num_minibatches=10,
#     # )
#     # all_demos.extend(cow_demos)
#     # task_pickle_paths.append(cow_path)

#     # # 5) Save one big pickle with *all* episode-level demos from all agents
#     # output_path = "irl_demonstrations_specialists_all_tasks_fixed.pickle"
#     # with open(output_path, "wb") as f:
#     #     pickle.dump(all_demos, f)
#     # print(f"\nSaved {len(all_demos)} demos to {output_path}")
#     # print("Agent mapping: 0=wood, 1=drink, 2=cow")

#     # # 6) (Optional) also create a flat transition dataset across all tasks
#     # #     -> one big list of [agent_id, state, action, logits, reward]
#     # combine_task_pickles_to_flat(
#     #     task_pickle_paths,
#     #     output_path="combined_flat_dataset_fixed.pickle",
#     # )


"""
Hunter parameter sweep ****************************************************************************
"""

# #!/usr/bin/env python3
# from typing import Any, Tuple

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
#     Create a ClassicMetaController configured for a specific task via achievement_weights.
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
#     Example: it200_ent0p02_lr2p5e-04_mb16
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
#     No saving of params / trajectories / videos.
#     """
#     full_task_name = f"{task_name}_{cfg_suffix}" if cfg_suffix else task_name
#     print(f"\n===== Training specialist for task: {full_task_name} =====")
#     print(
#         f"    num_iterations={num_iterations}, ent_coef={ent_coef}, "
#         f"learning_rate={learning_rate}, num_minibatches={num_minibatches}"
#     )

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

#     print(f"[{full_task_name}] Training complete.")
#     return metacontroller, params


# def train_for_task(
#     agent_id: int,
#     base_task_name: str,
#     weights: jnp.ndarray,
#     num_iterations: int,
#     ent_coef: float,
#     learning_rate: float,
#     num_minibatches: int = 16,
# ) -> None:
#     """
#     Convenience wrapper:
#     - builds config suffix
#     - picks W&B project
#     - trains the specialist
#     No saving, no rollouts.
#     """
#     cfg_suffix = _make_cfg_suffix(
#         num_iterations,
#         ent_coef,
#         learning_rate,
#         num_minibatches,
#     )
#     full_task_name = f"{base_task_name}_{cfg_suffix}"

#     wandb_project = AGENT_PROJECTS.get(base_task_name, "MARLCraftax")
#     print("wandb_project: ", wandb_project)

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

#     print(f"[{full_task_name}] Done (no pickles/videos).")


# if __name__ == "__main__":
#     # -----------------------
#     # 1) Fixed configs for wood + water
#     # -----------------------

#     # Wood / carpenter_agent
#     train_for_task(
#         agent_id=0,
#         base_task_name="collect_wood",
#         weights=WOOD_WEIGHTS,
#         num_iterations=200,
#         ent_coef=0.02,
#         learning_rate=2.5e-4,
#         num_minibatches=16,
#     )

#     # Water gatherer / water_gather_agent
#     train_for_task(
#         agent_id=1,
#         base_task_name="collect_drink",
#         weights=DRINK_WEIGHTS,
#         num_iterations=200,
#         ent_coef=0.01,
#         learning_rate=1.25e-4,
#         num_minibatches=10,
#     )

#     # -----------------------
#     # 2) Parameter sweep ONLY for hunter
#     # -----------------------
#     HUNTER_SWEEP_NUM_ITERATIONS  = [200, 400, 600, 800]
#     HUNTER_SWEEP_ENT_COEFS       = [0.01, 0.02, 0.04]
#     HUNTER_SWEEP_LEARNING_RATES  = [1.25e-4, 2.5e-4, 5e-4, 2.5e-3]
#     HUNTER_SWEEP_NUM_MINIBATCHES = [10, 16]

#     for num_iterations in HUNTER_SWEEP_NUM_ITERATIONS:
#         for ent_coef in HUNTER_SWEEP_ENT_COEFS:
#             for learning_rate in HUNTER_SWEEP_LEARNING_RATES:
#                 for num_minibatches in HUNTER_SWEEP_NUM_MINIBATCHES:
#                     print("\n==============================================")
#                     print("Hunter sweep config:")
#                     print(f"  num_iterations={num_iterations}")
#                     print(f"  ent_coef={ent_coef}")
#                     print(f"  learning_rate={learning_rate}")
#                     print(f"  num_minibatches={num_minibatches}")
#                     print("==============================================\n")

#                     train_for_task(
#                         agent_id=2,
#                         base_task_name="eat_cow",
#                         weights=COW_WEIGHTS,
#                         num_iterations=num_iterations,
#                         ent_coef=ent_coef,
#                         learning_rate=learning_rate,
#                         num_minibatches=num_minibatches,
#                     )

#     print("\nAll training (fixed + hunter sweep) finished — no pickles, no videos.")


"""
Collecting stone agent parameter sweeping *****************************************************************************
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


# # Only stone collecting now (index 9)
# STONE_WEIGHTS = make_empty_weights().at[9].set(50.0)

# # Map base_task_name -> W&B project name (only stone)
# AGENT_PROJECTS = {
#     "collect_stone": "stone_gather_agent",
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
#     task_name: str=""
# ) -> ClassicMetaController:
#     """
#     Create a ClassicMetaController configured for a specific task via achievement_weights,
#     with hyperparameters exposed for sweeps + W&B project/run naming.
#     """
#     env_params = EnvParams(
#         max_timesteps=300,
#         achievement_weights=achievement_weights,
#         task_name=task_name,
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
#         task_name=task_name,
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

#     # Quadruple nested sweep – ONLY stone collecting
#     for num_iterations in SWEEP_NUM_ITERATIONS:
#         for ent_coef in SWEEP_ENT_COEFS:
#             for learning_rate in SWEEP_LEARNING_RATES:
#                 for num_minibatches in SWEEP_NUM_MINIBATCHES:
#                     print("\n==============================================")
#                     print("Starting sweep config (stone):")
#                     print(f"  num_iterations={num_iterations}")
#                     print(f"  ent_coef={ent_coef}")
#                     print(f"  learning_rate={learning_rate}")
#                     print(f"  num_minibatches={num_minibatches}")
#                     print("==============================================\n")

#                     # agent 0: stone -> stone_gather_agent project
#                     train_and_log_for_task(
#                         agent_id=0,
#                         base_task_name="collect_stone",
#                         weights=STONE_WEIGHTS,
#                         num_episodes=NUM_EPISODES_PER_AGENT,
#                         num_iterations=num_iterations,
#                         ent_coef=ent_coef,
#                         learning_rate=learning_rate,
#                         num_minibatches=num_minibatches,
#                     )

#     print("\nAll sweep configs for stone collecting finished (no trajectories/videos saved).")



"""
Final gathering of trajectories ****************************************************************************
"""


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


# # Achievement indices:
# #   0: COLLECT_WOOD
# #   4: COLLECT_DRINK (water)
# #   9: COLLECT_STONE
# WOOD_WEIGHTS   = make_empty_weights().at[0].set(50.0)
# DRINK_WEIGHTS  = make_empty_weights().at[4].set(50.0)
# STONE_WEIGHTS  = make_empty_weights().at[9].set(50.0)

# # Map base_task_name -> W&B project name
# AGENT_PROJECTS = {
#     "collect_wood":  "carpenter_agent",
#     "collect_stone": "stone_gather_agent",
#     "collect_drink": "water_gather_agent",
# }


# def make_metacontroller_for_task(
#     achievement_weights,
#     num_players: int = 3,
#     num_iterations: int = 400,
#     ent_coef: float = 0.02,
#     learning_rate: float = 2.5e-4,
#     num_minibatches: int = 16,
#     wandb_project: str = "",
#     wandb_run_name: str | None = None,
#     task_id: int = 0,
# ) -> ClassicMetaController:
#     """
#     Create a ClassicMetaController configured for a specific task via achievement_weights.
#     task_id is used by the env (e.g., 0=wood, 1=stone, 2=water).
#     """
#     env_params = EnvParams(
#         max_timesteps=300,
#         achievement_weights=achievement_weights,
#         task_id=task_id,
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
#         task_id=task_id,
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
#     Example: it200_ent0p02_lr2p5e-04_mb16
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
#     task_id: int = 0,
# ) -> Tuple[ClassicMetaController, Any]:
#     """
#     Train one specialist metacontroller for a given task and return it with the trained params.
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
#         task_id=task_id,
#     )

#     # Train: returns (agent_params, aux_params)
#     (params, aux_params), opt_state, log = metacontroller.train()

#     # Save just the policy params
#     params_fname = f"params_ippo_{full_task_name}.p"
#     with open(params_fname, "wb") as f:
#         pickle.dump(params, f)
#     print(f"[{full_task_name}] Saved trained params to {params_fname}")

#     return metacontroller, params


# def collect_episodes_for_specialist(
#     agent_id: int,
#     task_name: str,
#     metacontroller: ClassicMetaController,
#     params: Any,
#     num_episodes: int,
# ) -> List[Dict[str, Any]]:
#     """
#     Roll out num_episodes episodes with the trained specialist and
#     return a list of trajectory dicts.
#     """
#     demos: List[Dict[str, Any]] = []
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
#     Save the list of episode-level demos (trajectories) for one task into a pickle file.
#     This is per-agent, per-hyperparam.
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
#     """
#     Optional: take multiple per-task demo pickles and convert them into
#     a single flat list of [agent_id, state, action, logits, reward].
#     """
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

#     print(f"Saved {len(combined)} [agent_id, state, action, logits, reward] records to {output_path}")


# def train_and_save_for_task(
#     agent_id: int,
#     base_task_name: str,
#     weights: jnp.ndarray,
#     num_episodes: int,
#     num_iterations: int,
#     ent_coef: float,
#     learning_rate: float,
#     num_minibatches: int = 16,
# ) -> Tuple[List[Dict[str, Any]], str]:
#     """
#     Train a specialist with the given hyperparams, collect num_episodes demos,
#     save them in a per-agent pickle, and save a video of the last episode.
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

#     # 1) Train specialist
#     mc, params = train_specialist(
#         task_name=base_task_name,
#         weights=weights,
#         num_iterations=num_iterations,
#         ent_coef=ent_coef,
#         learning_rate=learning_rate,
#         num_minibatches=num_minibatches,
#         cfg_suffix=cfg_suffix,
#         wandb_project=wandb_project,
#         task_id=agent_id,  # 0=wood, 1=stone, 2=water
#     )

#     # 2) Collect demos (these are the trajectories)
#     demos = collect_episodes_for_specialist(
#         agent_id=agent_id,
#         task_name=full_task_name,
#         metacontroller=mc,
#         params=params,
#         num_episodes=num_episodes,
#     )

#     # 3) Save per-task (per-agent) pickle of trajectories
#     demos_path = save_task_pickle(full_task_name, demos)

#     # 4) Save video from last episode (if any)
#     if len(demos) > 0:
#         last_states = demos[-1]["states"]
#         save_episode_videos(last_states, full_task_name, fps=15.0)

#     return demos, demos_path


# if __name__ == "__main__":
#     # Number of episodes per specialist to generate
#     NUM_EPISODES_PER_AGENT = 100

#     all_demos: List[Dict[str, Any]] = []
#     task_pickle_paths: List[str] = []

#     # -----------------------
#     # Fixed hyperparams per agent
#     # -----------------------

#     # Wood / carpenter_agent (task_id = 0)
#     wood_demos, wood_path = train_and_save_for_task(
#         agent_id=0,
#         base_task_name="collect_wood",
#         weights=WOOD_WEIGHTS,
#         num_episodes=NUM_EPISODES_PER_AGENT,
#         num_iterations=200,
#         ent_coef=0.02,
#         learning_rate=2.5e-4,
#         num_minibatches=16,
#     )
#     all_demos.extend(wood_demos)
#     task_pickle_paths.append(wood_path)

#     # Stone / stone_gather_agent (task_id = 1)
#     stone_demos, stone_path = train_and_save_for_task(
#         agent_id=1,
#         base_task_name="collect_stone",
#         weights=STONE_WEIGHTS,
#         num_episodes=NUM_EPISODES_PER_AGENT,
#         num_iterations=200,      # same as wood
#         ent_coef=0.02,           # same as wood
#         learning_rate=2.5e-4,    # same as wood
#         num_minibatches=16,      # same as wood
#     )
#     all_demos.extend(stone_demos)
#     task_pickle_paths.append(stone_path)

#     # Water gatherer / water_gather_agent (task_id = 2)
#     drink_demos, drink_path = train_and_save_for_task(
#         agent_id=2,
#         base_task_name="collect_drink",
#         weights=DRINK_WEIGHTS,
#         num_episodes=NUM_EPISODES_PER_AGENT,
#         num_iterations=200,
#         ent_coef=0.01,
#         learning_rate=1.25e-4,
#         num_minibatches=10,
#     )
#     all_demos.extend(drink_demos)
#     task_pickle_paths.append(drink_path)

#     # Optional: save combined datasets
#     output_path = "irl_demonstrations_wood_stone_water_fixed.pickle"
#     with open(output_path, "wb") as f:
#         pickle.dump(all_demos, f)
#     print(f"\nSaved {len(all_demos)} demos to {output_path}")
#     print("Agent mapping: 0=wood, 1=stone, 2=water")
    
#     combine_task_pickles_to_flat(
#         task_pickle_paths,
#         output_path="combined_flat_dataset_wood_stone_water_fixed.pickle",
#     )


# """
# Multi agent approach - one enviornment ***************************************************************************
# """

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


# Achievement indices:
#   0: COLLECT_WOOD
WOOD_WEIGHTS = make_empty_weights().at[0].set(50.0)

# Map base_task_name -> W&B project name
AGENT_PROJECTS = {
    "collect_wood": "carpenter_agent",
}


def make_metacontroller_for_task(
    achievement_weights,
    num_players: int = 3,
    num_iterations: int = 2,
    ent_coef: float = 0.02,
    learning_rate: float = 2.5e-4,
    num_minibatches: int = 16,
    wandb_project: str = "",
    wandb_run_name: str | None = None,
    task_id: int = 0,
) -> ClassicMetaController:
    """
    Create a ClassicMetaController configured for a specific task via achievement_weights.
    task_id can be used by the env (e.g., 0=wood).
    """
    env_params = EnvParams(
        max_timesteps=2,
        achievement_weights=achievement_weights,
        task_id=task_id,
    )

    static_params = StaticEnvParams(num_players=num_players)

    print("PROJECT: ", wandb_project)

    metacontroller = ClassicMetaController(
        env_params=env_params,
        static_parameters=static_params,
        num_envs=200,
        num_steps=2,
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
        task_id=task_id,
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
    task_id: int = 0,
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
        num_players=3,
        num_iterations=num_iterations,
        ent_coef=ent_coef,
        learning_rate=learning_rate,
        num_minibatches=num_minibatches,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
        task_id=task_id,
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
    task_name: str,
    metacontroller: ClassicMetaController,
    params: Any,
    num_episodes: int,
) -> List[Dict[str, Any]]:
    """
    Roll out num_episodes episodes with the trained specialist and
    return a list of *per-agent* trajectory dicts.

    For each episode and each agent_index p, we store:
      states  : list of EnvState, length T (global environment state)
      actions : (T,) for that agent only
      logits  : (T, ...) for that agent only
      rewards : (T,) for that agent only
    """
    demos: List[Dict[str, Any]] = []
    roles = ["wood", "diamond", "stone"]

    for ep in range(num_episodes):
        print(f"[{task_name}] Running episode {ep+1}/{num_episodes}")
        states, actions, logits, rewards = metacontroller.run_one_episode(params)
        # Typically:
        #   states  : list length T of EnvState
        #   actions : list length T of (num_players,) arrays  OR already (T, num_players)
        #   logits  : same idea but with extra dims
        #   rewards : list length T of (num_players,) arrays  OR already (T, num_players)

        # --- Normalize actions / logits / rewards to arrays so we can do [:, agent_index] ---

        if isinstance(actions, list):
            actions_arr = jnp.stack(actions, axis=0)   # (T, num_players)
        else:
            actions_arr = actions                      # assume already (T, num_players)

        if isinstance(rewards, list):
            rewards_arr = jnp.stack(rewards, axis=0)   # (T, num_players)
        else:
            rewards_arr = rewards

        if isinstance(logits, list):
            # If logits is per-timestep list of per-player arrays
            logits_arr = jnp.stack(logits, axis=0)     # (T, num_players, ...)
        else:
            logits_arr = logits

        # Infer num_players from the first state
        first_state = states[0]
        num_players = first_state.player_position.shape[0]
        ep_roles = roles[:num_players]

        T = len(states)

        for agent_index in range(num_players):
            demo = {
                "task_name": task_name,
                "episode_idx": ep,
                "agent_index": agent_index,          # 0=wood, 1=diamond, 2=stone
                "role": ep_roles[agent_index],       # "wood"/"diamond"/"stone"
                "states": states,                    # full global states, length T
                "actions": actions_arr[:, agent_index],   # (T,)
                "logits": logits_arr[:, agent_index],     # (T, ...)
                "rewards": rewards_arr[:, agent_index],   # (T,)
            }
            demos.append(demo)

    print(
        f"[{task_name}] Collected {len(demos)} per-agent demos "
        f"({num_episodes} episodes × {num_players} players)."
    )
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
    Save the list of per-agent, per-episode demos into a pickle file.
    """
    filename = f"demos_{task_name}.pickle"
    with open(filename, "wb") as f:
        pickle.dump(demos, f)
    print(f"[{task_name}] Saved {len(demos)} demos to {filename}")
    return filename


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
    save them in a per-task pickle, and save a video of the last episode.

    agent_id is used only as task_id for the env (e.g. 0=wood).
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
        task_id=agent_id,  # 0 = wood
    )

    # 2) Collect per-agent demos
    demos = collect_episodes_for_specialist(
        task_name=full_task_name,
        metacontroller=mc,
        params=params,
        num_episodes=num_episodes,
    )

    # 3) Save per-task pickle of trajectories
    demos_path = save_task_pickle(full_task_name, demos)

    # 4) Save video from last episode (if any) – still multi-agent video
    if len(demos) > 0:
        # All demos share the same underlying states object for given episode,
        # so we can just use the states from the last demo.
        last_states = demos[-1]["states"]
        save_episode_videos(last_states, full_task_name, fps=15.0)

    return demos, demos_path


if __name__ == "__main__":
    # Number of episodes per specialist to generate
    NUM_EPISODES_PER_AGENT = 2

    # Only wood / carpenter_agent (task_id = 0)
    wood_demos, wood_path = train_and_save_for_task(
        agent_id=0,
        base_task_name="collect_wood",
        weights=WOOD_WEIGHTS,
        num_episodes=NUM_EPISODES_PER_AGENT,
        num_iterations=2,
        ent_coef=0.02,
        learning_rate=2.5e-4,
        num_minibatches=15,
    )

    print(f"\nSaved wood+diamond+stone per-agent demos pickle to: {wood_path}")
    print("Mapping: agent_index 0=wood, 1=diamond, 2=stone.")


"""
Code for converting pickle -> hdf5 *************************************************************************
"""

# #!/usr/bin/env python3
# """
# pickle_to_hdf5_episode_batched.py

# Reorganize per-agent demos into per-episode, player-batched HDF5.

# Input (pickle):
#     demos: List[Dict], each dict is ONE (episode, agent) trajectory:
#         {
#           "task_name": str,
#           "episode_idx": int,
#           "agent_index": int,   # 0..num_players-1
#           "role": str,
#           "states": [EnvState_0, EnvState_1, ..., EnvState_{T-1}],
#           "actions": (T,),
#           "rewards": (T,),
#           "logits": ...
#         }

# Output (HDF5):
#     One group per episode: /episode_0, /episode_1, ...
#     Each group has:

#       attrs:
#         - task_name: str
#         - episode_idx: int
#         - num_players: int
#         - roles: (num_players,) array of strings

#       datasets:
#         - map              : (T, H, W)
#         - player_position  : (T, num_players, pos_dim)
#         - player_direction : (T, num_players, dir_dim)
#         - timestep         : (T,)
#         - actions          : (T, num_players)
#         - rewards          : (T, num_players)
# """

# import pickle
# from collections import defaultdict
# from typing import Any, Dict, List

# import numpy as np
# import h5py

# # Optional: helps unpickling EnvState, but script works as long as pickling environment matches.
# try:
#     from craftax.craftax_classic.envs.craftax_state import EnvState  # noqa: F401
# except Exception:
#     EnvState = None


# # ---------------------------------------------------------------------
# # STEP 1: Convert per-agent demos -> per-episode, player-batched HDF5
# # ---------------------------------------------------------------------

# def pickle_to_episode_batched_hdf5(
#     pickle_path: str,
#     hdf5_path: str,
# ) -> None:
#     with open(pickle_path, "rb") as f:
#         demos = pickle.load(f)

#     if not isinstance(demos, list):
#         raise TypeError(f"Expected list of demos in pickle, got {type(demos)}")

#     episodes: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
#     for d in demos:
#         ep_idx = int(d.get("episode_idx", -1))
#         episodes[ep_idx].append(d)

#     with h5py.File(hdf5_path, "w") as h5f:
#         for ep_idx in sorted(episodes.keys()):
#             ep_demos = episodes[ep_idx]
#             if not ep_demos:
#                 continue

#             # sort by agent_index so 0,1,2 map cleanly
#             ep_demos = sorted(ep_demos, key=lambda d: int(d.get("agent_index", -1)))

#             num_players = len(ep_demos)
#             first_demo = ep_demos[0]
#             task_name = first_demo.get("task_name", "")
#             roles = [d.get("role", "") for d in ep_demos]

#             states = first_demo.get("states", [])
#             if not states:
#                 continue

#             T_states = len(states)

#             first_state = states[0]
#             full_player_pos0 = np.asarray(getattr(first_state, "player_position", None))
#             if full_player_pos0 is None or full_player_pos0.size == 0:
#                 pos_dim = 2
#             else:
#                 pos_dim = full_player_pos0.shape[-1]

#             full_player_dir0 = np.asarray(getattr(first_state, "player_direction", None))
#             if full_player_dir0 is None or full_player_dir0.size == 0:
#                 dir_dim = 1
#             else:
#                 dir_dim = 1 if full_player_dir0.ndim == 1 else full_player_dir0.shape[-1]

#             T = T_states
#             for d in ep_demos:
#                 actions_arr = np.asarray(d["actions"])
#                 rewards_arr = np.asarray(d["rewards"])
#                 T = min(T, actions_arr.shape[0], rewards_arr.shape[0])

#             maps_list = []
#             pos_arr = np.zeros((T, num_players, pos_dim), dtype=np.int32)
#             dir_arr = np.zeros((T, num_players, dir_dim), dtype=np.int32)
#             ts_arr = np.zeros((T,), dtype=np.int32)
#             actions_ep = np.zeros((T, num_players), dtype=np.int32)
#             rewards_ep = np.zeros((T, num_players), dtype=np.float32)

#             for t in range(T):
#                 s_t = states[t]

#                 full_map = getattr(s_t, "map", None)
#                 if full_map is not None:
#                     map_arr = np.asarray(full_map)
#                 else:
#                     map_arr = np.zeros((1, 1), dtype=np.int32)
#                 maps_list.append(map_arr)

#                 full_player_pos = np.asarray(
#                     getattr(s_t, "player_position", np.zeros((num_players, pos_dim), dtype=np.int32))
#                 )
#                 if full_player_pos.ndim == 1:
#                     full_player_pos = full_player_pos.reshape(num_players, -1)
#                 pos_arr[t, :, :] = full_player_pos[:, :pos_dim]

#                 full_player_dir = np.asarray(
#                     getattr(s_t, "player_direction", np.zeros((num_players, dir_dim), dtype=np.int32))
#                 )
#                 if full_player_dir.ndim == 1:
#                     full_player_dir = full_player_dir.reshape(num_players, 1)
#                 dir_arr[t, :, :] = full_player_dir[:, :dir_dim]

#                 timestep_val = getattr(
#                     s_t,
#                     "timestep",
#                     getattr(s_t, "time_step", t),
#                 )
#                 ts_arr[t] = int(timestep_val)

#             maps_arr = np.stack(maps_list, axis=0)  # (T, H, W)

#             for p, d in enumerate(ep_demos):
#                 act = np.asarray(d["actions"])[:T]
#                 rew = np.asarray(d["rewards"])[:T]
#                 actions_ep[:, p] = act
#                 rewards_ep[:, p] = rew

#             grp = h5f.create_group(f"episode_{ep_idx}")
#             grp.attrs["task_name"] = str(task_name)
#             grp.attrs["episode_idx"] = ep_idx
#             grp.attrs["num_players"] = num_players

#             dt = h5py.string_dtype(encoding="utf-8")
#             grp.create_dataset("roles", data=np.array(roles, dtype=object), dtype=dt)

#             grp.create_dataset("map", data=maps_arr)
#             grp.create_dataset("player_position", data=pos_arr)
#             grp.create_dataset("player_direction", data=dir_arr)
#             grp.create_dataset("timestep", data=ts_arr)
#             grp.create_dataset("actions", data=actions_ep)
#             grp.create_dataset("rewards", data=rewards_ep)


# # ---------------------------------------------------------------------
# # STEP 2: Inspect HDF5 - only print the element (record) per timestep
# # ---------------------------------------------------------------------

# def print_episode_batched_from_hdf5(
#     hdf5_path: str,
#     max_episodes: int = 10,
#     max_steps: int = 5,
# ) -> None:
#     """
#     For each episode and timestep, print a single dict:

#     {
#       "episode_idx": int,
#       "timestep": int,
#       "roles": [role_0, role_1, ...],
#       "state": {
#           "map": (H, W) array,
#           "player_position": (num_players, pos_dim),
#           "player_direction": (num_players, dir_dim),
#       },
#       "actions": (num_players,),
#       "rewards": (num_players,),
#     }

#     No extra text.
#     """
#     with h5py.File(hdf5_path, "r") as h5f:
#         ep_names = sorted(h5f.keys())

#         for ep_name in ep_names[:max_episodes]:
#             grp = h5f[ep_name]

#             ep_idx = int(grp.attrs.get("episode_idx", -1))
#             roles = [
#                 r.decode("utf-8") if isinstance(r, bytes) else str(r)
#                 for r in grp["roles"][...]
#             ]

#             maps_arr   = grp["map"][...]              # (T, H, W)
#             pos_arr    = grp["player_position"][...]  # (T, num_players, pos_dim)
#             dir_arr    = grp["player_direction"][...] # (T, num_players, dir_dim)
#             ts_arr     = grp["timestep"][...]         # (T,)
#             actions_ep = grp["actions"][...]          # (T, num_players)
#             rewards_ep = grp["rewards"][...]          # (T, num_players)

#             T = maps_arr.shape[0]
#             steps_to_print = min(T, max_steps)

#             for t in range(steps_to_print):
#                 record = {
#                     "episode_idx": ep_idx,
#                     "timestep": int(ts_arr[t]),
#                     "roles": roles,
#                     "state": {
#                         "map": maps_arr[t],             # (H, W)
#                         "player_position": pos_arr[t],  # (num_players, pos_dim)
#                         "player_direction": dir_arr[t], # (num_players, dir_dim)
#                     },
#                     "actions": actions_ep[t],           # (num_players,)
#                     "rewards": rewards_ep[t],           # (num_players,)
#                 }
#                 print(record)


# # ---------------------------------------------------------------------
# # MAIN
# # ---------------------------------------------------------------------

# if __name__ == "__main__":
#     # CHANGE THESE PATHS AS NEEDED
#     pickle_path = "demos_collect_wood_it500_ent0p02_lr0p00025_mb16.pickle"
#     hdf5_path   = "demos_collect_wood_it500_ent0p02_lr0p00025_mb16_episode_batched.h5"

#     # 1) Convert pickle -> episode-batched HDF5
#     pickle_to_episode_batched_hdf5(pickle_path, hdf5_path)

#     # 2) Print only the dict elements (no extra text)
#     print_episode_batched_from_hdf5(hdf5_path, max_episodes=10, max_steps=5)
