# # craftax_keyboard_viewer.py
# import sys, time
# import numpy as np
# import jax, jax.numpy as jnp
# import pygame

# # --- pick the import path that exists in your install ---
# try:
#     from craftax.craftax_classic.envs.craftax_pixels_env import CraftaxClassicPixelsEnv as Env
# except ImportError:
#     # Some versions name it differently; if you only have symbolic, see the note below
#     print("Trying the symbolic env")
#     from craftax.craftax_classic.envs.craftax_symbolic_env import CraftaxClassicSymbolicEnv as Env

# from craftax.craftax_classic.constants import Action

# # ------ config ------
# FPS = 15
# NUM_PLAYERS = 1        # one avatar to control from keyboard
# SEED = 0

# # Map keyboard -> Craftax action (feel free to change)
# KEYMAP = {
#     pygame.K_UP:    Action.UP.value,
#     pygame.K_DOWN:  Action.DOWN.value,
#     pygame.K_LEFT:  Action.LEFT.value,
#     pygame.K_RIGHT: Action.RIGHT.value,
#     pygame.K_SPACE: Action.DO.value,       # interact / attack / eat / drink
#     pygame.K_s:     Action.SLEEP.value,    # rest
#     pygame.K_n:     0,                     # NOOP (id 0)
# }

# def to_rgb(obs, info):
#     """Return HxWx3 uint8 for display."""
#     # Pixel env: obs is (H, W, 3/4). Keep first 3 channels.
#     if obs.ndim == 3 and obs.shape[-1] in (3, 4):
#         img = np.asarray(obs)
#         if img.dtype != np.uint8:
#             img = (img.astype(np.float32).clip(0, 1) * 255).astype(np.uint8)
#         return img[..., :3]
#     # Some envs put a render in info["rgb"]
#     if isinstance(info, dict) and "rgb" in info:
#         img = np.asarray(info["rgb"])
#         if img.ndim == 2:
#             img = np.repeat(img[..., None], 3, axis=-1)
#         return img[..., :3].astype(np.uint8)
#     # Fallback: make a tiny debug strip for symbolic observations
#     v = np.asarray(obs).reshape(-1).astype(np.float32)
#     H, W = 96, 96
#     img = np.zeros((H, W, 3), dtype=np.uint8)
#     if v.size:
#         v = (v - v.min()) / (v.max() - v.min() + 1e-6)
#         n = min(W, v.size)
#         img[H//3, :n, 0] = (v[:n] * 255).astype(np.uint8)
#     return img

# def main():
#     # --- env ---
#     env = Env()
#     params = env.default_params
#     env.static_env_params = env.static_env_params.replace(num_players=NUM_PLAYERS)

#     rng = jax.random.PRNGKey(SEED)
#     rng, key = jax.random.split(rng)
#     obs, state = env.reset(key, params)

#     # Build first frame & window
#     dummy_info = {}
#     frame = to_rgb(np.array(obs[0]), dummy_info)
#     H, W = frame.shape[:2]

#     pygame.init()
#     screen = pygame.display.set_mode((W, H))
#     pygame.display.set_caption("Craftax (Python viewer)")
#     clock = pygame.time.Clock()

#     running = True
#     while running:
#         # default action is NOOP; we read the latest keydown and convert
#         action = np.full((NUM_PLAYERS,), 0, dtype=np.int32)  # NOOP=0
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 running = False
#             elif event.type in (pygame.KEYDOWN, pygame.KEYUP):
#                 if event.type == pygame.KEYDOWN:
#                     if event.key == pygame.K_q:
#                         running = False
#                     elif event.key == pygame.K_r:
#                         # reset
#                         rng, key = jax.random.split(rng)
#                         obs, state = env.reset(key, params)
#                         continue
#                     elif event.key in KEYMAP:
#                         action[:] = KEYMAP[event.key]

#         # Step
#         rng, sk = jax.random.split(rng)
#         obs, state, reward, done, info = env.step(sk, state, jnp.array(action), params)

#         # Render first player's view (UI is already drawn in pixel frames)
#         frame = to_rgb(np.array(obs[0]), info)
#         surf = pygame.image.frombuffer(frame.tobytes(), (frame.shape[1], frame.shape[0]), "RGB")
#         screen.blit(surf, (0, 0))
#         pygame.display.flip()

#         # Reset on terminal (or press 'r')
#         if bool(jnp.any(done)):
#             rng, key = jax.random.split(rng)
#             obs, state = env.reset(key, params)

#         clock.tick(FPS)

#     pygame.quit()
#     sys.exit(0)

# if __name__ == "__main__":
#     main()

# run_craftax_pixels_random.py
import argparse
import os

import jax
import jax.numpy as jnp
import numpy as np
import imageio.v3 as iio

from craftax.craftax_classic.envs.craftax_pixels_env import CraftaxClassicPixelsEnv as Env
from craftax.craftax_classic.constants import Action


def get_noop_action():
    # Some builds use NOOP; others use SLEEP.
    return getattr(Action, "NOOP", Action.SLEEP).value


ALLOWED_ACTIONS = jnp.array([
    get_noop_action(),
    Action.LEFT.value,
    Action.RIGHT.value,
    Action.UP.value,
    Action.DOWN.value,
    Action.DO.value,  # "interact"
])


def extract_rgb_frame(obs):
    """
    Try to robustly extract an RGB frame from a pixels env observation.
    Expected formats:
      - obs is a dict with a 'rgb' (or similar) key of shape (players, H, W, 3) or (H, W, 3)
      - obs is directly an array of shape (players, H, W, 3) or (H, W, 3)
    Returns: (H, W, 3) uint8
    """
    candidate_keys = ["rgb", "image", "pixels", "frame"]

    arr = None
    if isinstance(obs, dict):
        for k in candidate_keys:
            if k in obs:
                arr = obs[k]
                break
    else:
        arr = obs

    if arr is None:
        raise RuntimeError(
            "No RGB frame found in observation. Ensure you're using the pixels env."
        )

    # Convert JAX array → numpy
    arr = np.array(arr)

    # If per-player observations: (players, H, W, 3). Pick player 0.
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise RuntimeError(f"Unexpected RGB shape: {arr.shape}")

    # Ensure uint8
    if arr.dtype != np.uint8:
        # Many envs already give uint8; if not, clip & cast.
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def rollout_pixels_video(steps=300, num_players=1, seed=0, fps=20, out_path="craftax_pixels.mp4"):
    rng = jax.random.PRNGKey(seed)

    env = Env()
    env_params = env.default_params

    # Set number of players for the rollout
    env.static_env_params = env.static_env_params.replace(num_players=num_players)

    # Reset
    rng, _rng = jax.random.split(rng)
    obs, state = env.reset(_rng, env_params)

    frames = []
    for t in range(steps):
        # sample 1 action per player uniformly from allowed set
        rng, _rng = jax.random.split(rng)
        action_indices = jax.random.randint(_rng, shape=(num_players,), minval=0, maxval=ALLOWED_ACTIONS.shape[0])
        actions = ALLOWED_ACTIONS[action_indices]

        # step
        rng, _rng = jax.random.split(rng)
        obs, state, reward, done, info = env.step(_rng, state, actions, env_params)

        # grab a frame (player 0's view if multi-player)
        frame = extract_rgb_frame(obs)
        frames.append(frame)

        # If all players are done, you can reset; here we just stop early.
        if np.all(np.array(done)):
            break

    # Save video
    if not out_path.endswith(".mp4"):
        out_path += ".mp4"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    iio.imwrite(out_path, np.stack(frames, axis=0), fps=fps, codec="h264", quality=8)
    print(f"Saved {len(frames)} frames to {out_path} at {fps} FPS.")


def main():
    parser = argparse.ArgumentParser(description="Run a random agent in CraftaxClassicPixelsEnv and save an MP4.")
    parser.add_argument("--steps", type=int, default=300, help="Max environment steps.")
    parser.add_argument("--players", type=int, default=1, help="Number of players.")
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed.")
    parser.add_argument("--fps", type=int, default=20, help="Output video FPS.")
    parser.add_argument("--out", type=str, default="craftax_pixels.mp4", help="Output video path (.mp4).")
    args = parser.parse_args()

    rollout_pixels_video(
        steps=args.steps,
        num_players=args.players,
        seed=args.seed,
        fps=args.fps,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
