# import jax
# import jax.numpy as jnp

# from craftax.craftax_classic.envs.craftax_symbolic_env import CraftaxClassicSymbolicEnv
# from craftax.craftax_classic.constants import Action

# rng = jax.random.PRNGKey(42)  # generate a random number from a seed
# env = CraftaxClassicSymbolicEnv()  # create environment instance
# env_params = env.default_params
# # you can set the number of players like so
# env.static_env_params = env.static_env_params.replace(num_players=4)

# # reset the environment. Obs has shape (n, num_obs). You can get num_obs through env.observation_space(env_params).shape[0]
# # This gives one observation per player, for n players
# rng, _rng = jax.random.split(rng)
# obs, env_state = env.reset(_rng, env_params)

# # Let's pick some actions for the players (there are 17 different actions
# action = jnp.array([Action.UP.value, Action.DOWN.value, Action.DO.value, Action.SLEEP.value])

# # Step the environment
# rng, _rng = jax.random.split(rng)
# obs, env_state, reward, done, info = env.step(_rng, env_state, action, env_params)



"""
Actions that can be taken: 

1. NOOP
2 LEFT
3. RIGHT
4. UP
5. DOWN
6. DO (interact)
7. SLEEP
8. PLACE_STONE
9. PLACE_TABLE
10. PLACE_FURNACE
11. PLACE_PLANT
12. MAKE_WOOD_PICKAXE
13. MAKE_STONE_PICKAXE
14. MAKE_IRON_PICKAXE
15. MAKE_WOOD_SWORD
16. MAKE_STONE_SWORD
17. MAKE_IRON_SWORD

Limit actions: 
- left 
- right
- up 
- down 
- Do (this can allow it to do things such as picking berries)

"""

"""
Overall enviornment modifications
- only focus on avoiding lava 
- remove the pentalty for being alive, no reduction in health in the health parameters
   - hunger_decay, thirst_decay, energy_decay, starvation_damage, dehydration_damage, exhaustion_damage, alive_penalty
- make the reward for picking a berry higher than say other things to priotize the action of 
  picking a certain color berry for example

- train multiple episodes where they have a set number of steps. Say 100 episodes with 1000 steps each

"""


# import jax
# import jax.numpy as jnp

# from craftax.craftax_classic.envs.craftax_symbolic_env import CraftaxClassicSymbolicEnv
# from craftax.craftax_classic.constants import Action

# rng = jax.random.PRNGKey(42)

# env = CraftaxClassicSymbolicEnv()
# env_params = env.default_params
# env.static_env_params = env.static_env_params.replace(num_players=4)
# num_players = env.static_env_params.num_players

# # Some Craftax builds name the no-op as NOOP; others use SLEEP.
# # This picks NOOP if it exists, otherwise falls back to SLEEP.
# NOOP_OR_SLEEP = getattr(Action, "NOOP", Action.SLEEP)

# # Allowed action values we’ll sample from (uniformly)
# allowed_actions = jnp.array([
#     NOOP_OR_SLEEP.value,
#     Action.LEFT.value,
#     Action.RIGHT.value,
#     Action.UP.value,
#     Action.DOWN.value,
#     Action.DO.value,          # "interact"
# ])

# # Reset
# rng, _rng = jax.random.split(rng)
# obs, env_state = env.reset(_rng, env_params)

# # --- One random step ---
# rng, _rng = jax.random.split(rng)
# # Sample one action per player (i.i.d. uniform from allowed_actions)
# action = jax.random.choice(_rng, allowed_actions, shape=(num_players,))
# obs, env_state, reward, done, info = env.step(_rng, env_state, action, env_params)

# # --- Optional: run a few random steps ---
# T = 10
# for _ in range(T - 1):
#     rng, _rng = jax.random.split(rng)
#     action = jax.random.choice(_rng, allowed_actions, shape=(num_players,))
#     obs, env_state, reward, done, info = env.step(_rng, env_state, action, env_params)
#     # (optional) check per-player done flags in `done`

# run_craftax_frames.py
# make_grid_animation.py
import os
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
import jax, jax.numpy as jnp

from craftax.craftax_classic.envs.craftax_symbolic_env import CraftaxClassicSymbolicEnv as Env

from craftax.craftax_classic.constants import Action

OUTDIR = Path("./anim_out")
OUTDIR.mkdir(parents=True, exist_ok=True)

ROWS, COLS = 2, 3          # 2x3 grid
NUM_EPISODES = ROWS * COLS  # 6 episodes in the grid
MAX_STEPS = 120             # frames per episode
FPS = 10

# restrict actions to move + interact (you can replace with your policy)
ALLOWED = jnp.array([
    Action.UP.value, Action.DOWN.value,
    Action.LEFT.value, Action.RIGHT.value, Action.DO.value
], dtype=jnp.int32)

def to_uint8(img):
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        arr = arr - arr.min()
        if arr.max() > 0:
            arr = arr / (arr.max() + 1e-6)
        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    return arr

def to_rgb_frame(obs, info):
    """
    Try to return an HxWx3 uint8 image.
    Priority: pixel obs -> info['rgb'] -> simple debug visual if needed.
    """
    # case 1: obs already pixels (H,W,3 or 4)
    if obs.ndim == 3 and obs.shape[-1] in (3, 4):
        rgb = to_uint8(obs)
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        return rgb
    # case 2: info carries render
    if isinstance(info, dict) and "rgb" in info:
        rgb = to_uint8(info["rgb"])
        if rgb.ndim == 2:
            rgb = np.repeat(rgb[..., None], 3, axis=-1)
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        return rgb
    # fallback: tiny debug strip from vector obs
    vec = np.asarray(obs).reshape(-1).astype(np.float32)
    H, W = 96, 96
    img = np.zeros((H, W, 3), dtype=np.float32)
    if vec.size:
        v = vec.copy()
        v = v - v.min()
        if v.max() > 0:
            v = v / (v.max() + 1e-6)
        n = min(W, v.size)
        img[H//3, :n, 0] = v[:n]
        if v.size > n:
            m = min(W, v.size - n)
            img[H//2, :m, 2] = v[n:n+m]
    return (img * 255).astype(np.uint8)

def rollout_frames(env, env_params, rng, max_steps):
    """Run one episode of random actions and collect frames."""
    rng, key = jax.random.split(rng)
    obs, state = env.reset(key, env_params)
    frames = []
    for t in range(max_steps):
        rng, arng, sk = jax.random.split(rng, 3)
        idx = jax.random.randint(arng, (env.static_env_params.num_players,), 0, ALLOWED.shape[0])
        actions = ALLOWED[idx]
        obs, state, reward, done, info = env.step(sk, state, actions, env_params)
        # record frame for agent 0 viewpoint
        frame = to_rgb_frame(np.array(obs[0]), info)
        frames.append(frame)
        if bool(jnp.all(done)):
            break
    return rng, frames

def tile_frame_grid(frames_per_ep, rows, cols, background=(0, 0, 0)):
    """
    frames_per_ep: list of episode frame lists (each list: [T_i x H x W x 3])
    Returns a list of tiled frames of length T = min_i T_i
    """
    # equalize episode lengths to the minimum so we can sync
    T = min(len(fr) for fr in frames_per_ep)
    # resize everything to the smallest H,W among first frames
    Hmin = min(fr[0].shape[0] for fr in frames_per_ep)
    Wmin = min(fr[0].shape[1] for fr in frames_per_ep)

    def resize(img, H, W):
        # simple nearest neighbor using numpy slicing if already same size
        if img.shape[0] == H and img.shape[1] == W:
            return img
        # fallback: use PIL for resizing without adding a new dependency
        from PIL import Image
        return np.array(Image.fromarray(img).resize((W, H), resample=Image.NEAREST))

    tiled = []
    for t in range(T):
        cells = []
        for k in range(rows * cols):
            fr = frames_per_ep[k][t]
            fr_r = resize(fr, Hmin, Wmin)
            cells.append(fr_r)
        # build grid
        row_imgs = []
        for r in range(rows):
            row_strip = np.concatenate(cells[r*cols:(r+1)*cols], axis=1)
            row_imgs.append(row_strip)
        grid = np.concatenate(row_imgs, axis=0)
        tiled.append(grid)
    return tiled

def main():
    rng = jax.random.PRNGKey(0)

    # --- Make env: 1 player per episode is fine for a clean view; change if you want.
    env = Env()
    env_params = env.default_params
    env.static_env_params = env.static_env_params.replace(num_players=1)

    # --- Collect N episodes
    episodes = []
    for i in range(NUM_EPISODES):
        rng, frs = rollout_frames(env, env_params, rng, MAX_STEPS)
        episodes.append(frs)

    # --- Build the 2x3 tiled animation
    grid_frames = tile_frame_grid(episodes, ROWS, COLS)

    # --- Save GIF and MP4
    gif_path = OUTDIR / "craftax_grid.gif"
    mp4_path = OUTDIR / "craftax_grid.mp4"
    imageio.mimsave(gif_path, grid_frames, fps=FPS)
    try:
        imageio.mimsave(mp4_path, grid_frames, fps=FPS)  # needs imageio-ffmpeg installed
    except Exception:
        pass

    print(f"Saved: {gif_path}")
    if mp4_path.exists():
        print(f"Saved: {mp4_path}")

if __name__ == "__main__":
    main()
