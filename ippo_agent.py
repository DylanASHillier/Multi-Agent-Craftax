from craftax.craftax_classic.train.metacontroller_bc_aux import ClassicMetaController
from craftax.craftax_classic.envs.craftax_state import StaticEnvParams, EnvParams
from craftax.craftax_classic.constants import Action
from craftax.craftax_classic.train.visualize import replay_episode

import jax
import pickle

"""
Plan: 
- you will want to set the controller (env) for the ippo policy 
- then you will want to load in the ippo parameters
- 
"""

print("Initializing the ippo agent... ")
# initialize the controller in order to set the enviornment 
# and save the policy for this agent

# Set the number of agents
agents = 4

from num_agents_params_mod import NumAgentsParamsMod

in_path  = "params-ippo-v4.p"
out_path = "params-ippo.p"

mod = NumAgentsParamsMod.from_pickle(in_path)
report = mod.set_num_agents(
    new_n=agents,          # desired agent count
    old_n=None,            # auto-detect
    update_config_ints=True,
    jitter=1.0,            # optional symmetry-breaking noise when growing
    dry_run=False
)

# write the modified parameters to the new pickle
mod.to_pickle(out_path)

"""
- set it to fixed timesteps to true 
- set the max_timesteps: 100
- set the total num_steps: 1000
- episode num: num_steps / max_timesteps = 10 episodes
"""
metacontroller = ClassicMetaController(
    env_params=EnvParams(max_timesteps=250),
    num_envs=1,
    num_minibatches=1,
    static_parameters=StaticEnvParams(num_players=agents),
    fixed_timesteps=False,
    num_steps=1000000,
    wandb_project="MARLCraftax"
)

"""
Here you will be setting in the ippo paramerters
"""
print("Setting ippo parameters... ")
with open("params-ippo.p", "rb") as f:
    params = pickle.load(f)

"""
Now you will want to run one episode
- what does setting .rng do?
"""
print("Setting the seed... ")
metacontroller.rng = jax.random.PRNGKey(4253)

"""
Load in the model parameters:
- this can be loaded in from - wandb/run-20251117_181137-zhpsz0t7/logs/model_iter_80.pickle
"""
# ckpt_path = "wandb/run-20251117_181137-zhpsz0t7/files/model_iter_80.pickle"

# with open(ckpt_path, "rb") as f:
#     ckpt = pickle.load(f)   # ckpt is your top-level tuple

"""
This will train the learning agent with 
all of the expert agents
"""
print("Training ippo agent... ")
(params, aux_params), opt_state = metacontroller.train()

"""
Here you will want to run one episode with the ippo parameters 
"""
print("Run one episode to show performance to user... ")
states, actions, logits, rewards = metacontroller.run_one_episode(params)

# -------------------------
# Rendering for ALL agents
# -------------------------
import cv2
import numpy as np
from craftax.craftax_classic.renderer import render_craftax_pixels

print("Rendering videos for each agent... ")

# You can adjust these if your renderer changes resolution
FPS = 15.0
FRAME_SIZE = (576, 576)  # (width, height)

def render_frame(states, frame_number, player_idx, num_players):
    """
    Render a single frame for a specific player index.
    """
    # 64 is tile size, num_players should match 'agents'
    data = render_craftax_pixels(states[frame_number], 64, num_players, player_idx)
    # Convert to uint8 and BGR for OpenCV
    frame = np.asarray(data, dtype=np.uint8)[..., ::-1]
    return frame

num_frames = len(states)

for player_idx in range(agents):
    video_name = f"agent{player_idx}_video.mp4"
    print(f"  -> Writing {video_name} ...")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_name, fourcc, FPS, FRAME_SIZE)

    for frame_number in range(num_frames):
        frame = render_frame(states, frame_number, player_idx, agents)
        # If needed, you could resize here to FRAME_SIZE:
        # frame = cv2.resize(frame, FRAME_SIZE, interpolation=cv2.INTER_NEAREST)
        out.write(frame)

    out.release()

print("Done. Videos saved as agent0_video.mp4 ... agent3_video.mp4")
