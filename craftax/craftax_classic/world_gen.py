from functools import partial

from craftax.craftax_classic.constants import *
from craftax.craftax_classic.game_logic import calculate_light_level, get_distance_map
from craftax.craftax_classic.envs.craftax_state import EnvParams, EnvState, Inventory, Mobs, StaticEnvParams
from craftax.craftax_classic.util.noise import generate_fractal_noise_2d


def generate_world(rng, params: EnvParams, static_params: StaticEnvParams):
    fractal_noise_angles = params.fractal_noise_angles
    rng, _rng = jax.random.split(rng, num=2)

    player_position = jnp.array(
        [
            [1, 0],  # Player 0
            [7, 6],  # Player 1
        ],
        dtype=jnp.int32,
    )

# ------------------------------------------------------------------
    # FIXED 8x8 LAVA / GRASS / TREE MAP
    # ------------------------------------------------------------------
    # Optional: gate this with a flag in EnvParams, e.g. params.use_fixed_map
    # and/or assert map_size is (8, 8).
    #
    # if getattr(params, "use_fixed_map", False) and tuple(static_params.map_size) == (8, 8):
    #     use_fixed = True
    # else:
    #     use_fixed = False
    #
    # if use_fixed:
    #     ... (use the fixed map below and skip noise-based worldgen)
    # else:
    #     ... (original code)
    # ------------------------------------------------------------------

    fixed_chars = [
        "TLWLGLWT",
        "GTTLTGWT",
        "LWTGTLGT",
        "TGTWGLGT",
        "TGTGTGTG",
        "TTGLGTGT",
        "WLTLTGLG",
        "TTTTGTGT",
    ]


    char_to_block = {
        "G": BlockType.GRASS.value,
        "L": BlockType.LAVA.value,
        "T": BlockType.TREE.value,
        "W": BlockType.WATER.value,
    }


    fixed_map = jnp.array(
        [[char_to_block[c] for c in row] for row in fixed_chars],
        dtype=jnp.int32,
    )

    # Make sure all players spawn is grass
    fixed_map = fixed_map.at[player_position[:, 0], player_position[:, 1]].set(
        BlockType.GRASS.value
    )


    map = fixed_map

    
    # Zombies

    z_pos = jnp.zeros((static_params.max_zombies, 2), dtype=jnp.int32)
    z_health = jnp.ones(static_params.max_zombies, dtype=jnp.int32)
    z_mask = jnp.zeros(static_params.max_zombies, dtype=bool)

    # z_pos = z_pos.at[0].set(player_position + jnp.array([1, 0]))
    # z_mask = z_mask.at[0].set(True)
    # z_pos = z_pos.at[1].set(player_position + jnp.array([2, 0]))
    # z_mask = z_mask.at[1].set(True)

    zombies = Mobs(
        position=z_pos,
        health=z_health,
        mask=z_mask,
        attack_cooldown=jnp.zeros(static_params.max_zombies, dtype=jnp.int32),
    )

    # Skeletons
    sk_positions = jnp.zeros((static_params.max_skeletons, 2), dtype=jnp.int32)
    sk_healths = jnp.zeros(static_params.max_skeletons, dtype=jnp.int32)
    sk_mask = jnp.zeros(static_params.max_skeletons, dtype=bool)

    skeletons = Mobs(
        position=sk_positions,
        health=sk_healths,
        mask=sk_mask,
        attack_cooldown=jnp.zeros(static_params.max_skeletons, dtype=jnp.int32),
    )

    # Arrows
    arrow_positions = jnp.zeros((static_params.max_arrows, 2), dtype=jnp.int32)
    arrow_healths = jnp.zeros(static_params.max_arrows, dtype=jnp.int32)
    arrow_masks = jnp.zeros(static_params.max_arrows, dtype=bool)

    arrows = Mobs(
        position=arrow_positions,
        health=arrow_healths,
        mask=arrow_masks,
        attack_cooldown=jnp.zeros(static_params.max_arrows, dtype=jnp.int32),
    )

    arrow_directions = jnp.ones((static_params.max_arrows, 2), dtype=jnp.int32)

    # # Cows
    # cows = Mobs(
    #     position=jnp.zeros((static_params.max_cows, 2), dtype=jnp.int32),
    #     health=jnp.ones(static_params.max_cows, dtype=jnp.int32) * params.cow_health,
    #     mask=jnp.zeros(static_params.max_cows, dtype=bool),
    #     attack_cooldown=jnp.zeros(static_params.max_cows, dtype=jnp.int32),
    # )

    # Can you set the mask for the cows (position, mask)
        # --------------------
    # COWS – 7 hard-placed on grass tiles of fixed_chars
    # --------------------
    # Valid 'G' tiles in the new fixed_chars:
    # (row, col): (0,4), (1,0), (1,5), (2,3), (2,6), (3,1), (3,4),
    #             (3,6), (4,1), (4,3), (4,5), (4,7), (5,2), (5,4),
    #             (5,6), (6,5), (6,7), (7,4), (7,6)
    #
    # We'll pick 7 of these that are safely away from the spawn.

        # --------------------
    # COWS – 12 hard-placed on grass tiles of fixed_chars
    # --------------------
    # We choose 12 grass positions, avoiding player spawns (1,0) and (7,6).
    cow_positions_init = jnp.array(
        [
            [0, 4],  # G
            [1, 5],  # G
            [2, 3],  # G
            [2, 6],  # G
            [3, 1],  # G
            [3, 4],  # G
            [3, 6],  # G
            [4, 1],  # G
            [4, 3],  # G
            [4, 5],  # G
            [4, 7],  # G
            [5, 2],  # G
        ],
        dtype=jnp.int32,
    )

    # Make sure we don't exceed capacity of the environment
    num_init_cows = min(static_params.max_cows, cow_positions_init.shape[0])

    # Full position array (max_cows slots), first num_init_cows filled
    cow_positions = jnp.zeros((static_params.max_cows, 2), dtype=jnp.int32)
    cow_positions = cow_positions.at[:num_init_cows].set(
        cow_positions_init[:num_init_cows]
    )

    # Mask: first num_init_cows are alive / active
    cow_mask = jnp.zeros(static_params.max_cows, dtype=bool)
    cow_mask = cow_mask.at[:num_init_cows].set(True)

    cows = Mobs(
        position=cow_positions,
        health=jnp.ones(static_params.max_cows, dtype=jnp.int32) * params.cow_health,
        mask=cow_mask,
        attack_cooldown=jnp.zeros(static_params.max_cows, dtype=jnp.int32),
    )

    # Plants
    growing_plants_positions = jnp.zeros(
        (static_params.max_growing_plants, 2), dtype=jnp.int32
    )
    growing_plants_age = jnp.zeros(static_params.max_growing_plants, dtype=jnp.int32)
    growing_plants_mask = jnp.zeros(static_params.max_growing_plants, dtype=bool)

    rng, _rng = jax.random.split(rng)

    state = EnvState(
        map=map,
        mob_map=jnp.zeros(static_params.map_size, dtype=bool),
        player_position=player_position,
        player_direction=jnp.full(static_params.num_players, Action.UP.value, dtype=jnp.int32),
        player_health=jnp.full(static_params.num_players, 9, dtype=jnp.int32),
        player_food=jnp.full(static_params.num_players, 9, dtype=jnp.int32),
        player_drink=jnp.full(static_params.num_players, 9, dtype=jnp.int32),
        player_energy=jnp.full(static_params.num_players, 9, dtype=jnp.int32),
        player_recover=jnp.full(static_params.num_players, 0.0, dtype=jnp.float32),
        player_hunger=jnp.full(static_params.num_players, 0.0, dtype=jnp.float32),
        player_thirst=jnp.full(static_params.num_players, 0.0, dtype=jnp.float32),
        player_fatigue=jnp.full(static_params.num_players, 0.0, dtype=jnp.float32),
        is_sleeping=jnp.full(static_params.num_players, False),
        inventory=Inventory.generate_inventory(static_params.num_players),
        zombies=zombies,
        skeletons=skeletons,
        arrows=arrows,
        arrow_directions=arrow_directions,
        cows=cows,
        growing_plants_positions=growing_plants_positions,
        growing_plants_age=growing_plants_age,
        growing_plants_mask=growing_plants_mask,
        achievements=jnp.zeros((static_params.num_players, len(Achievement)), dtype=int),
        light_level=calculate_light_level(0, params),
        state_rng=_rng,
        timestep=0,
    )

    return state


###########################################################################################################################################################


# def generate_random_world(rng, params: EnvParams, static_params: StaticEnvParams):
#     # Zombies
#
#     z_pos = jnp.zeros((static_params.max_zombies, 2), dtype=jnp.int32)
#     z_health = jnp.ones(static_params.max_zombies, dtype=jnp.int32)
#     z_mask = jnp.zeros(static_params.max_zombies, dtype=bool)
#
#     # z_pos = z_pos.at[0].set(player_position + jnp.array([1, 0]))
#     # z_mask = z_mask.at[0].set(True)
#
#     zombies = Mobs(
#         position=z_pos,
#         health=z_health,
#         mask=z_mask,
#         attack_cooldown=jnp.zeros(static_params.max_zombies, dtype=jnp.int32),
#     )
#
#     # Skeletons
#     sk_positions = jnp.zeros((static_params.max_skeletons, 2), dtype=jnp.int32)
#     sk_healths = jnp.zeros(static_params.max_skeletons, dtype=jnp.int32)
#     sk_mask = jnp.zeros(static_params.max_skeletons, dtype=bool)
#
#     skeletons = Mobs(
#         position=sk_positions,
#         health=sk_healths,
#         mask=sk_mask,
#         attack_cooldown=jnp.zeros(static_params.max_skeletons, dtype=jnp.int32),
#     )
#
#     # Arrows
#     arrow_positions = jnp.zeros((static_params.max_arrows, 2), dtype=jnp.int32)
#     arrow_healths = jnp.zeros(static_params.max_arrows, dtype=jnp.int32)
#     arrow_masks = jnp.zeros(static_params.max_arrows, dtype=bool)
#
#     arrows = Mobs(
#         position=arrow_positions,
#         health=arrow_healths,
#         mask=arrow_masks,
#         attack_cooldown=jnp.zeros(static_params.max_arrows, dtype=jnp.int32),
#     )
#
#     arrow_directions = jnp.ones((static_params.max_arrows, 2), dtype=jnp.int32)
#
#     # Cows
#     cows = Mobs(
#         position=jnp.zeros((static_params.max_cows, 2), dtype=jnp.int32),
#         health=jnp.ones(static_params.max_cows, dtype=jnp.int32) * params.cow_health,
#         mask=jnp.zeros(static_params.max_cows, dtype=bool),
#         attack_cooldown=jnp.zeros(static_params.max_cows, dtype=jnp.int32),
#     )
#
#     # Plants
#     growing_plants_positions = jnp.zeros(
#         (static_params.max_growing_plants, 2), dtype=jnp.int32
#     )
#     growing_plants_age = jnp.zeros(static_params.max_growing_plants, dtype=jnp.int32)
#     growing_plants_mask = jnp.zeros(static_params.max_growing_plants, dtype=bool)
#
#     rng, _rng = jax.random.split(rng)
#     map = jax.random.choice(
#         _rng, jnp.arange(2, 17), shape=static_params.map_size
#     ).astype(int)
#
#     state = EnvState(
#         map=map,
#         player_position=jnp.zeros(2, dtype=jnp.int32),
#         player_direction=Action.UP.value,
#         player_health=9,
#         player_food=9,
#         player_drink=9,
#         player_energy=9,
#         player_recover=0.0,
#         player_hunger=0.0,
#         player_thirst=0.0,
#         player_fatigue=0.0,
#         is_sleeping=False,
#         inventory=Inventory.generate_inventory(params.num_players),
#         zombies=zombies,
#         skeletons=skeletons,
#         arrows=arrows,
#         arrow_directions=arrow_directions,
#         cows=cows,
#         growing_plants_positions=growing_plants_positions,
#         growing_plants_age=growing_plants_age,
#         growing_plants_mask=growing_plants_mask,
#         achievements=jnp.zeros((22,), dtype=bool),
#         light_level=calculate_light_level(0, params),
#         timestep=0,
#     )
#
#     return state
