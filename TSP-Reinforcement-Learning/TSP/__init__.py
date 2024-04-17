from gymnasium.envs.registration import register

register(
    id='TSP-v0',
    entry_point='TSP.envs:TSPEnv',
)