from gym.envs.registration import register

register(
    id='distopia-v0',
    entry_point='gym_distopia.envs:DistopiaEnv',
)
register(
    id='distopia-extrahard-v0',
    entry_point='gym_distopia.envs:DistopiaExtraHardEnv',
)
