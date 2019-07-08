from gym.envs.registration import register
from gym_distopia.envs.distopia_rewards import PopulationStdEvaluator
import numpy as np

register(
    id='distopia-v0',
    entry_point='gym_distopia.envs:DistopiaEnv',
    kwargs = {
        'screen_size' : (1920,1080),
        'reward_evaluator' : PopulationStdEvaluator(),
        'blocks_per_district' : 2,
        'step_size' : 4  
    }
   
)


register(
    id='distopia-initial4-v0',
    entry_point='gym_distopia.envs:DistopiaEnv',
    kwargs = {
        'screen_size' : (1920,1080),
        'reward_evaluator' : PopulationStdEvaluator(),
        'blocks_per_district' : 1,
        'step_size' : 1,
        # 'init_state': [
        #     [(12,6),(13,4)],
        #     [np.asarray((20,4)),np.asarray((20,6))],
        #     [np.asarray((20,15)),np.asarray((20,12))],
        #     [np.asarray((12,15)),np.asarray((13,12))],
        # ],
        'init_state':[
            [[12,8]],
            [[16,8]],
            [[15,15]],
            [[12,15]]
        ],
        'always_reset_to_initial':True  
    }
   
)
register(
    id='distopia-extrahard-v0',
    entry_point='gym_distopia.envs:DistopiaExtraHardEnv',
)
