import gym
from gym import error, spaces, utils
from gym.utils import seeding

class DistopiaEnv(gym.Env):
    """
    The main OpenAI Gym class. It encapsulates an environment with
    arbitrary behind-the-scenes dynamics. An environment can be
    partially or fully observed.
    The main API methods that users of this class need to know are:
        step
        reset
        render
        close
        seed
    And set the following attributes:
        action_space: The Space object corresponding to valid actions
        observation_space: The Space object corresponding to valid observations
        reward_range: A tuple corresponding to the min and max possible rewards
    Note: a default reward range set to [-inf,+inf] already exists. Set it if you want a narrower range.
    The methods are accessed publicly as "step", "reset", etc.. The
    non-underscored versions are wrapper methods to which we may add
    functionality over time.
    """
    metadata = {'render.modes': ['human']}
    NUM_DISTRICTS = 4
    BLOCKS_PER_DISTRICT = 3
    GRID_WIDTH = 100 #width of a grid in pixels

    moves = ['ADD0','ADD1','ADD2','ADD3','ADD4','ADD5','ADD6','ADD7','REMOVE','MOVE_N','MOVE_S','MOVE_E','MOVE_W']

    def __init__(self, screen_size):
        self.width, self.height = (dim//self.GRID_WIDTH for dim in screen_size)
        # each block can move in four directions
        self.action_space = spaces.Dict({
            'x': spaces.Discrete(self.width),
            'y': spaces.Discrete(self.height),
            'move': spaces.Discrete(13) #(Add0, Add1, Add2,...Remove,MoveN,MoveS,MoveE,MoveW)
        })
        # the state space is the x,y coords of all blocks i.e. (x0,y0,x1,y1,x2,y2...)
        self.observation_space = spaces.Box(low=0, high=NUM_DISTRICTS-1, shape=(self.height, self.width), dtype=np.uint8) # IMPORTANT is this w x h or h x w??
        spaces.Tuple((spaces.Discrete(10) for x in range(self.NUM_DISTRICTS * self.BLOCKS_PER_DISTRICT * 2)))
        self.reward_range = (-float('inf'), float('inf'))
    def step(self, action):
        """
        Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.
        Accepts an action and returns a tuple (observation, reward, done, info).
        Args:
            action (object): an action provided by the agent
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.
        Returns: 
            observation (object): the initial observation.
        """
    def render(self, mode-'human'):
        """
        Renders the environment.
        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:
        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).
        Note:
            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.
        Args:
            mode (str): the mode to render with
        Example:
        class MyEnv(Env):
            metadata = {'render.modes': ['human', 'rgb_array']}
            def render(self, mode='human'):
                if mode == 'rgb_array':
                    return np.array(...) # return RGB frame suitable for video
                elif mode == 'human':
                    ... # pop up a window and render
                else:
                    super(MyEnv, self).render(mode=mode) # just raise an exception
        """
    def close(self):
        """
        Override close in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).
        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.
        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        return

