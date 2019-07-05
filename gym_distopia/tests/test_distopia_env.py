from gym_distopia.envs.distopia_env import DistopiaEnv
from copy import deepcopy
import numpy as np

class TestDistopiaEnv:
    
    def state_equal(self,state1,state2):
        for i,district in enumerate(state1):
            for j,block in enumerate(district):
                for k,coord in enumerate(block):
                    if coord != state2[i][j][k]:
                        return False
        return True
    
    def test_constructor(self):
        # make sure the constructor runs
        de = DistopiaEnv((1920,1080),'population')
        assert de != None
        # check to see that things got created correctly
        # check to see that the action space is right
        # check the dimensions of the state space
        assert np.asarray(de.observation_space).shape == (de.NUM_DISTRICTS,de.BLOCKS_PER_DISTRICT,2)
        assert np.asarray(de.districts).shape == (de.NUM_DISTRICTS,de.BLOCKS_PER_DISTRICT,2)
        # check something about the voronoi
    def test_update_state(self):
        #de = DistopiaEnv((1920,1080),'population')
        #de._update_state(district,block,old_loc,new_loc)
        #state = np.zeros()
        assert 1 == 1
    def test_apply_action(self):
        de = DistopiaEnv((1920,1080),'population')
        district = np.random.randint(de.NUM_DISTRICTS)
        block = np.random.randint(de.BLOCKS_PER_DISTRICT)
        # check that the "do nothing" action does nothing
        state = deepcopy(de.districts)
        res = de._apply_action(district,block,0)
        assert res == True
        assert self.state_equal(state,de.districts) == True
        # check that moving a block to an open spot updates the state
        
        target_loc = de.districts[district][block] + (0,-1) # let's try going south
        while target_loc[1] < 0:
            district = np.random.randint(de.NUM_DISTRICTS)
            block = np.random.randint(de.BLOCKS_PER_DISTRICT)
            target_loc = de.districts[district][block] + (0,-1)
        if str(target_loc) in de.occupied:
            print("overlap in the test. let's move this block somewhere illegal")
            mvdist,mvblock = de.occupied[str(target_loc)]
            de._update_state(mvdist,mvblock,target_loc,np.asarray(-1,-1))
        state = deepcopy(de.districts)
        res = de._apply_action(district,block,2)
        assert res == True
        assert self.state_equal(state,de.districts) != True
        assert np.all([de.districts[district][block],target_loc])


        target_loc = de.districts[district][block] + (1,0) # let's try going east
        while target_loc[0] > de.width:
            district = np.random.randint(de.NUM_DISTRICTS)
            block = np.random.randint(de.BLOCKS_PER_DISTRICT)
            target_loc = de.districts[district][block] + (1,0)
        if str(target_loc) in de.occupied:
            print("overlap in the test. let's move this block somewhere illegal")
            mvdist,mvblock = de.occupied[str(target_loc)]
            de._update_state(mvdist,mvblock,target_loc,np.asarray(-1,-1))
        state = deepcopy(de.districts)
        res = de._apply_action(district,block,3)
        assert res == True
        assert self.state_equal(state,de.districts) != True
        assert np.all([de.districts[district][block],target_loc])

        # check that moving a block where there already is one fails
        occupied_loc = de.districts[district][block]
        if de.NUM_DISTRICTS > 1:
            target_district = district + 1 if district < (de.NUM_DISTRICTS-1) else district - 1
        if de.BLOCKS_PER_DISTRICT > 1:
            target_block = block + 1 if block < (de.BLOCKS_PER_DISTRICT-1) else block - 1
        target_old_loc = de.districts[target_district][target_block]
        target_loc = occupied_loc + (0, -1)
        de._update_state(target_district,target_block,target_old_loc,target_loc)
        state = deepcopy(de.districts)
        res = de._apply_action(target_district,target_block,1)
        assert res == False
        assert self.state_equal(state,de.districts)
    def test_take_action(self):
        assert 1 == 1
    def test_place_block(self):
        assert 1 == 1
    def reset(self):
        de = DistopiaEnv((1920,1080),'population')
        state = [[np.zeros(2) for block in de.BLOCKS_PER_DISTRICT] for district in de.NUM_DISTRICTS]
        de.reset(initial=state)
        assert de.districts == state
        de.reset()
        assert de.districts != state
