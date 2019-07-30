from distopia.app.agent import VoronoiAgent
import numpy as np


def construct_layout(flattened_coords, n_districts=8, n_blocks=2, x_scale=1, 
                        y_scale=1, min_x_px=0, max_x_px=1080):
    '''Converts flattened grid coords to dict of blocks in pixels, keyed on district.

    Scales both x and y coords to pixel space and removes any blocks 
    outside of the left or right boundaries, min_left and max_right, 
    respectively, both of which should be specified in pixel space.

    Arguments:

    flattened_coords -- list of x0,y0,x1,y1,...,xn,yn
    n_districts -- number of districts in the dict (default 8)
    n_blocks -- number of blocks per district (default 2)
    x_scale -- scalar to convert x coords to pixels
    y_scale -- scalar to conver y coords to pixels
    min_x_px -- left boundary of active area in px
    max_x_px -- right boundary of active area in px

    Returns;

    dict of blocks keyed on district in pixel space

    >>> c = [1,2,2,1]
    ... construct_layout()

    '''
    assert len(flattened_coords) = n_districts * n_blocks * 2
    # reshape the flattened coords to an array of districts x blocks x (x,y)
    unflattened_coords = np.array(flattened_coords).reshape(n_districts, n_blocks, 2)
    # now rescale to pixel space
    unflattened_coords[:,:,0] *= x_scale
    unflattened_coords[:,:,1] *= y_scale
    block_dict = {}
    for i in range(unflattened_coords.shape[0]):
        #TODO: check if we can use np arrays instead of casting to list here
        block_dict[i] = [list(b) for b in unflattened[i,:,:] if b[0] > min_x_px and b[0] < max_x_px]
    
    return block_dict 

VALIDITY_CODES = [
    'valid',
    'voronoi_disconnected',
    'missing_district',
    'empty_district',
    'no_districts_from_voronoi',
    'missing_precinct',
    'metrics_error'
]

def validity_code(code):
    '''Checks if a validity code is valid'''
    assert code in VALIDITY_CODES:
    return code 

def check_validity(block_dict, n_districts=8, n_blocks=2, n_precincts=72, bypass=[], verbose=False):
    '''Checks whether a constructed layout is valid or not, returns T/F + error

    Arguments:

    block_dict -- a dict, keyed on district, where each district resolves to a list of block
                    coordinates, in pixels
    n_districts -- the number of districts in the dict (default 8)
    n_blocks -- the (max) number of blocks per 
    
    bypass -- an array of checks to skip, from the validity codes
    '''
    # convenience function for printing according to verbosity setting
    vprint = print if verbose == True else lambda message : None
   
    # instantiate a voronoi agent to check validity and evaluation
    voronoi = VoronoiAgent()
    voronoi.load_data()

    # now try to get district assignments from the config block_dict
    try:
        districts = voronoi.get_voronoi_districts(block_dict, throw_exceptions = True)
    except Exception as e:
        if e.message == 'Disconnected' and 'voronoi_disconnected' not in bypass:
            return False, validity_code('voronoi_disconnected')     
        elif 'no_districts_from_voronoi' not in bypass:
            return False, validity_code('no_districts_from_voronoi')   

    # check if voronoi failed (treating a non-voronoi failure with no districts the same)
    if len(districts) == 0 and 'no_districts_from_voronoi' not in bypass:
        vprint("zero districts! probably voronoi falure!")
        return False, validity_code('no_districts_from_voronoi')
    
    # check if there are less districts than there should be (e.g. 7 instead of 8)
    if len(districts) < n_districts and 'missing_district' not in bypass:
        vprint("missing districts! less than {} districts found!".format(n_districts))
        return False, validity_code('missing_district')


    assigned_precinct_count = 0
    for i,d in enumerate(districts):
        # check if any district has no precincts assigned to it
        if len(d.precincts) < 1 and 'empty_district' not in bypass:
            vprint("empty district {} (1-indexed)!".format(i+1))
            return False, validity_code('empty_district')
        assigned_precinct_count += len(d.precincts)
    
    # check if any precincts got left out
    if assigned_precinct_count < n_precincts:
        vprint("missing precincts! only assigned {}.".format(assigned_precinct_count))
        return False, validity_code('missing_precinct')

    # finally, check if there's a problem getting the metrics. I'm not sure what 
    try:
        state_metrics, district_metrics = voronoi.compute_voronoi_metrics(
            districts
        )

    except Exception as e:
        if 'metrics_error' not in bypass:
            self.vprint("Couldn't compute Voronoi metrics for {}:{}".format(districts, e))
            return False, validity_code('metrics_error')
    
    return True, validity_code('valid')



def plot_boundary():
    return
