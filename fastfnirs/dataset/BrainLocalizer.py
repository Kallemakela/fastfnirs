import numpy as np

from fnemo.utils.montage import project_each_sensor_to_sphere, scale as scale_mtg
from fnemo.utils.standard_montage import get_used_rows_cols, trim_standard_montage, get_std_sensor_grid

class BrainLocalizer:
    """
    Class for localizing channels on a montage.
    
    E.g. get_used_chs() returns the sensors that are closest to the given channels. 
    """
    def __init__(self, montage, montage_name=None, scale=1.0):
        self.montage = montage
        self.montage_name = montage_name
        self.scale = scale

        if scale == 'sphere':
            self.montage, self.scalars = project_each_sensor_to_sphere(self.montage, return_inverse=True)
        elif isinstance(scale, (int, float)):
            self.montage = scale_mtg(self.montage, scale)
        
        self.channel_locations = self.montage.get_positions()
        
    @property
    def channel_locations_array(self):
        return np.array([v for v in self.channel_locations['ch_pos'].values()])
    
    @property
    def ch_names(self):
        return np.array(list(self.channel_locations['ch_pos'].keys()))

    def get_ix(self, x, y, z):
        """Returns index of closest channel"""
        dists = np.linalg.norm(self.channel_locations_array - [x, y, z], axis=1)
        return np.argmin(dists)
    
    def get_coords(self, x_ix):
        """Returns coordinates of closest channel"""
        return self.channel_locations_array[x_ix]
    
    def get_ch2grid(self, chs):
        """Returns a dict with channel names as keys and (x_ix, y_ix) as values"""
        self.ch2grid = {}
        for ch in chs:
            x, y, z = ch['loc'][:3]
            ch_name = ch['ch_name']
            ch_ix = self.get_ix(x, y, z)
            self.ch2grid[ch_name] = ch_ix
        return self.ch2grid
    
    def get_ch_name(self, ch_ix):
        """Returns channel name"""
        return self.ch_names[ch_ix]

    def get_used_chs(self, chs):
        ch2grid = self.get_ch2grid(chs)
        used_chs = [self.get_ch_name(ch_ix) for ch_ix in ch2grid.values()]
        return used_chs

def trim_bl(bl, chs):
    mtg_name = bl.montage_name
    grid = get_std_sensor_grid(mtg_name)
    used_chs = bl.get_used_chs(chs)
    used_rows, used_cols = get_used_rows_cols(used_chs, grid)
    new_mtg = trim_standard_montage(bl.montage, used_rows, used_cols, grid)
    return BrainLocalizer(new_mtg, scale=bl.scale)