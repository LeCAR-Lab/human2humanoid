# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np
from numpy.random import choice
from scipy import interpolate

from isaacgym import terrain_utils
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg

class Terrain:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:

        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", 'plane']:
            return
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        self.proportions = [np.sum(cfg.terrain_proportions[:i+1]) for i in range(len(cfg.terrain_proportions))]

        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        self.border = int(cfg.border_size/self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)
        if cfg.curriculum:
            self.curriculum_terrain()
        else:    
            self.randomized_terrain()   
        
        self.heightsamples = self.height_field_raw
        if self.type=="trimesh":
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(   self.height_field_raw,
                                                                                            self.cfg.horizontal_scale,
                                                                                            self.cfg.vertical_scale,
                                                                                            self.cfg.slope_treshold)
    
    def randomized_terrain(self):
        proportions = np.array(self.cfg.terrain_proportions) / np.sum(self.cfg.terrain_proportions)
        for k in range(self.cfg.num_sub_terrains):
            print('generating randomized terrains %d / %d     '%(k, self.cfg.num_sub_terrains), end='\r')
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain_type = np.random.choice(self.cfg.terrain_types, p=proportions)
            difficulty = np.random.choice([0.5, 0.75, 0.9])
            if terrain_type == 'smooth_slope' or terrain_type == 'rough_slope':
                difficulty = i / self.cfg.num_rows
            terrain = self.make_terrain(terrain_type, difficulty)
            self.add_terrain_to_map(terrain, i, j)
        print('\n generated all randomized terrains!')
        
    def curriculum_terrain(self):
        proportions = np.array(self.cfg.terrain_proportions) / np.sum(self.cfg.terrain_proportions)
        already_taken_porp = 0.0
        start_col = 0
        end_col = 0
        sub_terrain_dict = {}
        for ter in range(len(self.cfg.terrain_types)):
            terrain_type = self.cfg.terrain_types[ter]
            start_col = end_col + 0
            already_taken_porp += proportions[ter]
            while end_col + 0.1 < self.cfg.num_cols * already_taken_porp: end_col += 1
            sub_terrain_dict[terrain_type] = (start_col, end_col)
            print(terrain_type, 'col:',start_col,':', end_col)

        for terrain_type, col_range in sub_terrain_dict.items():
            print('generating curriculum terrains %s    '%(terrain_type), end='\r')
            start_col = col_range[0]
            end_col = col_range[1]
            for j in range(start_col, end_col):
                for i in range(self.cfg.num_rows):
                    difficulty = i / self.cfg.num_rows
                    terrain = self.make_terrain(terrain_type, difficulty)
                    self.add_terrain_to_map(terrain, i, j)
        print('\n generated all curriculum terrains!')
    
    def make_terrain(self, terrain_type, difficulty):
        terrain = terrain_utils.SubTerrain(   "terrain",
                                width=self.width_per_env_pixels,
                                length=self.width_per_env_pixels,
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale)

        terrain_func = getattr(self, terrain_type+'_terrain_func')
        terrain_func(terrain, difficulty)

        return terrain

    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        env_origin_x = (i + 0.5) * self.env_length
        env_origin_y = (j + 0.5) * self.env_width
        x1 = int((self.env_length/2. - 0.5) / terrain.horizontal_scale)
        x2 = int((self.env_length/2. + 0.5) / terrain.horizontal_scale)
        y1 = int((self.env_width/2. - 0.5) / terrain.horizontal_scale)
        y2 = int((self.env_width/2. + 0.5) / terrain.horizontal_scale)
        env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]

    def gap_terrain_func(self, terrain, difficulty):
        gap_size = 1 * difficulty
        platform_size=3.
        gap_size = int(gap_size / terrain.horizontal_scale)
        platform_size = int(platform_size / terrain.horizontal_scale)

        center_x = terrain.length // 2
        center_y = terrain.width // 2
        x1 = (terrain.length - platform_size) // 2
        x2 = x1 + gap_size
        y1 = (terrain.width - platform_size) // 2
        y2 = y1 + gap_size
    
        terrain.height_field_raw[center_x-x2 : center_x + x2, center_y-y2 : center_y + y2] = -1000
        terrain.height_field_raw[center_x-x1 : center_x + x1, center_y-y1 : center_y + y1] = 0

    def pit_terrain_func(self, terrain, difficulty):
        depth = 1 * difficulty
        platform_size=4.
        depth = int(depth / terrain.vertical_scale)
        platform_size = int(platform_size / terrain.horizontal_scale / 2)
        x1 = terrain.length // 2 - platform_size
        x2 = terrain.length // 2 + platform_size
        y1 = terrain.width // 2 - platform_size
        y2 = terrain.width // 2 + platform_size
        terrain.height_field_raw[x1:x2, y1:y2] = -depth

    def flat_terrain_func(self, terrain, difficulty):
        terrain.height_field_raw[:] = 0.

    def rough_terrain_func(self, terrain, difficulty):
        max_height = 0.025 * difficulty / 0.9
        terrain.height_field_raw = np.random.uniform(-max_height*2-0.025, -0.025, terrain.height_field_raw.shape) / terrain.vertical_scale

    def smooth_slope_terrain_func(self, terrain, difficulty):
        slope = difficulty * 0.15
        random_01 = np.random.randint(0,2)
        down = random_01 * 2 - 1
        slope *= down
        terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=0.)

    def rough_slope_terrain_func(self, terrain, difficulty):
        slope = difficulty * 0.15
        random_01 = np.random.randint(0,2)
        down = random_01 * 2 - 1
        slope *= down
        terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=0.)
        terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005,
                                                downsampled_scale=0.2)

    def low_obst_terrain_func(self, terrain, difficulty):
        max_height = 0.03 * difficulty / 0.9
        obst_size = terrain.width // 10
        obst_num = 30
        xs = np.random.randint(0, terrain.length-obst_size, (obst_num,)) 
        ys = np.random.randint(0, terrain.width-obst_size, (obst_num,)) 
        terrain.height_field_raw[:] = 0.
        for i in range(obst_num):
            terrain.height_field_raw[xs[i]:xs[i]+obst_size, ys[i]:ys[i]+obst_size] = -max_height / terrain.vertical_scale

    def maze_terrain_func(self, terrain, difficulty):
        terrain.height_field_raw[:] = 1.0 / terrain.vertical_scale
        path_width = int((1.61 - difficulty * 1.0) / terrain.horizontal_scale)
        room_size = int(1.51 / terrain.horizontal_scale/2)
        midroom_size = int(2.01 / terrain.horizontal_scale/2) + path_width//2
        center_x = terrain.length // 2
        center_y = terrain.width // 2
        
        y_low = np.random.randint(-center_y, center_y-path_width, terrain.length)
        y_high = np.random.randint(-center_y, center_y-path_width, terrain.length)
        y_low, y_high = np.minimum(y_low, y_high), np.maximum(y_low, y_high) + path_width
        y_low[center_x-midroom_size:center_x+midroom_size] = - midroom_size
        y_high[center_x-midroom_size:center_x+midroom_size] = + midroom_size
        y_low[-room_size:] =  - room_size
        y_high[-room_size:] = + room_size
        y_low[:room_size] = - room_size
        y_high[:room_size] = + room_size
        for _col in range(0,terrain.length,path_width):
            if _col > path_width-1:
                if y_high[_col] < y_low[_col-path_width] + path_width: y_high[_col] = y_low[_col-path_width] + path_width
                if y_low[_col] > y_high[_col-path_width] - path_width: y_low[_col] = y_high[_col-path_width] - path_width
            terrain.height_field_raw[_col:_col+path_width, center_y+y_low[_col]:center_y+y_high[_col]] = 0.
        terrain.height_field_raw[ :room_size, center_y-room_size:center_y+room_size] = 0.
        terrain.height_field_raw[-room_size:, center_y-room_size:center_y+room_size] = 0.
        terrain.height_field_raw[ room_size:room_size+path_width, 2:-2] = 0.
        terrain.height_field_raw[-room_size-path_width:-room_size, 2:-2] = 0.
