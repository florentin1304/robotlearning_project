"""Implementation of the Hopper environment supporting
domain randomization optimization."""
import csv
import pdb
from copy import deepcopy

import numpy as np
import gym
from gym import utils
from .mujoco_env import MujocoEnv
from scipy.stats import truncnorm

class CustomHopper(MujocoEnv, utils.EzPickle):
    def __init__(self, domain=None, mode=None):
        MujocoEnv.__init__(self, 4)
        utils.EzPickle.__init__(self)
        self.domain = domain
        self.mode = mode

        self.original_masses = np.copy(self.sim.model.body_mass[1:])    # Default link masses
        self.min_mass = 0.1
        if self.domain == 'source':  # Source environment has an imprecise torso mass (1kg shift)
            self.sim.model.body_mass[1] -= 1.0
        
        self.unrandomized_masses = np.copy(self.sim.model.body_mass[1:])

        # self.delta = None
        # self.perc = None
        # self.mean = None
        # self.vars = None

    def get_name(self):
        name = ""

        if self.domain == "source":
            if self.mode == "udr":
                name = "udr"
                for p in self.delta:
                    name += "_" + str(p).replace(".", "-")
                if self.perc:
                    name += "_perc"

            elif self.mode == "Gauss":
                name = "gauss"
            
            else:
                name = "source"
            
        if self.domain == "target":
            name = "target"
    
        return name

    def set_random_parameters(self):
        """Set random masses
        TODO
        """
        if self.mode == 'udr' and self.perc is None:
            raise Exception('must call set_udr_delta in order to use sample_parameters()')        
        if self.mode == 'Gauss' and self.mean is None:
            raise Exception('must call set_Gaussian_mean_var in order to use sample_parameters()')
        
        self.set_parameters(self.sample_parameters())

    def set_udr_delta(self, delta=1, perc=False):
        if self.mode != 'udr':
            raise Exception('wrong environment !!! You should use CustomHopper-udr-v0')
        if isinstance(delta, float):
            self.delta = delta*np.ones((3,))
        elif isinstance(delta, np.ndarray):
            if delta.size == 3:
                self.delta = delta
            else:
                raise Exception(f"Delta size not compliant, should be 3")
        else:
            raise Exception(f"Delta type in set_udr_delta is not compliant {type(delta)}")
        
        self.perc = perc
        if self.perc:
            for d in self.delta:
                if not (0 <= d <= 1):
                    raise Exception("Deltas in percentile not in (0,1) range")

    def set_Gaussian_mean_var(self, mean, var):
        if self.mode != 'Gauss':
            raise Exception('wrong environment !!! You should use CustomHopper-Gauss-v0')
        if len(mean) != len(var):
            raise Exception('mean and var have different lengths')
        self.mean = mean
        self.var = var

    def sample_parameters(self):
        """Sample masses according to a domain randomization distribution
        TODO
        """
        new_masses = np.copy(self.unrandomized_masses)

        if self.mode == 'udr':
            if self.perc :
                for i in range(1, len(new_masses)): #start from index 1 because index 0 is torso mass
                    lb = new_masses[i] * (1-self.delta[i-1])
                    ub = new_masses[i] * (1+self.delta[i-1])

                    new_value = float('-inf')
                    while new_value < self.min_mass:
                        new_value = np.random.uniform(lb, ub)
                        # print(new_value, lb, ub)
                    new_masses[i] = new_value
            else:
                for i in range(1, len(new_masses)): #start from index 1 because index 0 is torso mass
                    lb = new_masses[i] - self.delta[i-1]
                    ub = new_masses[i] + self.delta[i-1]

                    new_value = float('-inf')
                    while new_value < self.min_mass:
                        new_value = np.random.uniform(lb, ub)
                        # print(new_value, lb, ub)

                    new_masses[i] = new_value
        elif self.mode == 'Gauss':
            for mass in range(1, len(new_masses)): #start from index 1 because index 0 is torso mass

                new_value = float('-inf')
                while new_value < self.min_mass:
                    new_value = np.random.normal( self.mean[mass-1], np.sqrt(self.var[mass-1]))

                new_masses[mass] = new_value

        return new_masses

    def get_parameters(self):
        """Get value of mass for each link"""
        masses = np.array( self.sim.model.body_mass[1:] )
        return masses

    def set_parameters(self, task):
        """Set each hopper link's mass to a new value"""
        self.sim.model.body_mass[1:] = task

    def step(self, a):
        """Step the simulation to the next timestep

        Parameters
        ----------
        a : ndarray,
            action to be taken at the current timestep
        """
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and (height > .7) and (abs(ang) < .2))
        ob = self._get_obs()

        return ob, reward, done, {}

    def _get_obs(self):
        """Get current state"""
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat
        ])

    def reset_model(self):
        """Reset the environment to a random initial state"""
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
    
        if self.mode is not None:
            self.set_random_parameters()
    
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20



"""
    Registered environments
"""
gym.envs.register(
        id="CustomHopper-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
)

gym.envs.register(
        id="CustomHopper-source-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "source"}
)

gym.envs.register(
        id="CustomHopper-target-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "target"}
)

gym.envs.register(
        id="CustomHopper-udr-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "source",
                "mode": 'udr'}
)

gym.envs.register(
        id="CustomHopper-Gauss-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "source",
                "mode": 'Gauss'}
)
