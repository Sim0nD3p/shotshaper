# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 18:29:10 2021

@author: 2913452

TODO:
    - height of athlete, optimal angle shot put
        - note that biomech can influence
          how much force an athlete can emit for each angle
    - 
"""

from abc import ABC, abstractmethod
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from .transforms import T_12, T_23, T_34, T_14, T_41, T_31
import matplotlib.pyplot as pl
from numpy import exp,matmul,pi,sqrt,arctan2,radians,degrees,sin,cos,array,concatenate,linspace,zeros_like,cross,zeros,argmin
from numpy.linalg import norm
from . import environment
import os
import yaml

T_END = 60
N_STEP = 200

def hit_ground(t, y, *args): 
    return y[2]

def stopped(t, y, *args):
    U = norm(y[3:6])
    return U - 1e-4

class Shot:
    """Container for the results of the trajectory (position, velocity, time)
        attitude??
    """
    def __init__(self,t,x,v,att=None):
        self.time = t
        self.position = x
        self.velocity = v
        if att is not None:
            self.attitude = att
        

class _Projectile(ABC):
    """
    Methods:
        - Shoot
        - Advance (abstract)

    Args:
        ABC (_type_): _description_
    """
    def __init__(self):
        pass
   
    def _shoot(self, advance_function, y0, *args):
        """Solve the IVP and returns Shot object

        Args:
            advance_function (function): calculates forces and moment acting for given conditions
            y0 (_type_): _description_

        Returns:
            _type_: _description_
        """
        # Setup terminaison events for the solver
        hit_ground.terminal = True
        hit_ground.direction = -1
        stopped.terminal = True
        stopped.direction = -1
        
        # Solving ODE
        sol = solve_ivp(advance_function,[0,T_END],y0,
                        dense_output=True,args=args,
                        method='RK45',
                        events=(hit_ground,stopped))
        

        t = linspace(0,sol.t[-1],N_STEP)    # making vector with all results
        
        f = sol.sol(t)              # dealing with the solution from solve_ivp
        pos = array([f[0],f[1],f[2]])   # position vector
        vel = array([f[3],f[4],f[5]])   # velocity vector
        

        # Creating shot with solution of the problem
        if len(f) <= 6:
            shot = Shot(t, pos, vel)
        else:    
            att = array([f[6],f[7],f[8]])
            shot = Shot(t, pos, vel, att)
        
        return shot
         
    @abstractmethod
    def advance(self,t,vec,*args):
        """
        :param float T: Thrust
        :param float Q: Torque
        :param float P: Power
        :return: Right hand side of kinematic equations for a projectile
        :rtype: array
        """
        
class _Particle(_Projectile):

    """Used with SphericalParticules
    """
    def __init__(self):
        super().__init__()
        
        self.g = environment.g
        
    def initialize_shot(self, **kwargs):
        kwargs.setdefault('yaw', 0.0) 
        
        pitch = radians(kwargs["pitch"])
        yaw = radians(kwargs["yaw"])
        U = kwargs["speed"]
        xy = cos(pitch)
        u = U*xy*cos(yaw)
        w = U*sin(pitch)
        v = U*xy*sin(-yaw)
        if "position" in kwargs:
            x,y,z = kwargs["position"]
        else:
            x = 0.
            y = 0.
            z = 0.
        
        y0 = array((x,y,z,u,v,w))
        return y0
            
    def shoot(self, **kwargs):

        y0 = self.initialize_shot(**kwargs)
        shot = self._shoot(self.advance, y0)
        
        return shot
        
    def gravity_force(self, x=None):
        if x is None:
            return array((0,0,environment.g))
        else:
            # Messy way to return g array...
            f = zeros_like(x)
            f[0,:] = 0
            f[1,:] = 0
            f[2,:] = environment.g
            return f
        
    def advance(self, t, vec, *args):
        # x, y, z, u, v, w = vec
        x = vec[0:3]
        u = vec[3:6]
        
        f = self.gravity_force()
        
        return concatenate((u,f))
        
    

        

