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
    def __init__(self,t,x,v,att=None):
        self.time = t
        self.position = x
        self.velocity = v
        if att is not None:
            self.attitude = att
        

class _Projectile(ABC):
    def __init__(self):
        pass
   
    def _shoot(self, advance_function, y0, *args):
        hit_ground.terminal = True
        hit_ground.direction = -1
        stopped.terminal = True
        stopped.direction = -1
        
        sol = solve_ivp(advance_function,[0,T_END],y0,
                        dense_output=True,args=args,
                        method='RK45',
                        events=(hit_ground,stopped))
        
        t = linspace(0,sol.t[-1],N_STEP)
        
        f = sol.sol(t)
        pos = array([f[0],f[1],f[2]])
        vel = array([f[3],f[4],f[5]])
        
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
        
    
class _SphericalParticleAirResistance(_Particle):
    def __init__(self, mass, diameter):
        super().__init__()
        
        self.mass = mass
        self.diameter = diameter
        self.radius = 0.5*diameter
        self.area = 0.25*pi*diameter**2
        self.volume = 4./3.*pi*self.radius**3
        
    
    def air_resistance_force(self, U, Cd):
        
        f = -0.5*environment.rho*self.area*Cd*norm(U)*U/self.mass
        #f = -0.5*environment.rho*self.area*Cd*Umag*U/self.mass
        
        return f
    
    def advance(self, t, vec, *args):
        x = vec[0:3]
        u = vec[3:6]
        
        Cd = self.drag_coefficient(norm(u))
        
        f = self.air_resistance_force(u, Cd) \
          + self.gravity_force()
        
        return concatenate((u,f))
       
        
    def reynolds_number(self, velocity):
        """
        Reynolds number, non-dimensional number giving the 
        ratio of inertial forces to viscous forces. Used
        for calculating the drag coefficient.
        
        :param float velocity: Velocity seen by particle
        :return: Reynolds number
        :rtype: float
        
        """
        return environment.rho*velocity*self.diameter/environment.mu
    
    def drag_coefficient(self, velocity):
        """
        Drag coefficient for sphere, empirical curve fit
        taken from:
        
        F. A. Morrison, An Introduction to Fluid Mechanics, (Cambridge
        University Press, New York, 2013). This correlation appears in
        Figure 8.13 on page 625. 

        The full formula is:

        .. math::
            F = \\frac{2}{\\pi}\\cos^{-1}e^{-f} \\\\
            f = \\frac{B}{2}\\frac{R-r}{r\\sin\\phi}


        :param float velocity: Velocity seen by particle
        :return: Drag coefficient
        :rtype: float
        """
    
        Re = self.reynolds_number(velocity)
        
        if Re <= 0:
            return 1e30
        
        tmp1 = Re/5.0
        tmp2 = Re/2.63e5
        tmp3 = Re/1e6
        
        Cd = 24.0/Re \
           + 2.6*tmp1/(1 + tmp1**1.52) \
           + 0.411*tmp2**-7.94/(1 + tmp2**-8) \
           + 0.25*tmp3/(1 + tmp3) 
           
        return Cd
    

class _SphericalParticleAirResistanceSpin(_SphericalParticleAirResistance):
    def __init__(self, mass, diameter):
        super().__init__(mass, diameter)
        
        
    def lift_coefficient(self, Umag, omega):
        # TODO - complex dependency on Re. For now,
        #        assume constant
        return 0.9
        
    def shoot(self, **kwargs):
        y0 = self.initialize_shot(**kwargs)
        spin = array((kwargs["spin"]))
        
        shot = self._shoot(self.advance, y0, spin)
        
        return shot        
    
    def spin_force(self,U,spin):
        
        Umag = norm(U)
        omega = norm(spin)
        
        Cl = self.lift_coefficient(Umag, omega)
        
        if U.ndim == 1:
            f = Cl*pi*self.radius**3*environment.rho*cross(spin, U)/self.mass
        else:
            # Messy way to return spin array for post-processing
            f = zeros_like(U)
            for i in range(U.shape[1]):
                f[:,i] = Cl*pi*self.radius**3*environment.rho*cross(spin, U[:,i])/self.mass
        
        return f
    
    def advance(self, t, vec, spin):
        x = vec[0:3]
        u = vec[3:6]
        
        Cd = self.drag_coefficient(norm(u), norm(spin))
        
        f = self.air_resistance_force(u, Cd) \
          + self.gravity_force() \
          + self.spin_force(u,spin)
        
        return concatenate((u,f))



        
        

 

        
        

