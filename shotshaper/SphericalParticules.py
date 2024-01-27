from . import environment
from .projectile import _Particle
from numpy.linalg import norm
from numpy import pi, cross, array, concatenate, zeros_like



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



        
        

 