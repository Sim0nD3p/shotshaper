from ..projectile import _SphericalParticleAirResistanceSpin
from numpy import exp


class SoccerBall(_SphericalParticleAirResistanceSpin):
    """
    Note that diameter can vary 110 mm to 130mm
    and 95 mm to 110 mm
    """
    def __init__(self, mass=0.430, diameter=0.22):
                    
        super(SoccerBall, self).__init__(mass, diameter)
    
    def drag_coefficient(self, velocity, omega):
        # Texture, sewing pattern and spin will alter
        # the drag coefficient.
        # Here, use correlation from
        
        # Goff, J. E., & Carré, M. J. (2010). Soccer ball lift 
        # coefficients via trajectory analysis. 
        # European Journal of Physics, 31(4), 775.
        
        
        vc = 12.19
        vs = 1.309
        
        S = omega*self.radius/velocity;
        if S > 0.05 and velocity > vc:
            Cd = 0.4127*S**0.3056;
        else:
            Cd = 0.155 + 0.346 / (1 + exp((velocity - vc)/vs))
        
        return Cd
    
    def lift_coefficient(self, Umag, omega):
        # TODO - complex dependency on Re and spin, skin texture etc
        return 0.9