from ..projectile import _Projectile
from shotshaper.SphericalParticules import  _SphericalParticleAirResistance

class ShotPutBall(_SphericalParticleAirResistance):
    """
    Note that diameter can vary 110 mm to 130mm
    and 95 mm to 110 mm
    """
    def __init__(self, weight_class):
        
        if weight_class == 'M':
            mass = 7.26
            diameter = 0.11
        elif weight_class == 'F':
            mass = 4.0
            diameter = 0.095
            
        super().__init__(mass, diameter)