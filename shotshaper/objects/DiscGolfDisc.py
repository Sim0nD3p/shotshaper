from .. import environment
from ..projectile import _Projectile
from scipy.interpolate import interp1d
import os
import yaml
from numpy import exp,matmul,pi,sqrt,arctan2,radians,degrees,sin,cos,array,concatenate,linspace,zeros_like,cross,zeros,argmin
import matplotlib.pyplot as pl
from ..transforms import T_12, T_23, T_34, T_14, T_41, T_31
from numpy.linalg import norm

class DiscGolfDisc(_Projectile):
    def __init__(self, name, mass=0.175):
        this_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(this_dir, 'discs', name + '.yaml')
    
        self.name = name
        
        with open(path, 'r') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
            
            self.diameter = data['diameter']
            self.mass = mass
            self.weight = environment.g*mass
            self.area = pi*self.diameter**2/4.0
            self.I_xy = mass*data['J_xy']
            self.I_z = mass*data['J_z']
            
            a = array(data['alpha'])
            cl = array(data['Cl'])
            cd = array(data['Cd'])
            cm = array(data['Cm'])
            
        
        self._alpha,self._Cl,self._Cd,self._Cm = self._flip(a,cl,cd,cm)
        kind = 'linear'
        self.Cl_func = interp1d(self._alpha, self._Cl, kind=kind)
        self.Cd_func = interp1d(self._alpha, self._Cd, kind=kind)
        self.Cm_func = interp1d(self._alpha, self._Cm, kind=kind)
        
    def _flip(self,a,cl,cd,cm):
        """
        Data given from -90 deg to 90 deg.
        Expand to -180 to 180 using symmetry considerations.
        """
        n = len(a)

        idx = argmin(abs(a))
        a2 = zeros(2*n)
        cl2 = zeros(2*n)
        cd2 = zeros(2*n)
        cm2 = zeros(2*n)
        
        a2[idx:idx+n] = a[:]
        cl2[idx:idx+n] = cl[:]
        cd2[idx:idx+n] = cd[:]
        cm2[idx:idx+n] = cm[:]
        for i in range(idx):
            a2[i] = -(180 + a[idx-i])
            cl2[i] = -cl[idx-i]
            cd2[i] =  cd[idx-i]
            cm2[i] = -cm[idx-i]
        
        for i in range(idx+n,2*n):
            a2[i] = 180 - a[idx+n-i-2]
            cl2[i] = -cl[idx+n-i-2]
            cd2[i] =  cd[idx+n-i-2]
            cm2[i] = -cm[idx+n-i-2]
        
        return a2,cl2,cd2,cm2
        
    def _normalize_angle(self, alpha):
        """
        Ensure that the angle fulfils :math:`-\\pi < \\alpha < \\pi`

        :param float alpha: Angle in radians
        :return: Normalized angle
        :rtype: float
        """

        return arctan2(sin(alpha), cos(alpha))
    
    def Cd(self, alpha): 
        """
        Provide drag coefficent for a given angle of attack.

        :param float alpha: Angle in radians
        :return: Drag coefficient
        :rtype: float
        """
        
        # NB! The stored data uses degrees for the angle
        return self.Cd_func(degrees(self._normalize_angle(alpha)))

    def Cl(self, alpha): 
        """
        Provide drag coefficent for a given angle of attack.

        :param float alpha: Angle in radians
        :return: Drag coefficient
        :rtype: float
        """
        
        # NB! The stored data uses degrees for the angle
        return self.Cl_func(degrees(self._normalize_angle(alpha)))

    def Cm(self, alpha): 
        """
        Provide coefficent of moment for a given angle of attack.

        :param float alpha: Angle in radians
        :return: Coefficient of moment
        :rtype: float
        """
    
        # NB! The stored data uses degrees for the angle
        return self.Cm_func(degrees(self._normalize_angle(alpha)))


    def plot_coeffs(self, color='k'):
        """
        Utility function to quickly explore disc coefficients.

        :param string color: Matplotlib color key. Default value is k, i.e. black.
        """
        pl.plot(self._alpha, self._Cl, 'C0-o',label='$C_L$')
        pl.plot(self._alpha, self._Cd, 'C1-o',label='$C_D$')
        pl.plot(self._alpha, 3*self._Cm, 'C2-o',label='$C_M$')
        
        a = linspace(-pi,pi,200)
        #pl.plot(degrees(a), self.Cl(a), 'C0-',label='$C_L$')
        #pl.plot(degrees(a), self.Cd(a), 'C1-',label='$C_D$')
        #pl.plot(degrees(a), 3*self.Cm(a), 'C2-',label='$C_M$')
        
        pl.xlabel('Angle of attack ($^\circ$)')
        pl.ylabel('Aerodynamic coefficients (-)')
        pl.legend(loc='upper left')
        ax = pl.gca()
        ax2 = pl.gca().twinx()
        ax2.set_ylabel("Aerodynamic efficiency, $C_L/C_D$")
        pl.plot(self._alpha, self._Cl/self._Cd, 'C3-.',label='$C_L/C_D$')
        ax2.legend(loc='upper right')
        
        return ax,ax2
    
    
    def empirical_spin(self, speed):
        # Simple empirical formula for spin rate, based on curve-fitting
        # data from:
        # https://www.dgcoursereview.com/dgr/forums/viewtopic.php?f=2&t=7097
        #omega = -0.257*speed**2 + 15.338*speed

        # Alternatively, experiments indicate a linear relationship,
        omega = 5.2*speed

        return omega

            
    
    def initialize_shot(self, **kwargs):
        U = kwargs["speed"]
        
        kwargs.setdefault('yaw', 0.0) 
        #kwargs.setdefault('omega', self.empirical_spin(U)) 
        
        pitch = radians(kwargs["pitch"])
        yaw = radians(kwargs["yaw"])
        omega = kwargs["omega"]
        
        # phi, theta
        roll_angle = radians(kwargs["roll_angle"]) # phi
        nose_angle = radians(kwargs["nose_angle"]) # theta
        # psi, rotation around z irrelevant for starting position
        #      since the disc is symmetric
        
        # Initialize position
        if "position" in kwargs:
            x,y,z = kwargs["position"]
        else:
            x = 0.
            y = 0.
            z = 0.
        
        # Initialize velocity
        xy = cos(pitch)
        u = U*xy*cos(yaw)
        v = U*xy*sin(-yaw)
        w = U*sin(pitch)
        
        # Initialize angles
        attitude = array([roll_angle, nose_angle, 0])
        # The initial orientation of the disc must also account for the
        # angle of the throw itself, i.e. the launch angle. 
        attitude += matmul(T_12(attitude), array((0, pitch, 0)))
        
        #attitude = matmul(T_23(yaw),attitude)
        #attitude += matmul(T_12(attitude), array((0, pitch, 0)))
        phi, theta, psi = attitude
        y0 = array((x,y,z,u,v,w,phi,theta,psi))
        return y0, omega
            
    def shoot(self, **kwargs):

        y0, omega = self.initialize_shot(**kwargs)
               
        shot = self._shoot(self.advance, y0, omega)
        
        return shot
    
    def post_process(self, s, omega):
        n = len(s.time)
        alphas = zeros(n)
        betas = zeros(n)
        lifts = zeros(n)
        drags = zeros(n)
        moms = zeros(n)
        rolls = zeros(n)
        for i in range(n):
            x = s.position[:,i]
            u = s.velocity[:,i]
            a = s.attitude[:,i]
            
            alpha, beta, Fd, Fl, M, g4 = self.forces(x, u, a, omega)
            
            alphas[i] = alpha
            betas[i] = beta
            lifts[i] = Fl
            drags[i] = Fd
            moms[i] = M
            rolls[i] = -M/(omega*(self.I_xy - self.I_z))
        
        arc_length = norm(s.position, axis=0)
        return arc_length,degrees(alphas),degrees(betas),lifts,drags,moms,degrees(rolls)
            
    def forces(self, x, u, a, omega):
        # Velocity in body axes
        urel = u - environment.wind_abl(x[2])
        u2 = matmul(T_12(a), urel)
        # Side slip angle is the angle between the x and y velocity
        beta = -arctan2(u2[1], u2[0])
        # Velocity in zero side slip axes
        u3 = matmul(T_23(beta), u2)
        # Angle of attack is the angle between 
        # vertical and horizontal velocity
        alpha = -arctan2(u3[2], u3[0])
        # Velocity in wind system, where forces are to be calculated
        u4 = matmul(T_34(alpha), u3)
        
        # Convert gravitational force from Earth to Wind axes
        g = array((0, 0, self.mass*environment.g))
        g4 = T_14(g, a, beta, alpha)
        
        # Aerodynamic forces
        q = 0.5*environment.rho*u4[0]**2
        S = self.area
        D = self.diameter
        
        Fd = q*S*self.Cd(alpha)
        Fl = q*S*self.Cl(alpha)
        M  = q*S*D*self.Cm(alpha)
        
        return alpha, beta, Fd, Fl, M, g4
        
    def advance(self, t, vec, omega):
        x = vec[0:3]
        u = vec[3:6]
        a = vec[6:9]
        
        alpha, beta, Fd, Fl, M, g4 = self.forces(x, u, a, omega)
        
        m = self.mass
        # Calculate accelerations
        dudt = (-Fd + g4[0])/m
        dvdt =        g4[1]/m
        dwdt = ( Fl + g4[2])/m
        acc4 = array((dudt,dvdt,dwdt))
        # Roll rate acts around x-axis (in axes 3: zero side slip axes)
        dphidt = -M/(omega*(self.I_xy - self.I_z))
        # Other angular rotations are ignored, assume zero wobble
        angvel3 = array((dphidt, 0, 0))
        
        acc1 = T_41(acc4, a, beta, alpha)
        angvel1 = T_31(angvel3, a, beta)
        
        return concatenate((u,acc1,angvel1)) 

    
    

    