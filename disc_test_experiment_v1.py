# -*- coding: utf-8 -*-
"""
Example showing a single disc throw.
"""

from shotshaper.objects.DiscGolfDisc import DiscGolfDisc
import matplotlib.pyplot as pl
import numpy as np


# Defining the disc and initial conditions
d = DiscGolfDisc('dd2')
U = 24.2
omega = 116.8   # spin rate of the disc
z0 = 1.3
pos = np.array((0,0,z0))
pitch = 5
nose = 10
roll = 0
yaw = 0

# Creating the shot (shoot method of DiscGolfDisc)
    # added yaw, default to 0 if not presented (2024-01-28)
shot = d.shoot(speed=U, omega=omega, pitch=pitch, 
               position=pos, nose_angle=nose, roll_angle=roll, yaw=yaw)

# Aller chercher les proprietes du disque

fig_specs, axs = pl.subplots(1, 1, figsize=(4, 4))
#axs.plot(d.Cl_func)
#fig=fig_specs
d.plot_coeffs()
d.plot_coeff_fun()



# Plot trajectory
fig, (ax1, ax2) = pl.subplots(2, 1, figsize=(8, 6))
x,y,z = shot.position
ax1.plot(x,y)
ax1.set_title('Vue de haut')

ax2.plot(x, z)
ax2.set_title('Vue de cote')

pl.tight_layout()

pl.xlabel('Distance (m)')
pl.ylabel('Drift (m)')
pl.axis('equal')

# Plot other parameters
arc,alphas,betas,lifts,drags,moms,rolls = d.post_process(shot, omega)
fig, axes = pl.subplots(nrows=2, ncols=3, dpi=80,figsize=(13,5))

axes[0,0].plot(arc, lifts)
axes[0,0].set_xlabel('Distance (m)')
axes[0,0].set_ylabel('Lift force (N)')

axes[0,1].plot(arc, drags)
axes[0,1].set_xlabel('Distance (m)')
axes[0,1].set_ylabel('Drag force (N)')

axes[0,2].plot(arc, moms)
axes[0,2].set_xlabel('Distance (m)')
axes[0,2].set_ylabel('Moment (Nm)')

axes[1,0].plot(arc, alphas)
axes[1,0].set_xlabel('Distance (m)')
axes[1,0].set_ylabel('Angle of attack (deg)')

axes[1,1].plot(arc, shot.velocity[0,:])
axes[1,1].plot(arc, shot.velocity[1,:])
axes[1,1].plot(arc, shot.velocity[2,:])
axes[1,1].set_xlabel('Distance (m)')
axes[1,1].set_ylabel('Velocities (m/s)')
axes[1,1].legend(('u','v','w'))

axes[1,2].plot(arc, rolls)
axes[1,2].set_xlabel('Distance (m)')
axes[1,2].set_ylabel('Roll rate (rad/s)')
pl.tight_layout()

pl.show()
