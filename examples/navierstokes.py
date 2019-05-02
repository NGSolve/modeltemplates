from ngsolve import *
from ngs_templates.NavierStokes import *

from ngsolve.internal import visoptions

ngsglobals.msg_level = 6

from netgen.geom2d import SplineGeometry
geo = SplineGeometry()
geo.AddRectangle( (0, 0), (2, 0.41), bcs = ("wall", "outlet", "wall", "inlet"))
geo.AddCircle ( (0.2, 0.2), r=0.05, leftdomain=0, rightdomain=1, bc="cyl", maxh=0.02)
mesh = Mesh( geo.GenerateMesh(maxh=0.07))
mesh.Curve(3)


timestep = 0.002
navstokes = NavierStokes (mesh, nu=0.001, order=2, timestep = timestep,
                              inflow="inlet", outflow="outlet", wall="wall|cyl",
                              uin=CoefficientFunction( (1.5*4*y*(0.41-y)/(0.41*0.41), 0) ))
                              

navstokes.SolveInitial()

Draw (navstokes.pressure, mesh, "pressure")
Draw (navstokes.velocity, mesh, "velocity")
visoptions.scalfunction='velocity:0'

tend = 100
t = 0

with TaskManager(pajetrace=100*1000*1000):
    while t < tend:
        print (t)
        navstokes.DoTimeStep()
        t = t+timestep
        Redraw()

