from ngsolve import *
import sys
sys.path.insert(0,'../templates')
from modeltemplates import *
from netgen.geom2d import *
from ngsolve.internal import visoptions
import math

geo = SplineGeometry()
geo.AddRectangle( (0,0), (0.06, 0.01), bcs=['b','r','t','l'])
mesh = Mesh(geo.GenerateMesh(maxh=0.002))

Draw (mesh)


timestep = 0.5
navstokes = NavierStokes (mesh, nu=1.04177e-6, order=3, timestep = timestep,
                              inflow="", outflow="", wall="l|r|b|t", uin=(0,0) )

T0 = 293
Tinitial = 293.5-50*y+y*(0.01-y)*1e3*sin(20/0.06*x*math.pi)

convdiff = ConvectionDiffusionEquation (mesh, order=3, lam=1.38e-7, wind = navstokes.Velocity(), dirichlet="b|t", udir=Tinitial, timestep=timestep)


convdiff.SetInitial(Tinitial)

beta = 2.07e-4


navstokes.AddForce ( (1-beta*(convdiff.Concentration()-T0))*(0, -9.81))

navstokes.SolveInitial()


Draw (navstokes.Pressure(), mesh, "pressure")
Draw (navstokes.Velocity(), mesh, "velocity")
visoptions.scalfunction='velocity:0'
Draw (convdiff.Concentration(), mesh, "temp")

input ("key")

tend = 1000
t = 0

with TaskManager(pajetrace=100*1000*1000):
    while t < tend:
        print (t)
        navstokes.DoTimeStep()
        convdiff.DoTimeStep()
        t = t+timestep
        Redraw()
        # input ("key")

                            



