from ngsolve import *
import sys
sys.path.insert(0,'../templates')
from modeltemplates import *


from netgen.geom2d import unit_square
mesh = Mesh( unit_square.GenerateMesh(maxh=0.1))

timestep = 10e-3
convdiff = ConvectionDiffusionEquation (mesh, order=3, lam=1e-3, wind = (y-0.5, -x+0.5), dirichlet=".*", timestep=timestep)

convdiff.SetInitial( exp (-100* ((x-0.8)**2 + (y-0.5)**2) ) )
Draw (convdiff.Concentration().components[0], mesh, "c")

tend = 100
t = 0

SetVisualization (min=0, max=1)
input ("key")

with TaskManager(pajetrace=100*1000*1000):
    while t < tend:
        print (t)
        convdiff.DoTimeStep()
        t = t+timestep
        Redraw()

