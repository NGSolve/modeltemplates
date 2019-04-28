from ngsolve import *
import sys
sys.path.insert(0,'../templates')
from modeltemplates import *


from netgen.geom2d import unit_square
mesh = Mesh( unit_square.GenerateMesh(maxh=0.1))

timestep = 10e-3
transport = TransportEquation (mesh, order=3, wind = (y-0.5, -x+0.5), timestep=timestep)

transport.SetInitial( exp (-400* ((x-0.8)**2 + (y-0.5)**2) ) )

Draw (transport.Concentration(), mesh, "c")

tend = 100
t = 0

SetVisualization (min=0, max=1)

with TaskManager(pajetrace=100*1000*1000):
    while t < tend:
        print (t)
        transport.DoTimeStep()
        t = t+timestep
        Redraw()

