from ngsolve import *
from ngs_templates.Transport import *

from netgen.geom2d import unit_square
mesh = Mesh( unit_square.GenerateMesh(maxh=0.1))

timestep = 3e-3
transport = TransportEquation (mesh, order=6, wind = (y-0.5, -x+0.5), timestep=timestep)

transport.SetInitial( exp (-100* ((x-0.8)**2 + (y-0.5)**2) ) )

Draw (transport.concentration, mesh, "c")

tend = 0.1
t = 0

SetVisualization (min=0, max=1)

with TaskManager(pajetrace=100*1000*1000):
    while t < tend:
        print (t)
        transport.DoTimeStep()
        t = t+timestep
        Redraw()

