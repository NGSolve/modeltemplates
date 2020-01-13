from ngsolve import *
from ngs_templates.Shell import *
from math import pi

thickness = 0.1
length    = 12
width     = 1

E = 1.2e6
nu = 0

clamp_bnd = ["left","left","left","left"]
free_bnd  = ["right|top|bottom","right|top|bottom","right|top|bottom","right|top|bottom"]

force = IfPos(x-length+1e-6, 1, 0)*50*pi/3
absforce = 50*pi/3


strx = 12
stry = 1
mapping = lambda x,y,z : (length*x, width*y,0)


mesh = MakeStructuredSurfaceMesh(False, strx, stry, mapping=mapping)


loadfactor = Parameter(0)

model = Shell(mesh=mesh, materiallaw=HookeShellMaterial(E,nu), \
                       thickness = thickness, \
                       moments=force, order=2)

                       
Draw (model.displacement, mesh, "displ")
Draw (model.moments, mesh, "moments")
SetVisualization (deformation=True)

loadsteps = 20
for i in range(1, loadsteps):
    loadfactor.Set(i/loadsteps)
    model.Solve()
    Redraw()
    input("key")
