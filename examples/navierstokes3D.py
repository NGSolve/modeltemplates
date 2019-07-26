from ngsolve import *
from ngs_templates.NavierStokesSIMPLE import *

from ngsolve.internal import visoptions

ngsglobals.msg_level = 8

from netgen.csg import *
geo = CSGeometry()
channel = OrthoBrick( Pnt(-1, 0, 0), Pnt(3, 0.41, 0.41) ).bc("wall")
inlet = Plane (Pnt(0,0,0), Vec(-1,0,0)).bc("inlet")
outlet = Plane (Pnt(2.5, 0,0), Vec(1,0,0)).bc("outlet")
cyl = Cylinder(Pnt(0.5, 0.2,0), Pnt(0.5,0.2,0.41), 0.05).bc("wall")
fluiddom = channel*inlet*outlet-cyl
geo.Add(fluiddom)
mesh = Mesh( geo.GenerateMesh(maxh=0.1))
mesh.Curve(3)
Draw(mesh)

# SetNumThreads(1)
SetHeapSize(100*1000*1000)
timestep = 0.001

with TaskManager(pajetrace=100*1000*1000):
  navstokes = NavierStokes (mesh, nu=0.001, order=3, timestep = timestep,
                              inflow="inlet", outflow="outlet", wall="wall|cyl",
                              uin=CoefficientFunction( (16*y*(0.41-y)*z*(0.41-z)/(0.41*0.41*0.41*0.41), 0, 0) ))
                              
print ("ndof =", navstokes.X.ndof)

with TaskManager(pajetrace=100*1000*1000):
    navstokes.SolveInitial()

Draw (navstokes.pressure, mesh, "pressure", draw_surf=False)
Draw (navstokes.velocity, mesh, "velocity")
visoptions.scalfunction='velocity:0'

tend = 10
t = 0

with TaskManager(pajetrace=100*1000*1000):
    while t < tend:
        print (t)
        navstokes.DoTimeStep()
        t = t+timestep
        Redraw()

