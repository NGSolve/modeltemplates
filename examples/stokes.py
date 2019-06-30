from ngsolve import *
from ngs_templates.Stokes import *

from ngsolve.internal import visoptions

ngsglobals.msg_level = 6
SetHeapSize(100*1000*1000)

dim = 2

if dim == 2:
    from netgen.geom2d import SplineGeometry
    geo = SplineGeometry()
    geo.AddRectangle( (0, 0), (2, 0.41), bcs = ("wall", "outlet", "wall", "inlet"))
    geo.AddCircle ( (0.2, 0.2), r=0.05, leftdomain=0, rightdomain=1, bc="cyl", maxh=0.02)
    mesh = Mesh( geo.GenerateMesh(maxh=0.07))
    mesh.Curve(3)
    uin = CoefficientFunction( (1.5*4*y*(0.41-y)/(0.41*0.41), 0) )
else:
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
    uin=CoefficientFunction( (16*y*(0.41-y)*z*(0.41-z)/(0.41*0.41*0.41*0.41), 0, 0))  
    

stokes = Stokes (mesh, nu=0.001, order=2, 
                     inflow="inlet", outflow="outlet", wall="wall|cyl",
                     uin=uin)
                              
print ("ndof = ", stokes.X.ndof)

with TaskManager():
    stokes.Solve()

Draw (stokes.pressure, mesh, "pressure")
Draw (stokes.velocity, mesh, "velocity")
visoptions.scalfunction='velocity:0'

