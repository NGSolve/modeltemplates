from ngsolve import *
from ngs_templates.Shell import *
from math import pi
import ngsolve.meshes as meshes
from netgen.csg import *

loadfactor = Parameter(0)
order = 2

if True:
    thickness = 0.1
    length    = 12
    width     = 1

    E = 1.2e6
    nu = 0
    
    dirichlet = ["left","left|bottom","left","left"]
    bboundaryforce = None
    moments        = loadfactor*IfPos(x-length+1e-6, 1, 0)*50*pi/3

    strx = 12
    stry = 1
    mapping = lambda x,y,z : (length*x, width*y,0)
    mesh = meshes.MakeStructuredSurfaceMesh(False, strx, stry, mapping=mapping)
else:
     thickness = 0.03
     length    = 3.048
     radius    = 1.016
     
     E = 2.0685e7
     nu = 0.3
     
     dirichlet = ["right|sym","right|sym","right|sym","right|sym"]
     bboundaryforce = loadfactor*CoefficientFunction( (0,0,-600) )
     moments        = None

     geo       = CSGeometry()
     cyl       = Cylinder(Pnt(0,0,0), Pnt(1,0,0), radius)
     bot       = Plane(Pnt(0,0,0), Vec(0,0,-1))
     right     = Plane( Pnt(length,0,0), Vec(1,0,0))
     left      = Plane(Pnt(0,0,0), Vec(-1,0,0))
     finitecyl = cyl * bot * left * right
        
     geo.AddSurface(cyl, finitecyl)
     geo.AddPoint(Pnt(0,0,radius), "pntload")
     
     geo.NameEdge(cyl,bot, "sym")
     geo.NameEdge(cyl,left, "left")
     geo.NameEdge(cyl,right, "right")
     
     mesh = Mesh(geo.GenerateMesh(maxh=0.3))
     mesh.Curve(order)
     
Draw(mesh)

model = Shell(mesh=mesh, materiallaw=HookeShellMaterial(E,nu), \
              thickness=thickness, moments=moments, bboundaryforce=bboundaryforce, \
              order=order, dirichlet=dirichlet)

                       
Draw (model.displacement, mesh, "displ")
Draw (model.moments, mesh, "moments")
SetVisualization (deformation=True)

loadsteps = 20
for i in range(1, loadsteps+1):
    loadfactor.Set(i/loadsteps)
    print("Loadstep = ", i, " / ", loadsteps)
    with TaskManager():
        model.Solve()
    Redraw()
    input("key")

