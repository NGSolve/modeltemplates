from ngsolve import *
from netgen.geom2d import unit_square, SplineGeometry


geo = SplineGeometry()
pnums = [ geo.AddPoint (x,y,maxh=maxh) for x,y,maxh in [(0,0,0.01), (1,0,1), (1,0.1,1), (0,0.1,0.01)] ]
for p1,p2,bc in [(0,1,"bottom"), (1,2,"right"), (2,3,"top"), (3,0,"left")]:
    geo.AddSegment(["line", pnums[p1], pnums[p2]], bc=bc)
mesh = Mesh(geo.GenerateMesh(maxh=0.05))


# geo = SplineGeometry()
# geo.AddRectangle( (0, 0), (1, 0.1), bcs = ("bottom", "right", "top", "left"))
# mesh = Mesh( geo.GenerateMesh(maxh=0.05))

print (mesh.GetBoundaries())

from ngs_templates.Elasticity import *

loadfactor = Parameter(0)

bndforce = CoefficientFunction( [(0,0.1) if bnd=="right" else (0,0) for bnd in mesh.GetBoundaries()] )
model = Elasticity(mesh=mesh, materiallaw=NeoHookeMaterial(200,0.2), \
                       # dirichlet="left",
                       # volumeforce=loadfactor*CoefficientFunction((0,1)), \
                       # boundaryforce = { "right" : loadfactor*(0,1) },
                       boundarydisplacement = { "left" : (0,0), "right" : loadfactor*(0,1) },
                       nonlinear=True, order=4)

                       
Draw (model.displacement)
Draw (model.stress, mesh, "stress")
SetVisualization (deformation=True)

loadsteps = 20
for i in range(1, loadsteps):
    loadfactor.Set(i/loadsteps)
    model.Solve()
    Redraw()
    input("key")
