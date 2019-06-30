from ngsolve import *
from netgen.geom2d import SplineGeometry
from netgen.meshing import MeshingParameters


geo = SplineGeometry()
geo.AddRectangle( (0, 0), (1, 0.1), bcs = ("bottom", "right", "top", "left"))

mp = MeshingParameters(maxh=0.05)
mp.RestrictH(0,0,0, h=0.01)
mp.RestrictH(0,0.1,0, h=0.01)
mesh = Mesh( geo.GenerateMesh(mp=mp))



from ngs_templates.Elasticity import *

loadfactor = Parameter(0)

model = Elasticity(mesh=mesh, materiallaw=NeoHookeMaterial(200,0.2), \
                       # dirichlet="left",
                       # volumeforce=loadfactor*CoefficientFunction((0,1)), \
                       boundarydisplacement = { "left" : (0,0) },
                       boundaryforce = { "right" : loadfactor*(0,1) },
                       # boundarydisplacement = { "left" : (0,0), "right" : loadfactor*(0,1) },
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
