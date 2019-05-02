from ngsolve import *
from netgen.geom2d import unit_square, SplineGeometry

geo = SplineGeometry()
geo.AddRectangle( (0, 0), (1, 0.1), bcs = ("bottom", "right", "top", "left"))
mesh = Mesh( geo.GenerateMesh(maxh=0.1))

from ngs_templates.Elasticity import *

loadfactor = Parameter(0)

model = Elasticity(mesh=mesh, materiallaw=HookMaterial(200,0.2), dirichlet="left", volumeforce=loadfactor*CoefficientFunction((0,1)), nonlinear=True)

Draw (model.displacement)
SetVisualization (deformation=True)

for i in range(1, 10):
    loadfactor.Set(i)
    model.Solve()
    Redraw()
    input("key")
