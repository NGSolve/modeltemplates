from ngs_templates.Elasticity import *
from netgen.meshing import *
from ngsolve import *

# mesh-file available from:
# https://nemesis.asc.tuwien.ac.at/index.php/s/fDdc6gn5bRRScJK
mesh = Mesh(ImportMesh("frame.unv"))
Draw(mesh)

model = Elasticity(mesh=mesh, materiallaw=HookeMaterial(200,0.2), \
                       dirichlet="holes",
                       boundaryforce = { "right" : (0,0,1), "left" : (0,0,-1) },
                       nonlinear=False, order=2)


Draw (model.displacement)
Draw (model.stress, mesh, "stress")

SetVisualization (deformation=True)

with TaskManager():
    model.Solve()
    Redraw()
    

myfes = H1(mesh, order=2)
normstress = GridFunction(myfes)
normstress.Set (Norm(model.stress))

SetVisualization (min=0, max=20, deformation=True)

Draw (normstress, mesh, "mises")

# Draw (BoundaryFromVolumeCF(Norm(model.stress)), mesh, "mises")

