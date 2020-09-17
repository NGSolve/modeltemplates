from ngsolve import *
from netgen.geom2d import SplineGeometry
from netgen.meshing import MeshingParameters
import pickle


from ngs_templates.Elasticity import *


from model import model
print (model)

mesh = Mesh(model["mesh"])

if model["type"] == "static":
    m = Elasticity(mesh=mesh, materiallaw=HookeMaterial(200,0.2),
                       boundarydisplacement = model["boundarydisplacement"], 
                       boundaryforce = model["boundaryforce"],
                       nonlinear=model["nonlinear"], order = model["order"])

    m.Solve()
    pickle.dump(m.displacement, open("displacement.pickle", "wb"))
    pickle.dump(m.displacement, open("stress.pickle", "wb"))
    
