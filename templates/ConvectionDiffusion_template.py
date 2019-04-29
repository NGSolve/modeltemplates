from ngsolve import *
from mt_global import *


class ConvectionDiffusionEquation:
    
    def __init__(self, mesh, wind, lam, timestep, order=2, dirichlet="", udir=0, source=None):

        wind = CoefficientFunction(wind)
        self.timestep = timestep
        self.udir = udir
        self.source = source
        
        fesT = L2(mesh, order=order, all_dofs_together=True)
        fesF = FacetFESpace(mesh, order=order, dirichlet=dirichlet)  # all_dofs_together=True -> buggy
        fes = FESpace([fesT,fesF])
        u,uh = fes.TrialFunction()
        v,vh = fes.TestFunction()

        self.bfconv = BilinearForm(fes, nonassemble=True)
        self.bfconv += SymbolicBFI (-u * wind * grad(v))
        n = specialcf.normal(mesh.dim)
        h = specialcf.mesh_size
        wn = wind*n
        self.bfconv += SymbolicBFI ( IfPos(wn, wn*u, wn*u.Other(bnd=udir)) * v, element_boundary=True)

        self.bfdiff = BilinearForm(fes)
        self.bfdiff += SymbolicBFI (lam*grad(u)*grad(v))
        self.bfdiff += SymbolicBFI (lam*(grad(u)*n*(vh-v)+grad(v)*n*(uh-u)+5*(order+1)**2/h*(u-uh)*(v-vh)), element_boundary=True)
        self.bfdiff.Assemble()

        self.bfmstar = BilinearForm(fes, symmetric=True)
        self.bfmstar += SymbolicBFI (timestep*lam*grad(u)*grad(v)+u*v)
        self.bfmstar += SymbolicBFI (timestep*lam*(grad(u)*n*(vh-v)+grad(v)*n*(uh-u)+5*(order+1)**2/h*(u-uh)*(v-vh)), element_boundary=True)
        self.bfmstar.Assemble()
        
        self.invmstar = self.bfmstar.mat.Inverse(freedofs=fes.FreeDofs(), inverse="sparsecholesky")
        self.gfu = GridFunction(fes)
    
    def SetInitial(self,u0):
        self.gfu.components[0].Set (u0)
        self.gfu.components[1].Set (u0, BND)

    def DoTimeStep(self):
        temp  = self.bfdiff.mat.CreateColVector()
        temp.data = self.bfdiff.mat * self.gfu.vec + self.bfconv.mat * self.gfu.vec
        self.gfu.vec.data -= self.timestep*self.invmstar *temp

    def Concentration(self):
        return self.gfu.components[0]
