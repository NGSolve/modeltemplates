from ngsolve import *
from .mt_global import *

__all__ = ["TransportEquation"]

class TransportEquation:
    
    def __init__(self, mesh, wind, timestep, order=2, uin=0, source=None):

        wind = CoefficientFunction(wind)
        self.timestep = timestep
        self.uin = uin
        self.source = source
        
        fes = L2(mesh, order=order, all_dofs_together=True)
        u,v = fes.TnT()
        
        usetrace = True
        if not usetrace:
            self.bfa = BilinearForm(fes, nonassemble=True)
            self.bfa += -u * wind * grad(v) * dx
            wn = wind*specialcf.normal(mesh.dim)
            self.bfa += (wn * IfPos(wn, u, u.Other(bnd=uin)) * v).Compile(True, wait=True) * dx(element_boundary=True)
            aop = self.bfa.mat
        else:
            fes_trace = Discontinuous(FacetFESpace(mesh, order=order))
            utr,vtr = fes_trace.TnT()
            trace = fes.TraceOperator(fes_trace, False)
            
            self.bfa = BilinearForm(fes, nonassemble=True)
            self.bfa += -u * wind * grad(v) * dx

            self.bfa_trace = BilinearForm(fes_trace, nonassemble=True)
            wn = wind*specialcf.normal(mesh.dim)
            self.bfa_trace += (wn * IfPos(wn, utr, utr.Other(bnd=uin)) * vtr).Compile(True,wait=True) * dx(element_boundary=True)

            aop = self.bfa.mat + trace.T @ self.bfa_trace.mat @ trace
        
        self.invmass = fes.Mass(rho=1).Inverse()
        self.invMA = self.invmass @ aop
        self.gfu = GridFunction(fes)
    
    def SetInitial(self,u0):
        self.gfu.Set (u0)

    def DoTimeStep(self):
        # second order RK
        tempu = self.bfa.mat.CreateColVector()

        tempu.data = self.gfu.vec - 0.5 * self.timestep * self.invMA * self.gfu.vec
        self.gfu.vec.data -= self.timestep * self.invMA * tempu

    @property
    def concentration(self):
        return self.gfu
