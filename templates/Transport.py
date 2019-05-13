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

        self.bfa = BilinearForm(fes, nonassemble=True)
        self.bfa += -u * wind * grad(v) * dx
        wn = wind*specialcf.normal(mesh.dim)
        self.bfa += wn * IfPos(wn, u, u.Other(bnd=uin)) * v * dx(element_boundary=True)

        self.invmass = fes.Mass(rho=1).Inverse()
        self.invMA = self.invmass @ self.bfa.mat
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
