from ngsolve import *
from mt_global import *


class TransportEquation:
    
    def __init__(self, mesh, wind, timestep, order=2, uin=0, source=None):

        wind = CoefficientFunction(wind)
        self.timestep = timestep
        self.uin = uin
        self.source = source
        
        fes = L2(mesh, order=order, all_dofs_together=True)
        u,v = fes.TnT()

        self.bfa = BilinearForm(fes, nonassemble=True)
        self.bfa += SymbolicBFI (-u * wind * grad(v))
        wn = wind*specialcf.normal(mesh.dim)
        self.bfa += SymbolicBFI ( IfPos(wn, wn*u, wn*u.Other(bnd=uin)) * v, element_boundary=True)

        self.invmass = fes.Mass(rho=1).Inverse()
        self.gfu = GridFunction(fes)
    
    def SetInitial(self,u0):
        self.gfu.Set (u0)

    def DoTimeStep(self):
        # second order RK
        temp  = self.bfa.mat.CreateColVector()
        tempu = self.bfa.mat.CreateColVector()
        
        temp.data = self.bfa.mat * self.gfu.vec
        tempu.data = self.gfu.vec - 0.5 * self.timestep * self.invmass *temp
        temp.data = self.bfa.mat * tempu
        self.gfu.vec.data -= self.timestep * self.invmass *temp

    def Concentration(self):
        return self.gfu
