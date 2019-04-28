from ngsolve import *


class HookMaterial:
    def __init__(self,E,nu, planestrain=False, planestress=False):
        self.E = E
        self.nu = nu
        self.mu  = E / 2 / (1+nu)
        self.lam = E * nu / ((1+nu)*(1-2*nu))

    def __call__(self,strain):
        return self.mu*InnerProduct(strain,strain) + 0.5*self.lam*Trace(strain)**2

    
    
class Elasticity:
    def __init__(self, materiallaw, mesh, nonlinear=False, order=2, volumeforce=None, boundaryforce=None, dirichlet=None):
        self.fes = VectorH1(mesh, order=order, dirichlet=dirichlet)
        self.bfa = BilinearForm(self.fes)
        self.displacement = GridFunction(self.fes, name="displacement")
        
        u = self.fes.TrialFunction()
        I = Id(mesh.dim)
        if nonlinear:
            F = I + grad(u)
            strain = 0.5 * (F.trans * F - I)
        else:
            strain = 0.5 * (grad(u)+grad(u).trans)
        self.bfa += SymbolicEnergy(materiallaw(strain))
        if volumeforce:
            self.bfa += SymbolicEnergy(-volumeforce*u)
        if boundaryforce:
            self.bfa += SymbolicEnergy(-boundaryfoce*u, BND)


    def Solve(self):
        solvers.Newton(self.bfa, self.displacement)

