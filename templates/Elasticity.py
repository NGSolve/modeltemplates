from ngsolve import *

__all__ = ["HookMaterial", "Elasticity"]

class HookMaterial:
    """Implements Hook materiallaw.

The law is given by

.. math::
    \\mu \\langle \\sigma, \\sigma \\rangle + \\frac{1}{2} \\lambda \\, \\text{trace}(\\sigma)^2

with

.. math::
    \\mu = \\frac{E}{2(1+\\mathrm{nu})} \\\\
    \\lambda = \\frac{E \\, \\mathrm{nu}}{(1+\\mathrm{nu})(1-2 \\, \\mathrm{nu})}

Can be called with some strain :math:`\\sigma` and returns materiallaw(:math:`\\sigma`).
"""
    def __init__(self,E,nu, planestrain=False, planestress=False):
        self.E = E
        self.nu = nu
        self.mu  = E / 2 / (1+nu)
        self.lam = E * nu / ((1+nu)*(1-2*nu))

    def __call__(self,strain):
        return self.mu*InnerProduct(strain,strain) + 0.5*self.lam*Trace(strain)**2

    
    
class Elasticity:
    """Elasticity problem template.

Implements minimization of

.. math::
    \\int_{\\mathrm{VOL}(\\mathrm{mesh})} \\mathrm{materiallaw}(\\sigma (u)) \\, dx

with

.. math:: 
    \\sigma (u) := \\left\\{ \\begin{array} \\math{F} = \\mathrm{Id} + \\nabla u, & \\sigma = \\frac{1}{2} (F^T F - \\mathrm{Id}), & \\text{if nonlinear} \\\\\\ & \\frac{1}{2} (\\nabla u + (\\nabla u)^T), & \\text{else}\\
 \\end{array} \\right.

"""
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

