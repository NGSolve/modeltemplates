from ngsolve import *

__all__ = ["HookeMaterial", "NeoHookeMaterial", "Elasticity"]

class HookeMaterial:
    """Implements Hooke materiallaw.

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

    def Stress(self,strain):
        return 2*self.mu*strain + self.lam*Trace(strain)*Id(strain.dims[0])
    def Energy(self,strain):
        return self.mu*InnerProduct(strain,strain) + 0.5*self.lam*Trace(strain)**2



class NeoHookeMaterial:
    def __init__(self,E,nu, planestrain=True, planestress=True):
        self.E = E
        self.nu = nu
        self.mu  = E / 2 / (1+nu)
        self.lam = E * nu / ((1+nu)*(1-2*nu))

    def Stress(self,strain):
        # return 2*self.mu*strain + self.lam*Trace(strain)*Id(strain.dims[0])
        I = Id(strain.dims[0])        
        return self.mu * (I - Det(2*strain+I)**(-self.lam/self.mu) * Inv(2*strain+I))
    
    def Energy(self,strain):
        I = Id(strain.dims[0])
        C = 2*strain+I
        return self.mu * (Trace(strain) + self.mu/self.lam * (Det(C)**(-self.lam/2/self.mu) - 1))

    
    
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
    def __init__(self, materiallaw, mesh, \
                     nonlinear=False, order=2, \
                     volumeforce=None, boundaryforce=None, dirichlet=None,
                     boundarydisplacement=None):
                     
        # self.fes = VectorH1(mesh, order=order, dirichlet=dirichlet)
        self.fes = H1(mesh, order=order, dirichlet=dirichlet, dim=mesh.dim)
        self.bfa = BilinearForm(self.fes)
        self.displacement = GridFunction(self.fes, name="displacement")
        self.materiallaw = materiallaw
        self.nonlinear = nonlinear
        
        u = self.fes.TrialFunction()
        I = Id(mesh.dim)
        if nonlinear:
            F = I + Grad(u)
            strain = 0.5 * (F.trans * F - I)
        else:
            strain = 0.5 * (Grad(u)+Grad(u).trans)
        self.bfa += Variation( (materiallaw.Energy(strain)*dx).Compile())
        
        if volumeforce:
            self.bfa += Variation (-volumeforce*u*dx)
            
        if boundaryforce:
            if type(boundaryforce) is dict:
                for key,val in boundaryforce.items():
                    self.bfa += Variation (-val*u*ds(key))
            else:
                self.bfa += Variation (-boundaryforce*u*ds)


        if boundarydisplacement:
            if type(boundarydisplacement) is dict:
                print ("dim u = ", u.dims)
                for key,val in boundarydisplacement.items():
                    val = CoefficientFunction(val)
                    print ("dim val = ", val.dims)
                    self.bfa += Variation (10**8* InnerProduct(u-val,u-val)*ds(key))
            
                

                
    def Solve(self):
        solvers.Newton(self.bfa, self.displacement)


        
    @property
    def F(self):
        I = Id(self.displacement.dim)        
        return I + Grad(self.displacement)
        
    @property
    def strain(self):
        if self.nonlinear:
            I = Id(self.displacement.dim)
            F = I + Grad(self.displacement)
            return 0.5 * (F.trans*F-I)
        else:
            return 0.5 * (Grad(self.displacement)+Grad(self.displacement).trans)
        
    @property
    def stress(self):
        return self.materiallaw.Stress(self.strain).Compile()

