from ngsolve import *
import ngsolve.comp

__all__ = ["HookeMaterial", "NeoHookeMaterial", "Elasticity"]

class HookeMaterial:
    """Hookean materiallaw.

The energy density is given by

.. math::
    \\mu \\langle \\sigma, \\sigma \\rangle + \\frac{1}{2} \\lambda \\, \\text{trace}(\\sigma)^2

with

.. math::
    \\mu = \\frac{E}{2(1+\\mathrm{nu})} \\\\
    \\lambda = \\frac{E \\, \\mathrm{nu}}{(1+\\mathrm{nu})(1-2 \\, \\mathrm{nu})}

E is the Young modulus, and nu is the Poisson ratio.
"""
    def __init__(self,E,nu, planestrain=False, planestress=False):
        self.E = E
        self.nu = nu
        self.mu  = E / 2 / (1+nu)
        self.lam = E * nu / ((1+nu)*(1-2*nu))
        self.linear = True

    def Stress(self,strain):
        return 2*self.mu*strain + self.lam*Trace(strain)*Id(strain.dims[0])
    def Energy(self,strain):
        return self.mu*InnerProduct(strain,strain) + 0.5*self.lam*Trace(strain)**2



class NeoHookeMaterial:
    """Hookean materiallaw.

The energy density is given by

.. math::
    \\mu \\langle \\sigma, \\sigma \\rangle + \\frac{1}{2} \\lambda \\, \\text{trace}(\\sigma)^2
"""
    
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


CF = CoefficientFunction
    
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
                     boundarydisplacement=None, meandisplacement=None):
                     
        # self.fes = VectorH1(mesh, order=order, dirichlet=dirichlet)
        bndnames = []
        if boundarydisplacement is not None:
            bndnames = list(boundarydisplacement.keys())
        if not dirichlet == None:
            bndnames.append(dirichlet)
        dirbnds = "|".join(bndnames)
        # print ("***********************  dirbnds =", dirbnds)
        self.fes = H1(mesh, order=order, dirichlet=dirbnds, dim=mesh.dim)
        
        if type(meandisplacement) is dict:
            l = [self.fes]
            for key,val in meandisplacement.items():                
                l.append(NumberSpace(mesh, dim=mesh.dim, definedon="", definedonbound=key))
            self.fes = FESpace(l)

        self.bfa = BilinearForm(self.fes)
        self.solution = GridFunction(self.fes, name="solution")
        self.materiallaw = materiallaw
        self.nonlinear = nonlinear
        self.dirichletvalues = GridFunction(self.fes, name="dir")
        self.boundarydisplacement = boundarydisplacement
        
        if not self.nonlinear and self.materiallaw.linear:
            self.lff = LinearForm(self.fes)
            u,v = self.fes.TnT()
            self.bfa += InnerProduct(materiallaw.Stress(PySym(Grad(u))), PySym(Grad(v))) * dx

            if volumeforce:            
                self.lff += -volumeforce*v*dx

            if boundaryforce:
                if type(boundaryforce) is dict:
                    for key,val in boundaryforce.items():
                        self.lff += -CF(val)*v*ds(key)
                else:
                    self.lff += -boundaryforce*v*ds

            if not self.nonlinear and self.materiallaw.linear:
                self.pre = MultiGridPreconditioner(self.bfa, inverse="sparsecholesky")
        else:
            if type(self.solution.space) is ngsolve.comp.CompoundFESpace:            
                u,*lam = self.fes.TrialFunction()
            else:
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
                        self.bfa += Variation (-CF(val)*u*ds(key))
                else:
                    self.bfa += Variation (-boundaryforce*u*ds)

                # if type(boundarydisplacement) is dict:
                # for key,val in boundarydisplacement.items():
                # val = CoefficientFunction(val)
                # self.bfa += Variation (10**10*materiallaw.E*InnerProduct(u-val,u-val)*ds(key))
                # self.bfa += Variation (10**8*materiallaw.E* ( (u-val)*n )**2 *ds(key))

            if type(meandisplacement) is dict:
                nr = 0
                for key,val in meandisplacement.items():
                    val = CoefficientFunction(val)
                    self.bfa += Variation (InnerProduct (u-val,lam[nr])*ds(key))
                    self.bfa += Variation (-1e-10*InnerProduct (lam[nr],lam[nr])*ds(key))
                    nr = nr+1

                
    def Solve(self, **kwargs):
        if not self.nonlinear and self.materiallaw.linear:
            self.bfa.Assemble()
            self.lff.Assemble()
            solvers.BVP(bf=self.bfa, lf=self.lff, gf=self.solution, pre=self.pre)
        else:
            
            if self.boundarydisplacement:
                cf = CoefficientFunction( [self.boundarydisplacement[bc] if bc in self.boundarydisplacement.keys() else None for bc in self.fes.mesh.GetBoundaries()] )
                reg = self.fes.mesh.Boundaries( "|".join (self.boundarydisplacement.keys()))
                self.dirichletvalues.Set(cf, definedon=reg)
                # Draw (self.dirichletvalues, self.fes.mesh, "dirichlet")

            self.bfa.AssembleLinearization(self.solution.vec)

            solvers.Newton(self.bfa, self.solution, inverse="sparsecholesky", dirichletvalues=self.dirichletvalues.vec, printing = ngsglobals.msg_level >= 1, **kwargs)


    @property
    def displacement(self):
        print (type(self.solution.space))
        if type(self.solution.space) is ngsolve.comp.CompoundFESpace:
            print ("is compound")
            return self.solution.components[0]
        else:
            return self.solution
        
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

    
    def ReactionForces(self, bnd=None):
        w = GridFunction(self.fes)
        res = self.displacement.vec.CreateVector()
        self.bfa.Apply (self.displacement.vec, res)
        force = []
        for dir in [(1,0,0), (0,1,0), (0,0,1)]:
            if bnd is None:
                w.Set(dir)
            else:
                w.Set(dir, definedon=self.bfa.space.mesh.Boundaries(bnd))
            force.append(InnerProduct(w.vec, res))
        return force
    
