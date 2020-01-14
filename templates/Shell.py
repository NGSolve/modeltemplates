from ngsolve import *
import ngsolve.comp
import netgen.csg as csg
import netgen.meshing as meshing
import re as re

__all__ = ["HookeShellMaterial", "Shell"]

def Normalize (v):
    return 1/Norm(v) * v

class HookeShellMaterial:
    """Hookean shell materiallaw.

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

    def MaterialNorm(self, mat):
        return self.E/(1-self.nu**2)*((1-self.nu)*InnerProduct(mat,mat)+self.nu*Trace(mat)**2)

    def MaterialNormInv(self, mat):
        return (1+self.nu)/self.E*(InnerProduct(mat,mat)-self.nu/(2*self.nu+1)*Trace(mat)**2)


CF = CoefficientFunction
    
class Shell:
    """Shell problem template.

"""
    def __init__(self, materiallaw, mesh, thickness=0.1, order=2, \
                 moments=None, volumeforce=None, boundaryforce=None, bboundaryforce=None, \
                 dirichlet=None, optimize=True):

        if not dirichlet:
            dirichlet = ["","","",""]

        self.thickness = thickness
        

        self.fesU = VectorH1(mesh, order=order, dirichletx_bbnd=dirichlet[0], dirichlety_bbnd=dirichlet[1], dirichletz_bbnd=dirichlet[2])
        self.fesQ = HDivDivSurface(mesh, order=order-1, discontinuous=True)
        self.fesH = HDivSurface(mesh, order=order-1, orderinner=0, dirichlet_bbnd=dirichlet[3])
        self.fesR = HCurlCurl(mesh, order=order-1, discontinuous=True)
        self.fes  = FESpace( [self.fesU, self.fesQ, self.fesH, self.fesR, self.fesR] )

        self.fesVF = VectorFacetSurface(mesh, order=order+1)

        
        fesclamped = FacetSurface(mesh,order=0)
        gfclamped = GridFunction(fesclamped)
        gfclamped.Set(1,definedon=mesh.BBoundaries(dirichlet[3]))

        self.bfa = BilinearForm(self.fes, symmetric=True, condense=True)
        self.bfF = BilinearForm(self.fesVF, symmetric=True)
        self.solution = GridFunction(self.fes, name="solution")
        self.averednv = GridFunction(self.fesVF, name="averednv")
        self.averednv_start = GridFunction(self.fesVF, name="averednv_start")
        
        u,sigma,hyb,C,R = self.fes.TrialFunction()
        sigma,hyb,C,R   = sigma.Trace(), hyb.Trace(), C.Trace(), R.Operator("dualbnd")

        b = self.fesVF.TrialFunction().Trace()
        b.dims=(3,)

        nsurf = specialcf.normal(mesh.dim)
        t     = specialcf.tangential(mesh.dim)
        nel   = Cross(nsurf, t)
    
        Ptau    = Id(mesh.dim) - OuterProduct(nsurf,nsurf)
        Ftau    = grad(u).Trace() + Ptau
        Etautau = 0.5*(grad(u).Trace().trans*grad(u).Trace() + Ptau*grad(u).Trace() + grad(u).Trace().trans*Ptau)
    
        J, Jbnd = Norm(Cof(Ftau)), Norm(Ftau*t)
        nphys   = Normalize(Cof(Ftau)*nsurf)
        tphys   = Normalize(Ftau*t)
        nelphys = Cross(nphys,tphys)

        Hn = CF( (u.Operator("hesseboundary").trans*nphys), dims=(3,3) )

        cfnphys = Normalize(Cof(Ptau+grad(self.solution.components[0]))*nsurf)

        cfn  = Normalize(CF( self.averednv.components ))
        cfnR = Normalize(CF( self.averednv_start.components ))
        pnaverage = Normalize( cfn - (tphys*cfn)*tphys )

        self.bfF += Variation( (0.5*b*b - ((1-gfclamped)*cfnphys+gfclamped*nsurf)*b).Compile(optimize,wait=True)*ds(element_boundary=True))
    
        self.rf = self.averednv.vec.CreateVector()
        self.bfF.Apply(self.averednv.vec, self.rf)
        self.bfF.AssembleLinearization(self.averednv.vec)
        self.invF = self.bfF.mat.Inverse(self.fesVF.FreeDofs(), inverse="sparsecholesky")
        self.averednv.vec.data -= self.invF*self.rf
        self.averednv_start.vec.data = self.averednv.vec

        gradn = specialcf.Weingarten(mesh.dim)

        self.bfa = BilinearForm(self.fes, symmetric=True, condense=True)
        self.bfa += Variation( (-6/self.thickness**3*materiallaw.MaterialNormInv(sigma)
                                + InnerProduct(1/J*sigma, Hn + (J - nphys*nsurf)*gradn)).Compile(optimize,wait=True)*ds )
        self.bfa += Variation( (0.5*self.thickness*materiallaw.MaterialNorm(C)).Compile(optimize,wait=True)*ds )
        self.bfa += Variation( InnerProduct(C-Etautau, R).Compile(optimize,wait=True)*ds(element_vb=BND) )
        self.bfa += Variation( InnerProduct(C-Etautau, R).Compile(optimize,wait=True)*ds(element_vb=VOL) )
        self.bfa += Variation( (-(acos(nel*cfnR)-acos(nelphys*pnaverage)-hyb*nel)*(sigma*nel)*nel).Compile(optimize,wait=True)*ds(element_boundary=True) )
        
        if moments:
            if type(moments) is dict:
                for key,val in moments.items():
                    raise Exception("not implemented!")
                    #self.bfa += Variation( (-CF(moments)*(hyb*nel)).Compile(optimize,wait=True)*ds(key) )
            else:
                self.bfa += Variation( (-moments*(hyb*nel)).Compile(optimize,wait=True)*ds(element_boundary=True) )
            
        if volumeforce:
            self.bfa += Variation( -volumeforce*u*ds )
                
        if boundaryforce:
            if type(boundaryforce) is dict:
                for key,val in boundaryforce.items():
                    self.bfa += Variation( (-CF(val)*u).Compile(optimize,wait=True)*ds(definedon=mesh.BBoundaries(key)) )
            else:
                self.bfa += Variation( (-boundaryforce*u).Compile(optimize,wait=True)*ds(definedon=mesh.BBoundaries(".*")) )
                                       
        if bboundaryforce:
            if type(bboundaryforce) is dict:
                for key,val in bboundaryforce.items():
                    self.bfa += Variation( (-CF(val)*u).Compile(optimize,wait=True)*ds(definedon=mesh.BBoundaries(key)) )
            else:
                self.bfa += Variation( (-bboundaryforce*u).Compile(optimize,wait=True)*ds(definedon=mesh.BBBoundaries(".*")) )
            

                
    def Solve(self, **kwargs):
            self.bfF.Apply(self.averednv.vec, self.rf)
            self.bfF.AssembleLinearization(self.averednv.vec)
            self.invF.Update()
            self.averednv.vec.data -= self.invF*self.rf

            solvers.Newton(self.bfa, self.solution, inverse="sparsecholesky", printing = ngsglobals.msg_level >= 1, maxerr=1e-10, maxit=20, **kwargs)


    @property
    def displacement(self):
        return self.solution.components[0]

    @property
    def moments(self):
        return self.solution.components[1]

