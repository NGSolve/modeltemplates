from ngsolve import *
from .mt_global import *

__all__ = ["Stokes"]


class Stokes:
    
    def __init__(self, mesh, nu, inflow, outflow, wall, uin, order=2, volumeforce=None):

        self.nu = nu
        self.uin = uin
        self.inflow = inflow
        self.outflow = outflow
        self.wall = wall
        
        V = HDiv(mesh, order=order, dirichlet=inflow+"|"+wall, RT=False)
        self.V = V
        Vhat = VectorFacet(mesh, order=order-1, dirichlet=inflow+"|"+wall+"|"+outflow) # , hide_highest_order_dc=True)
        Q = L2(mesh, order=order-1, lowest_order_wb=True)
        Sigma = HCurlDiv(mesh, order = order-1, orderinner=order, discontinuous=True)
        if mesh.dim == 2:
            S = L2(mesh, order=order-1)            
        else:
            S = VectorL2(mesh, order=order-1)
        
        Sigma.SetCouplingType(IntRange(0,Sigma.ndof), COUPLING_TYPE.HIDDEN_DOF)
        Sigma = Compress(Sigma)
        S.SetCouplingType(IntRange(0,S.ndof), COUPLING_TYPE.HIDDEN_DOF)
        S = Compress(S)
        
        self.X = FESpace ([V,Vhat,Q,Sigma,S])
        for i in range(self.X.ndof):
            if self.X.CouplingType(i) == COUPLING_TYPE.WIREBASKET_DOF:
                self.X.SetCouplingType(i, COUPLING_TYPE.INTERFACE_DOF)
        self.v1dofs = self.X.Range(0)
        
        u, uhat, p, sigma, W  = self.X.TrialFunction()
        v, vhat, q, tau, R  = self.X.TestFunction()

        if mesh.dim == 2:
            def Skew2Vec(m):
                return m[1,0]-m[0,1]
        else:
            def Skew2Vec(m):   
                return CoefficientFunction( (m[0,1]-m[1,0], m[2,0]-m[0,2], m[1,2]-m[2,1]) )

        dS = dx(element_boundary=True)
        n = specialcf.normal(mesh.dim)
        def tang(u): return u-(u*n)*n
        
        stokesA = -0.5/nu * InnerProduct(sigma,tau) * dx + \
          (div(sigma)*v+div(tau)*u) * dx + \
          (InnerProduct(W,Skew2Vec(tau)) + InnerProduct(R,Skew2Vec(sigma))) * dx + \
          -(((sigma*n)*n) * (v*n) + ((tau*n)*n )* (u*n)) * dS + \
          ( (sigma*n)*tang(vhat) + (tau*n)*tang(uhat)) * dS 
        stokesB = (div(u)*q + div(v)*p)*dx
        stokesA += 10*nu*div(u)*div(v)*dx
        
        self.a = BilinearForm (self.X, eliminate_hidden = True, condense=False)
        self.a += stokesA + stokesB

        self.norm = BilinearForm (self.X, eliminate_hidden = True, condense=False)
        h = specialcf.mesh_size
        self.norm += (1/(h*h)*u*v+p*q+InnerProduct(sigma,tau)+W*R)*dx + 1/h*tang(uhat)*tang(vhat)*dS
        # self.pre = Preconditioner(self.norm, "local")
        self.pre = Preconditioner(self.norm, "bddc")
        
        self.gfu = GridFunction(self.X)
        self.f = LinearForm(self.X)
                
    @property
    def velocity(self):
        return self.gfu.components[0]
    @property
    def pressure(self):
        return self.gfu.components[2]
        
    def Solve(self):
        self.a.Assemble()        
        self.norm.Assemble()
        self.f.Assemble()
        
        # temp = self.a.mat.CreateColVector()
        self.gfu.components[0].Set (self.uin, definedon=self.X.mesh.Boundaries(self.inflow))
        self.gfu.components[1].Set (self.uin, definedon=self.X.mesh.Boundaries(self.inflow))

        # temp.data = -self.a.mat * self.gfu.vec + self.f.vec
        solvers.MinRes(mat=self.a.mat, rhs=self.f.vec, sol=self.gfu.vec, pre=self.pre, initialize=False, maxsteps=10000)
        # self.gfu.vec.data += self.a.harmonic_extension * self.gfu.vec.data

    def AddForce(self, force):
        force = CoefficientFunction(force)
        v, vhat, q, tau, R  = self.X.TestFunction()        
        self.f += SymbolicLFI(force * v)
        
