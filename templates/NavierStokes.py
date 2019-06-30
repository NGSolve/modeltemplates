from ngsolve import *
from .mt_global import *

__all__ = ["NavierStokes"]

class NavierStokes:
    
    def __init__(self, mesh, nu, inflow, outflow, wall, uin, timestep, order=2, volumeforce=None):

        self.nu = nu
        self.timestep = timestep
        self.uin = uin
        self.inflow = inflow
        self.outflow = outflow
        self.wall = wall
        
        V1 = HDiv(mesh, order=order, dirichlet=inflow+"|"+wall, RT=False)
        Vhat = TangentialFacetFESpace(mesh, order=order-1, dirichlet=inflow+"|"+wall+"|"+outflow) # , hide_highest_order_dc=True)
        Sigma = HCurlDiv(mesh, order = order-1, orderinner=order, discontinuous=True)

        if mesh.dim == 2:
            S = L2(mesh, order=order-1)            
        else:
            S = VectorL2(mesh, order=order-1)

        Sigma.SetCouplingType(IntRange(0,Sigma.ndof), COUPLING_TYPE.HIDDEN_DOF)
        Sigma = Compress(Sigma)
        S.SetCouplingType(IntRange(0,S.ndof), COUPLING_TYPE.HIDDEN_DOF)
        S = Compress(S)
            
        self.V = FESpace ([V1,Vhat, Sigma, S])

        # self.V.SetCouplingType(self.V.Range(2), COUPLING_TYPE.HIDDEN_DOF)
        # self.V.SetCouplingType(self.V.Range(3), COUPLING_TYPE.HIDDEN_DOF)
        self.v1dofs = self.V.Range(0)
        
        u, uhat, sigma, W  = self.V.TrialFunction()
        v, vhat, tau, R  = self.V.TestFunction()

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
        (-(sigma*n)*tang(vhat) - (tau*n)*tang(uhat)) * dS

            
        self.a = BilinearForm (self.V, eliminate_hidden = True)
        self.a += stokesA + 1e12*nu*div(u)*div(v) * dx

        self.gfu = GridFunction(self.V)

        self.f = LinearForm(self.V)

        self.mstar = BilinearForm(self.V, eliminate_hidden = True)
        self.mstar += u*v * dx + timestep * stokesA + timestep * 1e12*nu*div(u)*div(v)*dx
        
        if False:
            u,v = V1.TnT()
            self.conv = BilinearForm(V1, nonassemble=True)
            self.conv += SymbolicBFI(InnerProduct(grad(v)*u, u).Compile(True, wait=True), bonus_intorder=order)
            self.conv += SymbolicBFI((-IfPos(u * n, u*n*u*v, u*n*u.Other(bnd=self.uin)*v)).Compile(True, wait=True), element_boundary = True)
            emb = Embedding(self.V.ndof, self.v1dofs)
            self.conv_operator = emb @ self.conv.mat @ emb.T
        else:
            VL2 = VectorL2(mesh, order=order, piola=True)
            ul2,vl2 = VL2.TnT()
            self.conv_l2 = BilinearForm(VL2, nonassemble=True)
            self.conv_l2 += SymbolicBFI(InnerProduct(grad(vl2)*ul2, ul2).Compile(realcompile=realcompile, wait=True), bonus_intorder=order)
            self.conv_l2 += SymbolicBFI((-IfPos(ul2 * n, ul2*n*ul2*vl2, ul2*n*ul2.Other(bnd=self.uin)*vl2)).Compile(realcompile=realcompile, wait=True), element_boundary = True, bonus_intorder=order)
        
            self.convertl2 = V1.ConvertL2Operator(VL2) @ Embedding(self.V.ndof, self.v1dofs).T
            self.conv_operator = self.convertl2.T @ self.conv_l2.mat @ self.convertl2

            
        self.invmstar = None

        
    @property
    def velocity(self):
        return self.gfu.components[0]
    @property
    def pressure(self):
        return 1e6/self.nu*div(self.gfu.components[0])
        
    def SolveInitial(self):
        self.a.Assemble()
        self.f.Assemble()
        
        temp = self.a.mat.CreateColVector()
        self.gfu.components[0].Set (self.uin, definedon=self.V.mesh.Boundaries(self.inflow))
        self.gfu.components[1].Set (self.uin, definedon=self.V.mesh.Boundaries(self.inflow))
        inv = self.a.mat.Inverse(self.V.FreeDofs(), inverse="sparsecholesky")
        temp.data = -self.a.mat * self.gfu.vec + self.f.vec
        self.gfu.vec.data += inv * temp

    def AddForce(self, force):
        force = CoefficientFunction(force)
        v, vhat, tau, R  = self.V.TestFunction()        
        self.f += force*v*dx
        
    def DoTimeStep(self):
        
        if not self.invmstar:
            self.mstar.Assemble()
            self.invmstar = self.mstar.mat.Inverse(self.V.FreeDofs(), inverse="sparsecholesky")
        
        self.temp = self.a.mat.CreateColVector()        
        self.f.Assemble()
        self.temp.data = self.conv_operator * self.gfu.vec
        self.temp.data += self.f.vec
        self.temp.data += -self.a.mat * self.gfu.vec
        
        self.gfu.vec.data += self.timestep * self.invmstar * self.temp


        
