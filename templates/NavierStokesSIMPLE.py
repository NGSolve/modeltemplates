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
        self.V1 = V1
        Vhat = VectorFacet(mesh, order=order-1, dirichlet=inflow+"|"+wall+"|"+outflow, hide_highest_order_dc=True)
        Sigma = HCurlDiv(mesh, order = order-1, orderinner=order, discontinuous=True)
        if mesh.dim == 2:
            S = L2(mesh, order=order-1)            
        else:
            S = VectorL2(mesh, order=order-1)

        Sigma.SetCouplingType(IntRange(0,Sigma.ndof), COUPLING_TYPE.HIDDEN_DOF)
        Sigma = Compress(Sigma)
        S.SetCouplingType(IntRange(0,S.ndof), COUPLING_TYPE.HIDDEN_DOF)
        S = Compress(S)
        
        self.X = FESpace ([V1,Vhat, Sigma, S])
        for i in range(self.X.ndof):
            if self.X.CouplingType(i) == COUPLING_TYPE.WIREBASKET_DOF:
                self.X.SetCouplingType(i, COUPLING_TYPE.INTERFACE_DOF)
        self.v1dofs = self.X.Range(0)
        
        u, uhat, sigma, W  = self.X.TrialFunction()
        v, vhat, tau, R  = self.X.TestFunction()

        if mesh.dim == 2:
            def Vec2Skew(v):
                return CoefficientFunction((0, v, -v, 0), dims = (2,2))
        else:
            def Vec2Skew(v):
                return CoefficientFunction((0, v[0], -v[1], -v[0],  0,  v[2], v[1],  -v[2], 0), dims = (3,3))

        W = Vec2Skew(W)
        R = Vec2Skew(R)

        n = specialcf.normal(mesh.dim)
        def tang(u): return u-(u*n)*n

        self.astokes = BilinearForm (self.X, eliminate_hidden = True)
        self.astokes += SymbolicBFI ( -0.5/nu * InnerProduct (sigma,tau))
        self.astokes += SymbolicBFI ( (div(sigma) * v + div(tau) * u))
        self.astokes += SymbolicBFI ( -(((sigma*n)*n ) * (v*n) + ((tau*n)*n )* (u*n)) , element_boundary = True)
        self.astokes += SymbolicBFI ( (sigma*n)*tang(vhat) + (tau*n)*tang(uhat), element_boundary = True)
        self.astokes += SymbolicBFI ( InnerProduct(W,tau) + InnerProduct(R,sigma) )
        self.astokes += SymbolicBFI ( 1e12*nu*div(u)*div(v))

            
        self.a = BilinearForm (self.X, eliminate_hidden = True)
        self.a += SymbolicBFI ( -0.5/nu * InnerProduct ( sigma,tau))
        self.a += SymbolicBFI ( (div(sigma) * v + div(tau) * u))
        self.a += SymbolicBFI ( -(((sigma*n)*n ) * (v*n) + ((tau*n)*n )* (u*n)) , element_boundary = True)
        self.a += SymbolicBFI ( (sigma*n)*tang(vhat) + (tau*n)*tang(uhat), element_boundary = True)
        self.a += SymbolicBFI ( InnerProduct(W,tau) + InnerProduct(R,sigma) )
        # self.a += SymbolicBFI ( 1*nu*div(u)*div(v))

        self.gfu = GridFunction(self.X)

        self.f = LinearForm(self.X)

        self.mstar = BilinearForm(self.X, eliminate_hidden = True, condense=True)
        self.mstar += SymbolicBFI ( -timestep*0.5/nu * InnerProduct ( sigma,tau))
        self.mstar += SymbolicBFI ( timestep*(div(sigma) * v + div(tau) * u))
        self.mstar += SymbolicBFI ( timestep*(-(((sigma*n)*n ) * (v*n) + ((tau*n)*n )* (u*n))) , element_boundary = True)
        self.mstar += SymbolicBFI ( timestep*((sigma*n)*tang(vhat) + (tau*n)*tang(uhat)), element_boundary = True)
        self.mstar += SymbolicBFI ( timestep*(InnerProduct(W,tau) + InnerProduct(R,sigma)) )
        # self.mstar += SymbolicBFI ( timestep*1*nu*div(u)*div(v))
        self.mstar += SymbolicBFI ( u*v )

        self.premstar = Preconditioner(self.mstar, "bddc")
        self.mstar.Assemble()
        # self.invmstar = self.mstar.mat.Inverse(self.X.FreeDofs(), inverse="sparsecholesky")
        
        # self.invmstar1 = self.mstar.mat.Inverse(self.X.FreeDofs(self.mstar.condense), inverse="sparsecholesky")
        self.invmstar1 = CGSolver(self.mstar.mat, pre=self.premstar, precision=1e-4)
        ext = IdentityMatrix(self.X.ndof)+self.mstar.harmonic_extension
        extT = IdentityMatrix(self.X.ndof)+self.mstar.harmonic_extension_trans
        self.invmstar = ext @ self.invmstar1 @ extT + self.mstar.inner_solve


        if False:
            u,v = V1.TnT()
            self.conv = BilinearForm(V1, nonassemble=True)
            self.conv += SymbolicBFI(InnerProduct(grad(v)*u, u).Compile(True, wait=True))
            self.conv += SymbolicBFI((-IfPos(u * n, u*n*u*v, u*n*u.Other(bnd=self.uin)*v)).Compile(True, wait=True), element_boundary = True)
            emb = Embedding(self.X.ndof, self.v1dofs)
            self.conv_operator = emb @ self.conv.mat @ emb.T
        else:
            VL2 = VectorL2(mesh, order=order, piola=True)
            ul2,vl2 = VL2.TnT()
            self.conv_l2 = BilinearForm(VL2, nonassemble=True)
            self.conv_l2 += SymbolicBFI(InnerProduct(grad(vl2)*ul2, ul2).Compile(realcompile=realcompile, wait=True))
            self.conv_l2 += SymbolicBFI((-IfPos(ul2 * n, ul2*n*ul2*vl2, ul2*n*ul2.Other(bnd=self.uin)*vl2)).Compile(realcompile=realcompile, wait=True), element_boundary = True)
        
            self.convertl2 = V1.ConvertL2Operator(VL2) @ Embedding(self.X.ndof, self.v1dofs).T
            self.conv_operator = self.convertl2.T @ self.conv_l2.mat @ self.convertl2

            
        
        self.V2 = HDiv(mesh, order=order, RT=False, discontinuous=True)
        self.Q = L2(mesh, order=order-1)
        self.Qhat = FacetFESpace(mesh, order=order, dirichlet=outflow)        
        self.Xproj = FESpace ( [self.V2, self.Q, self.Qhat] )
        (u,p,phat),(v,q,qhat) = self.Xproj.TnT()
        aproj = BilinearForm(self.Xproj, condense=True)
        aproj += SymbolicBFI(u*v+ div(u)*q + div(v) * p)
        aproj += SymbolicBFI(u*n*qhat+v*n*phat, element_boundary=True)
        aproj.Assemble()
        
        self.invproj1 = aproj.mat.Inverse(self.Xproj.FreeDofs(aproj.condense), inverse="sparsecholesky")
        ext = IdentityMatrix(self.Xproj.ndof)+aproj.harmonic_extension
        extT = IdentityMatrix(self.Xproj.ndof)+aproj.harmonic_extension_trans
        self.invproj = ext @ self.invproj1 @ extT + aproj.inner_solve
        
        self.bproj = BilinearForm(trialspace=self.V1, testspace=self.Xproj)
        self.bproj += SymbolicBFI(div(self.V1.TrialFunction())*q)
        self.bproj.Assemble()

        ind = self.V1.ndof * [0]
        for el in mesh.Elements(VOL):
            dofs1 = self.V1.GetDofNrs(el)
            dofs2 = self.V2.GetDofNrs(el)
            for d1,d2 in zip(dofs1,dofs2):
                ind[d1] = d2
        self.mapV = PermutationMatrix(self.Xproj.ndof, ind)
        
                
    @property
    def velocity(self):
        return self.gfu.components[0]
    @property
    def pressure(self):
        return 1e6/self.nu*div(self.gfu.components[0])
        
    def SolveInitial(self):
        self.a.Assemble()        
        self.astokes.Assemble()
        self.f.Assemble()
        
        temp = self.astokes.mat.CreateColVector()
        self.gfu.components[0].Set (self.uin, definedon=self.X.mesh.Boundaries(self.inflow))
        self.gfu.components[1].Set (self.uin, definedon=self.X.mesh.Boundaries(self.inflow))

        inv = self.astokes.mat.Inverse(self.X.FreeDofs(), inverse="sparsecholesky")
        temp.data = -self.astokes.mat * self.gfu.vec + self.f.vec
        self.gfu.vec.data += inv * temp

    def AddForce(self, force):
        force = CoefficientFunction(force)
        v, vhat, tau, R  = self.X.TestFunction()        
        self.f += SymbolicLFI(force * v)
        
    def DoTimeStep(self):
        
        self.temp = self.a.mat.CreateColVector()
        self.temp2 = self.a.mat.CreateColVector()        
        self.f.Assemble()
        self.temp.data = self.conv_operator * self.gfu.vec
        self.temp.data += self.f.vec
        self.temp.data += -self.a.mat * self.gfu.vec
        

        self.temp2.data = self.invmstar * self.temp
        self.Project(self.temp2[0:self.V1.ndof])
        self.gfu.vec.data += self.timestep * self.temp2.data

        
    def Project(self,vel):        
        vel.data -= (self.mapV @ self.invproj @ self.bproj.mat) * vel
