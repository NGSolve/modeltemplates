from ngsolve import *
import ngsolve.comp
import netgen.csg as csg
import netgen.meshing as meshing

__all__ = ["HookeShellMaterial", "Shell", "MakeStructuredSurfaceMesh"]

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
                 dirichlet=None):
                     
        bndnames = []
       
        if not dirichlet == None:
            bndnames.append(dirichlet)
        dirbnds = "|".join(bndnames)

        self.thickness = thickness
        

        self.fesU = VectorH1(mesh, order=order)
        self.fesQ = HDivDivSurface(mesh, order=order-1, discontinuous=True)
        self.fesH = HDivSurface(mesh, order=order-1, orderinner=0)
        self.fesR = HCurlCurl(mesh, order=order-1, discontinuous=True)
        self.fes  = FESpace( [self.fesU, self.fesQ, self.fesH, self.fesR, self.fesR] )

        self.fesF = FacetSurface(mesh, order=order)
        self.fesVF = FESpace( [self.fesF, self.fesF, self.fesF] )
        
        self.fesSL2  = SurfaceL2(mesh, order=order)
        self.fesVSL2 = FESpace( [self.fesSL2,self.fesSL2,self.fesSL2] )

        """
        fesclamped = FacetSurface(mesh,order=0)
        gfclamped = GridFunction(fesclamped)
        for el in fesclamped.Elements(BBND):
            if re.match(clBM.GetClampedBnd()[3], el.mat, 0):
                for dof in el.dofs:
                    gfclamped.vec[dof] = 1

        ###### Dirichlet Data ######
        freedofs      = BitArray(fes.FreeDofs(el_int))
        if vh1:
            SetDirichletBBND(freedofs, [0], fes, clBM.GetClampedBnd()[0])
            if clBM.GetClampedBBnd():
                SetDirichletBBBND(freedofs, [0], fes, clBM.GetClampedBBnd()[0])
            SetDirichletBBND(freedofs, [2], fes, clBM.GetClampedBnd()[3])
        else:
            for i in range(3):
                SetDirichletBBND(freedofs, [i], fes, clBM.GetClampedBnd()[i])
                if clBM.GetClampedBBnd():
                    SetDirichletBBBND(freedofs, [i], fes, clBM.GetClampedBBnd()[i])
            SetDirichletBBND(freedofs, [4], fes, clBM.GetClampedBnd()[3])
        """
        
        self.bfa = BilinearForm(self.fes, symmetric=True, condense=True)
        self.bfF = BilinearForm(self.fes, symmetric=True)
        self.solution = GridFunction(self.fes, name="solution")
        self.averednv = GridFunction(self.fesVF, name="averednv")
        self.averednv_start = GridFunction(self.fesVF, name="averednv_start")
        self.normalvec = GridFunction(self.fesVSL2)
        
        u,sigma,hyb,C,R = self.fes.TrialFunction()
        sigma,hyb,C,R   = sigma.Trace(), hyb.Trace(), C.Trace(), R.Operator("dualbnd")

        bx,by,bz = self.fesVF.TrialFunction()
        b        = CF( (bx.Trace(),by.Trace(),bz.Trace()) )


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

        self.bfF += Variation((0.5*b*b - ((1-gfclamped)*gfnphys+gfclamped*nsurf)*b)*ds(element_boundary=True))
    
    
        self.rf = self.averednv.vec.CreateVector()
        self.bfF.Apply(self.averednv.vec, self.rf)
        self.bfF.AssembleLinearization(self.averednv.vec)
        self.invF = self.bfF.mat.Inverse(fesFV.FreeDofs(), inverse="sparsecholesky")
        self.averednv.vec.data -= self.invF*self.rf
        averednv_old.vec.data = self.averednv.vec
        

        self.normalvec = GridFunction(self.fesVSL2)
        self.normalvec.components[0].Set( nsurf[0], definedon=mesh.Boundaries(".*") )
        self.normalvec.components[1].Set( nsurf[1], definedon=mesh.Boundaries(".*") )
        self.normalvec.components[2].Set( nsurf[2], definedon=mesh.Boundaries(".*") )
        gradn = CF( (grad(self.normalvec.components[0]),grad(self.normalvec.components[1]),grad(self.normalvec.components[2])), dims=(3,3) )

        self.bfa = BilinearForm(self.fes, symmetric=True, condense=True)
        self.bfa += Variation( (-6/self.thickness**3*materiallaw.MaterialNormInv(sigma)
                                + InnerProduct(1/J*sigma, Hn + (J - nphys*nsurf)*gradn))*ds() )
        self.bfa += Variation( 0.5*self.thickness*materiallaw.MaterialNorm(C)*ds )
        self.bfa += SymbolicEnergy( InnerProduct(C-Etautau, R), BND, element_vb=BND )
        self.bfa += SymbolicEnergy( InnerProduct(C-Etautau, R), BND, element_vb=VOL )
        self.bfa += SymbolicEnergy( -(acos(nel*cfnR)-acos(nelphys*pnaverage)-hyb*nel)*(sigma*nel)*nel, BND, element_boundary=True)
        
        if moments:
            if type(moments) is dict:
                for key,val in moments.items():
                    raise Exception("not implemented!")
                    #self.bfa += SymbolicEnergy( -CF(moments)*(hyb*nel)*ds(key) )
                else:
                    self.bfa += SymbolicEnergy( -moments*(hyb*nel), BND, element_boundary=True)
            
        if volumeforce:
            self.bfa += Variation( -volumeforce*u*ds )
                
        if boundaryforce:
            if type(boundaryforce) is dict:
                for key,val in boundaryforce.items():
                    self.bfa += Variation( -CF(val)*u*dx(definedon=mesh.BBoundaries(key)) )
            else:
                self.bfa += Variation( -boundaryforce*u*dx(definedon=mesh.BBoundaries(".*")) )
                                       
        if bboundaryforce:
            if type(bboundaryforce) is dict:
                for key,val in bboundaryforce.items():
                    self.bfa += Variation( -CF(val)*u*dx(definedon=mesh.BBoundaries(key)) )
            else:
                self.bfa += Variation( -bboundaryforce*u*ds(definedon=mesh.BBBoundaries(".*")) )
            

                
    def Solve(self, **kwargs):
            self.bfF.Apply(self.averednv.vec, self.rf)
            self.bfF.AssembleLinearization(self.averednv.vec)
            self.invF.Update()
            self.averednv.vec.data -= self.invF*self.rf

            solvers.Newton(self.bfa, self.solution, inverse="sparsecholesky", printing = ngsglobals.msg_level >= 1, **kwargs)


    @property
    def displacement(self):
        return self.solution.components[0]

    @property
    def moments(self):
        return self.solution.components[1]








def MakeStructuredSurfaceMesh(quads=True, nx=10, ny=10, mapping = None, secondorder=False, bbbpts=None, bbbnames=None):
    """
    Generate a structured 2D mesh

    Parameters
    ----------
    quads : bool
      If True, a quadrilateral mesh is generated. If False, the quads are split to triangles.

    nx : int
      Number of cells in x-direction.

    ny : int
      Number of cells in y-direction.

    mapping: lamda
      Mapping to transform the generated points. If None, the identity mapping is used.
    

    Returns
    -------
    (ngsolve.mesh)
      Returns generated NGSolve mesh

    """
    mesh = meshing.Mesh()
    mesh.dim=3

    found = []
    indbbbpts = []
    if bbbpts:
        for i in range(len(bbbpts)):
            found.append(False)
            indbbbpts.append(None)

    pids = []
    for i in range(ny+1):
        for j in range(nx+1):
            x,y,z = j/nx, i/ny, 0
            pids.append(mesh.Add (meshing.MeshPoint(csg.Pnt(x,y,z))))
            

    mesh.Add(meshing.FaceDescriptor(surfnr=1,domin=1,bc=1))
    
    for i in range(ny):
        for j in range(nx):
            base = i * (nx+1) + j
            if quads:
                pnum = [base,base+1,base+nx+2,base+nx+1]
                elpids = [pids[p] for p in pnum]
                el = meshing.Element2D(1,elpids)
                if not mapping:
                    el.curved=False
                mesh.Add(el)
            else:
                pnum1 = [base,base+1,base+nx+1]
                pnum2 = [base+1,base+nx+2,base+nx+1]
                elpids1 = [pids[p] for p in pnum1]
                elpids2 = [pids[p] for p in pnum2]
                mesh.Add(meshing.Element2D(1,elpids1)) 
                mesh.Add(meshing.Element2D(1,elpids2))                          

    for i in range(nx):
        mesh.Add(meshing.Element1D([pids[i], pids[i+1]], index=1))
    for i in range(ny):
        mesh.Add(meshing.Element1D([pids[i*(nx+1)+nx], pids[(i+1)*(nx+1)+nx]], index=2))
    for i in range(nx):
        mesh.Add(meshing.Element1D([pids[ny*(nx+1)+i+1], pids[ny*(nx+1)+i]], index=3))
    for i in range(ny):
        mesh.Add(meshing.Element1D([pids[(i+1)*(nx+1)], pids[i*(nx+1)]], index=4))

    mesh.SetCD2Name(1, "bottom")        
    mesh.SetCD2Name(2, "right")        
    mesh.SetCD2Name(3, "top")        
    mesh.SetCD2Name(4, "left")

    if secondorder:
        mesh.SecondOrder()
    
    if mapping:
        i = 0
        for p in mesh.Points():
            x,y,z = p.p
            x,y,z = mapping(x,y,z)
            p[0] = x
            p[1] = y
            p[2] = z
            for k in range(len(found)):
                if abs(x-bbbpts[k][0])+abs(y-bbbpts[k][1])+abs(z-bbbpts[k][2]) < 1e-6:
                    indbbbpts[k] = pids[i]
                    found[k] = True
            i += 1

                    
    for k in range(len(found)):
        if found[k] == False:
            raise Exception("bbbpnt[",k,"] not in structured mesh!")

    for i in range(len(indbbbpts)):
        mesh.Add(meshing.Element0D(indbbbpts[i], index=i+1))
        mesh.SetCD3Name(i+1, bbbnames[i])

    mesh.Compress()       
    ngsmesh = Mesh(mesh)
    return ngsmesh
