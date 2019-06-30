from netgen.geom2d import *
from ngsolve import *
import math

geo = SplineGeometry()

# naca0015
t = 0.15
def profile(x):
    return 5*t*(0.2969*math.sqrt(x)-0.1260*x-0.3516*x*x+0.2843*x*x*x-0.1015*x*x*x*x)

geo.AddCurve( lambda t : (t,profile(t)), leftdomain=1, rightdomain=0, bc="wall", maxh=0.01)
geo.AddCurve( lambda t : (t,-profile(t)), leftdomain=0, rightdomain=1, bc="wall",maxh=0.01)
geo.AddCurve( lambda t : (1, profile(1)*t-profile(1)*(1-t)), leftdomain=0, rightdomain=1, bc="wall", maxh=0.01)
geo.AddRectangle((-1,-2),(3,2), bcs=["inflow","outflow","inflow","inflow"])
mesh = Mesh(geo.GenerateMesh(maxh=0.2))
mesh.Curve(3)
Draw (mesh)


from ngs_templates.NavierStokes import NavierStokes
from ngsolve.internal import visoptions

# angle of atack
alpha = 5 * math.pi/180
timestep = 0.001
navstokes = NavierStokes (mesh, nu=0.0005, order=3, timestep = timestep,
                              inflow="inflow", outflow="outlet", wall="wall",
                              uin=CoefficientFunction( (math.cos(alpha),math.sin(alpha)) ))
                              
navstokes.SolveInitial()
Draw (navstokes.pressure, mesh, "pressure")
Draw (navstokes.velocity, mesh, "velocity")
visoptions.scalfunction='velocity:0'


tend = 10
t = 0

with TaskManager():
    while t < tend:
        print (t)
        navstokes.DoTimeStep()
        t = t+timestep
        Redraw(blocking=True)



