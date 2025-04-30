# Imports
import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy.linalg as la
from matplotlib.animation import PillowWriter

# Satellitt
mass = 270
lx, ly, lz = 0.5, 0.5, 0.35
I = (mass/12)*np.diag([ly**2+lz**2, lx**2+lz**2, lx**2+ly**2])
I_inv = np.linalg.inv(I)

# Initial condisions
perturb = np.deg2rad(5)
q0 = R.from_euler('xyz', [perturb, 0, 0]).as_quat()
q0 = np.roll(q0, 1)           # [w,x,y,z]
omega0 = np.array([0.01, -0.02, 0.015])
y0 = np.concatenate((q0, omega0))

# 
wn, zeta = 0.5, 2.0
K1 = np.eye(3)*(wn**2)
K2 = np.eye(3)*(2*zeta*wn)

def quat_mult(a,b):
    w0,x0,y0,z0 = a
    w1,x1,y1,z1 = b
    return np.array([
      w0*w1 - x0*x1 - y0*y1 - z0*z1,
      w0*x1 + x0*w1 + y0*z1 - z0*y1,
      w0*y1 - x0*z1 + y0*w1 + z0*x1,
      w0*z1 + x0*y1 - y0*x1 + z0*w1
    ])

def S(v):
    return np.array([[0,-v[2], v[1]],
                     [v[2], 0,-v[0]],
                     [-v[1],v[0],  0]])

def omega_mat(o):
    wx,wy,wz = o
    return np.array([[0,-wx,-wy,-wz],
                     [wx,  0, wz,-wy],
                     [wy,-wz,  0, wx],
                     [wz, wy,-wx,  0]])

def backstep_ctrl(t,y):
    q = y[:4]/la.norm(y[:4])
    om= y[4:]
    qc= np.array([q[0],-q[1],-q[2],-q[3]])
    qe= quat_mult([1,0,0,0], qc); qe/= la.norm(qe)
    eta,epsilon= qe[0], qe[1:]
    G= eta*np.eye(3)+S(epsilon)
    alpha1= -K1@(G@epsilon)
    z2= om-alpha1
    return -K2@z2 - G@epsilon + np.cross(om,I@om)

def dynamics(t,y):
    q = y[:4]/la.norm(y[:4])
    om= y[4:]
    dq   = 0.5*omega_mat(om)@q
    dom  = I_inv@(backstep_ctrl(t,y)-np.cross(om,I@om))
    return np.concatenate((dq,dom))

# Simulering av attitude settling
T_settle = 100
N_frames = 200
t_settle = np.linspace(0,T_settle,N_frames)
sol = solve_ivp(dynamics,(0,T_settle),y0,t_eval=t_settle,rtol=1e-9,atol=1e-9)
Qmats = sol.y[:4].T
rots  = R.from_quat(np.roll(Qmats,-1,axis=1))
R_settle = rots.as_matrix()

# Sweep
M = 100 # Antall bilder per sweep
phi_max = np.deg2rad(30) # max sweep
phis = np.linspace(-phi_max,phi_max,M)
sweep_R = []
# Attitude matrix:
Rf = R_settle[-1]
for phi in phis:
    Ry = R.from_euler('y', phi).as_matrix()
    sweep_R.append(Rf @ Ry)
# Tilbake til original pos
for phi in phis[::-1]:
    Ry = R.from_euler('y', phi).as_matrix()
    sweep_R.append(Rf @ Ry)

# Kombiner forskjellige operasjoner, e.g settle + sweep
R_all = sweep_R
#R_all = np.vstack([R_settle, sweep_R])

# Satellit tube og solceller
hx,hy,hz = 0.25,0.25,0.175
body = np.array([[-hx,-hy,-hz],[-hx, hy,-hz],[ hx, hy,-hz],[ hx,-hy,-hz],
                 [-hx,-hy, hz],[-hx, hy, hz],[ hx, hy, hz],[ hx,-hy, hz]])
faces = [[0,1,2,3],[4,5,6,7],[0,1,5,4],[2,3,7,6],[1,2,6,5],[0,3,7,4]]
L=1.8
left  = np.array([[-hx-L,-hy,0],[-hx,-hy,0],[-hx, hy,0],[-hx-L, hy,0]])
right = np.array([[ hx,-hy,0],[ hx+L,-hy,0],[ hx+L, hy,0],[ hx, hy,0]])
rotX  = np.array([[1,0,0],[0,0,-1],[0,1,0]])
left,right = left@rotX.T, right@rotX.T

# Animasjon
fig = plt.figure(figsize=(6,6))
ax  = fig.add_subplot(111, projection='3d')
ax.set_box_aspect([1,1,1])
ax.set_xlim(-1,1); ax.set_ylim(-1,1); ax.set_zlim(-1,1)
ax.set_title("Simulering av satelitt manøvrering")

body_poly = Poly3DCollection([],facecolors='lightblue',edgecolors='k',alpha=0.9)
lp = Poly3DCollection([],facecolors='orange',edgecolors='k')
rp = Poly3DCollection([],facecolors='orange',edgecolors='k')
# boresight “swath” linje
swath_line, = ax.plot([],[],[],'r-',lw=2)

ax.add_collection3d(body_poly)
ax.add_collection3d(lp)
ax.add_collection3d(rp)

def update(i):
    Rm = R_all[i]
    # satellitt
    verts = body @ Rm.T
    body_poly.set_verts([[verts[j] for j in f] for f in faces])
    lp.set_verts([left @ Rm.T])
    rp.set_verts([right@ Rm.T])
    # boresight, body-z axis [0,0,1] i inertial-ref
    b = Rm @ np.array([0,0,1])
    # boresight linje
    swath_line.set_data([0,b[0]],[0,b[1]])
    swath_line.set_3d_properties([0,b[2]])
    return body_poly,lp,rp,swath_line

total = len(R_all)
ani = animation.FuncAnimation(fig,update,frames=total,interval=50,blit=False)
writer = PillowWriter(fps=20)

# Lagre gif:
ani.save(r'C:\Users\wighu\OneDrive\Dokumenter\Satelitt\Satelitt\sweep .gif', writer=writer)
