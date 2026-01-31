import numpy as np
import mgis.behaviour as mgis_bv
import sys
sys.path.append("../")
import random

micro_name="polycrystal"
phase=np.load("./"+micro_name+".npy")
N0 = len(phase)
print("size: ",N0)

from Non_linear_schemes import Non_linear_Stress_BS

N=[N0,N0,N0]
NN=N0**3
L0=1.
L=[L0,L0,L0]
dim=3
d=6
I=np.eye(d)
J=np.zeros((d,d))
for i in range(dim):
    for j in range(dim):
        J[i,j]=1./3
K=I-J

# with linear stress basic scheme, the reference medium has to be softer than twice the minimum stiffness
# but for non linear schemes, the reference medium needs to be adapted to the stiffness, which depends
# on the magnitude of the resolved stress "tau" for this case
kref=0.1
muref=0.1
Cref=3*kref*J+2*muref*K
Sref=np.linalg.inv(Cref)

tf=1.
eps_rate_final = 20
Nt = 20
Nt=int(Nt)

eps_list=np.linspace(0,eps_rate_final,Nt+1)
precision = Nt * [0.001]
dt = Nt * [tf/Nt]

lib_path = "./src/libcrystal-generic.dll"
bv_name="crystalcrystalline"
hypothesis = mgis_bv.Hypothesis.Tridimensional
behaviour = mgis_bv.load(lib_path, bv_name, hypothesis)

Np=10
ngauss=np.zeros((Np,),dtype=int)
frac=np.zeros((Np,))
NN=N0**3
for i in np.ndindex(tuple(N)):
    ngauss[phase[i]]+=1
    frac[phase[i]]+=1/NN

print("number of voxel on each grain: ",ngauss)
print("volume fractions: ",frac)

behaviours=[]
for r in range(Np):
    data_r=mgis_bv.MaterialDataManager(behaviour, ngauss[r])
    for state_manager in [data_r.s0, data_r.s1]:
        mgis_bv.setMaterialProperty(state_manager, "r", r)
        mgis_bv.setMaterialProperty(state_manager, "tau0", 3+10*random.random())
        mgis_bv.setMaterialProperty(state_manager, "gamma0", 1)
        mgis_bv.setMaterialProperty(state_manager, "n", 1+2*random.random())
        mgis_bv.setExternalStateVariable(state_manager, "Temperature", 293.15)
    behaviours.append(data_r)

# For non linear behaviours, we provide an history E_history which is a uniform remote strain which evolves in time. Indeed, it would be heavy to provide a history of fields.
Ej=np.array([1.,-1.,0.,0.,0.,0.])
E_history=np.zeros((Nt,d))
for i_t in range(Nt):
    E_history[i_t]=Ej*eps_list[i_t+1]


Non_linear_Stress_BS(E_history,dt,N,L,phase,behaviours,Sref,precision,"name","Willot")

    

