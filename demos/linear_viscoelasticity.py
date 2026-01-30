import numpy as np
import mgis.behaviour as mgis_bv
import sys
sys.path.append("../")

micro_name="micro_randomly_oriented_voxelized"
phase=np.load("./"+micro_name+".npy")
N0 = len(phase)
print("size: ",N0)

from Non_linear_schemes import Non_linear_Strain_BS

N=[N0,N0,N0]
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
kref=15
muref=20
Cref=3*kref*J+2*muref*K

tf=10.
eps_max = tf*0.01
Nt = 20
Nt=int(Nt)

eps_list=np.linspace(0,eps_max,Nt+1)
precision = Nt * [1e-3]
dt = Nt * [tf/Nt]

lib_path = "./src/libmatrix-generic.dll"
bv_name1="matrixmaxwell"
bv_name2="matrixmaxwell"
hypothesis = mgis_bv.Hypothesis.Tridimensional
behaviour_m = mgis_bv.load(lib_path, bv_name1, hypothesis)
behaviour_i = mgis_bv.load(lib_path, bv_name2, hypothesis)

ngauss_i=0
ngauss_m=0
NN=N0**3
for i in np.ndindex(tuple(N)):  
    if phase[i]==1:
        ngauss_i+=1
    else:
        ngauss_m+=1
print("frac :",ngauss_i/NN)

data_m = mgis_bv.MaterialDataManager(behaviour_m, ngauss_m)
data_i = mgis_bv.MaterialDataManager(behaviour_i, ngauss_i)

for state_manager in [data_m.s0, data_m.s1]:
    mgis_bv.setMaterialProperty(state_manager, "ke", 1)
    mgis_bv.setMaterialProperty(state_manager, "Ge", 1)
    mgis_bv.setMaterialProperty(state_manager, "kv", 2)
    mgis_bv.setMaterialProperty(state_manager, "Gv", 0.5)
    mgis_bv.setExternalStateVariable(state_manager, "Temperature", 293.15)

for state_manager in [data_i.s0, data_i.s1]:
    mgis_bv.setMaterialProperty(state_manager, "ke", 30)
    mgis_bv.setMaterialProperty(state_manager, "Ge", 40)
    mgis_bv.setMaterialProperty(state_manager, "kv", 100)
    mgis_bv.setMaterialProperty(state_manager, "Gv", 100)
    mgis_bv.setExternalStateVariable(state_manager, "Temperature", 293.15)

behaviours = [data_m,data_i]

# For non linear behaviours, we provide an history E_history which is a uniform remote strain which evolves in time. Indeed, it would be heavy to provide a history of fields.
Ej=np.array([1.,0.,0.,0.,0.,0.])
E_history=np.zeros((Nt,d))
for i_t in range(Nt):
    E_history[i_t]=Ej*eps_list[i_t+1]


Non_linear_Strain_BS(E_history,dt,N,L,phase,behaviours,Cref,precision,"name","Willot")

    

