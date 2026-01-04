import numpy as np
import sys
sys.path.append("../")


micro_name="micro_randomly_oriented_voxelized"
phase=np.load("./"+micro_name+".npy")
N0 = len(phase)
print("size: ",N0)
cpt=0.
for i in np.ndindex(phase.shape):  
    if phase[i]==1:
        cpt+=1
print("frac: ",cpt/N0**3)


from Linear_schemes import *

N=[N0,N0,N0]
L0=1.
L=[L0,L0,L0]
dim=3
d=int(dim*(dim+1)/2)

# The schemes require a remote field E_field, which can be non uniform. Here we construct this field. (The implementation could be modified of course).
Ej=np.array([1.,0.,0.,0.,0.,0.])
Ej_field=np.zeros(tuple(N)+(d,))
for i in np.ndindex(tuple(N)):
    Ej_field[i]=Ej

I=np.eye(d)
J=np.zeros((d,d))
for i in range(dim):
    for j in range(dim):
        J[i,j]=1./3
K=I-J

# Strain-based scheme is better for rigid matrix with soft inclusions: c_i small !
c_i=0.01
k0=13./6
mu0=1.
kinc=k0*c_i
muinc=mu0*c_i
C0=3*k0*J+2*mu0*K
Cinc=3*kinc*J+2*muinc*K
Ci=np.zeros(tuple(N)+(d,d))

for i in np.ndindex(tuple(N)):
    if phase[i]==0:
        Ci[i]=C0
    if phase[i]==1:
        Ci[i]=Cinc

kref=k0*(1+c_i)/2.
muref=mu0*(1+c_i)/2
Cref=3*kref*J+2*muref*K
name='name'
Strain_BS(Ej_field,N,L,Cref,Ci,1e-4,name,"Moulinec-Suquet")  ## Types "Willot" and "Brisard-Dormieux" also possible

# Stress-based scheme is better for soft matrix with rigid inclusions: c_i high !
c_i=100
kinc=k0*c_i
muinc=mu0*c_i
Sinc=1/(3*kinc)*J+1/(2*muinc)*K
kref=2*k0*c_i/(1+c_i)
muref=2*mu0*c_i/(1+c_i)
Sref=1/(3*kref)*J+1/(2*muref)*K
name='name'
S0=1/(3*k0)*J+1/(2*mu0)*K
Si=np.zeros(tuple(N)+(d,d))

for i in np.ndindex(tuple(N)):
    if phase[i]==0:
        Si[i]=S0
    if phase[i]==1:
        Si[i]=Sinc

Stress_BS(Ej_field,N,L,Sref,Si,1e-4,name,"Willot")   ## Types "Moulinec-Suquet" and "Brisard-Dormieux" also possible

# For Eyre-Milton scheme, you have to provide the array of (Ci+Cref)^{-1}.
c_i=100
kinc=k0*c_i
muinc=mu0*c_i
Cinc=3*kinc*J+2*muinc*K
for i in np.ndindex(tuple(N)):
    if phase[i]==0:
        Ci[i]=C0
    if phase[i]==1:
        Ci[i]=Cinc
kref=k0*np.sqrt(c_i)
muref=mu0*np.sqrt(c_i)
Cref=3*kref*J+2*muref*K
Mi=np.zeros(tuple(N)+(d,d))
for i in np.ndindex(tuple(N)):
    Mi[i]=np.linalg.inv(Cref+Ci[i])
name='name'
EM(Ej_field,N,L,Cref,Ci,Mi,1e-4,name,"Brisard-Dormieux")  ## Types "Moulinec-Suquet" and "Willot" also possible
