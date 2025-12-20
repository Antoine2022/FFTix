import numpy as np
from spherocylinders import *
    
l=0.5 #semi_length
e=10 #aspect ratio
D=1 #side of the box
f=0.12 # target fraction (will be smaller after voxellization)

micro=generate_micro(D/2,l,l/e/2,f)
micro_name="micro_randomly_oriented"
np.save("./"+micro_name+".npy",micro)

from voxel import *
size=128
micro_v=voxelize_ell_n(micro,size,l,e,D)
write_vtk(micro_v,micro_name,size)

f=0.1 # target fraction (will be smaller after voxellization)
micro=generate_micro_aligned(D/2,l,l/e/2,f)
micro_name="micro_aligned"
np.save("./"+micro_name+".npy",micro)
micro_v=voxelize_ell(micro,size,l,e,D)
write_vtk(micro_v,micro_name,size)

from Schemes import *

Ej=np.array([1.,0.,0.,0.,0.,0.])
N0=32
N=[N0,N0,N0]
L0=1
L=[L0,L0,L0]
d=6
dim=3
I=np.eye(d)
J=np.zeros((d,d))
for i in range(dim):
    for j in range(dim):
        J[i,j]=1./3
K=I-J
C0=3*0.4*J+2*0.2*K
Ci=np.zeros(tuple(N)+(d,d))
l=0.2
e=1.
D=1.

phase=micro_v
cpt=0.
c_i=1e3

k0=13./6
mu0=1.
kinc=13./6*c_i
muinc=1*c_i
kp=13./6/c_i
mup=1./c_i
kref=13./6/2*(1/c_i+c_i)
muref=(1/c_i+c_i)/2.
C0=3*k0*J+2*mu0*K
Cinc=3*kinc*J+2*muinc*K
Cp=3*kp*J+2*mup*K
Cref=3*kref*J+2*muref*K
for i in np.ndindex(tuple(N)):
    if phase[i]==0:
        Ci[i]=C0
    if phase[i]==1:
        Ci[i]=Cinc
        cpt+=1.
    if phase[i]==2:
        Ci[i]=Cp
frac=cpt/N0**dim
name='BS_double_inclusion_N0=32_c=1000000'
Strain_BS(Ej,N,L,C0,Cref,Ci,12,1e-10,name)


C0=3*J+2*K
Ci=np.zeros(tuple(N)+(d,d))
l=0.2
e=1.
D=1.

cpt=0
c_i=1000
NN=N0**dim
for i in np.ndindex(tuple(N)):  
    if phase[i]==0:
        Ci[i]=C0
    else:
        cpt+=1
        Ci[i]=c_i*C0
print("frac :",cpt/NN)
print('compute')
name='BD_CG_hard_spheres_N0='+str(N0)+'_c='+str(c_i)+'_frac='+str(0.13)+'.txt'
Stress_BS(Ej,N,L,(c_i+1)/2.*C0,Ci,15,1e-7,name)

c_i=1e3
c_i_2=1e3
k0=13./6
mu0=1.
kinc=13./6*c_i_2
muinc=c_i_2
kp=13./6/c_i
mup=1/c_i
kref=13./6#*np.sqrt(c_i_2/c_i)
muref=1#*np.sqrt(c_i_2/c_i)
C0=3*k0*J+2*mu0*K
Cinc=3*kinc*J+2*muinc*K
Cp=3*kp*J+2*mup*K
Cref=3*kref*J+2*muref*K
for i in np.ndindex(tuple(N)):
    if phase[i]==0:
        Ci[i]=C0
    if phase[i]==1:
        Ci[i]=Cinc
    if phase[i]==2:
        Ci[i]=Cp

name='EM_double_inclusion_N0=32_c=1000000'
EM(Ej,N,L,C0,Cref,Ci,12,1e-10,name)
