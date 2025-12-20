import numpy as np
import time

from Operators_elastic import *
from microstructure_periodique import *
from voxelize import *


@njit(parallel=True)
def initialize_z_tau(Ci,C0,tau_field,field,E,N,dim,NN):
    for i in prange(NN):
        n=compute_n(i,N,dim)
        tau_i=np.array([0.,0.,0.,0.,0.,0.])
        tau_field[n[0],n[1],n[2]]=tau_i
        C_i=Ci[n[0],n[1],n[2]]
        Mi=np.linalg.inv(C_i+C0)
        z_tau=np.dot(np.dot(C_i-C0,Mi),tau_i)
        field[n[0],n[1],n[2]]=z_tau


@njit(parallel=True)
def update_tau(Ci,C0,field,tau_field,E,I,N,dim,NN):
    norm=0.
    for i in prange(NN):
        n=compute_n(i,N,dim)
        C_i=Ci[n[0],n[1],n[2]]
        Mi=np.linalg.inv(C_i+C0)
        tau_i=tau_field[n[0],n[1],n[2]]
        tau_i_bis=np.dot(I-2*np.dot(C0,Mi),tau_i)+2*np.dot(C0,E+field[n[0],n[1],n[2]])
        norm+=np.linalg.norm(tau_i_bis-tau_i)/NN
        tau_field[n[0],n[1],n[2]]=tau_i_bis
        eps_i=np.dot(Mi,tau_i_bis)
        z_tau=np.dot(C_i-C0,eps_i)
        field[n[0],n[1],n[2]]=z_tau
    return norm


def compute_energie_EM(Ci,C0,tau_field,N,NN):
    energie=0.
    for i in np.ndindex(tuple(N)):
        C_i=Ci[i]
        eps_i=np.dot(np.linalg.inv(C_i+C0),tau_field[i])
        energie+=0.5*np.dot(eps_i,np.dot(C_i,eps_i))/NN
    return energie

def EM(Ej,N,L,C0,Cref,Ci,numThreads,prec,name):
    set_num_threads(12)
    dim=len(N)
    N=np.array(N)
    L=np.array(L)
    NN=1
    for i in range(dim):
        NN*=N[i]
    d=int(dim*(dim+1)/2)
    tau_field=np.zeros(tuple(N)+(d,))
    I=np.eye(d)
    it=0
    err=2*prec
    err2=20*prec
    t0=time.time()
    print("init fft fields")
    field,field_fourier,fft,ifft,K = initialize_fft(N,dim,numThreads)
    eps_field_C,eps_field_fourier,fft_e,ifft_e,K = initialize_fft(N,dim,numThreads)
    sig_field_C,sig_field_fourier,fft_s,ifft_s,K = initialize_fft(N,dim,numThreads)
    print("init tau field")
    initialize_z_tau(Ci,Cref,tau_field,field,Ej,N,dim,NN)
    print("init Gamma")
    NK=1
    for i in range(dim):
        NK*=K[i]
    Gamma_field=np.zeros(tuple(K)+(d,d),dtype=np.complex128)
    initialize_Gamma(Cref,K,N,L,dim,NK,NN,Gamma_field)
    print("init done")
    energie=-Cref[0,0]
    #file=open('./data/'+name,'a')
    while True:# abs(err2)>prec:# or abs(err)>10*prec:
        print("temps :",time.time()-t0)
        t0=time.time()
        field=spatial_convolution(field,field_fourier,fft,ifft,Gamma_field,dim,K,NK)
        res=residual(Ci,field,Ej,C0,Cref,N,dim,NN,sig_field_fourier,sig_field_C,eps_field_fourier,eps_field_C,fft_s,ifft_s,fft_e,ifft_e,Gamma_field,K,NK)
        nn=update_tau(Ci,Cref,field,tau_field,Ej,I,N,dim,NN)
        it+=1
        prev_energie=energie
        energie=compute_energie_EM(Ci,Cref,tau_field,N,NN)
        #file.write(str(it)+" "+str(2*energie)+"\n")
        err=(prev_energie-energie)/prev_energie
        err2=prev_energie-energie
        print(it,nn,err,err2,2*energie)
        print("res",res)
    #file.flush()


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
C0=3*J+2*K
Ci=np.zeros(tuple(N)+(d,d))
l=0.2
e=1.
D=1.
print("gen")
#micro=genere_micro(D/2,l,l/e/2,0.)
#np.save("microstructures/micro_test_"+str(N0)+".npy",micro)
#micro=np.load("microstructures/micro_test_"+str(N0)+".npy")
print("vox")
#phase=voxelize_ell(micro,N0,l,e,D)
#np.save("microstructures/micro_test_"+str(N0)+"_vox.npy",phase)
#phase=np.load("microstructures/micro_test_"+str(N0)+"_vox.npy")
phase=np.load("microstructures/double_contrast_32.npy")
#phase=np.load("microstructures/micro_hard_sph_exc=0.01_ratio=0.06_frac=0.3.npy")
# plt.figure()
# plt.imshow(phase[10,:,:])
# plt.show()
# plt.close()
cpt=0
NN=N0**dim
# c_i=10
# for i in np.ndindex(tuple(N)):
# #NN=N0**dim
# #for i in range(NN):
#     #n=compute_n(i,N,dim)
#     if phase[i]==0:
#         Ci[i]=C0
#         #Ci[tuple(n)]=C0
#     else:
#         cpt+=1
#         Ci[i]=c_i*C0
#         #Ci[tuple(n)]=100*C0
# print("frac :",cpt/NN)

print('compute')
#name='EM_hard_spheres_N0='+str(N0)+'_c='+str(c_i)+'_frac='+str(0.13)+'.txt'
#EM(Ej,N,L,np.sqrt(c_i)*C0,Ci,5,1e-5,name)
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
