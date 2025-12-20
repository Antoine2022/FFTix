import numpy as np
import time

from Operators_elastic import *
from microstructure_periodique import *
from voxelize import *

@njit(parallel=True)
def initialize_tau(Ci,C0,tau_field,field,E,N,dim,NN):
    for i in prange(NN):
        n=compute_n(i,N,dim)
        tau_i=np.dot(Ci[n[0],n[1],n[2]]-C0,E)
        tau_field[n[0],n[1],n[2]]=tau_i
        field[n[0],n[1],n[2]]=tau_i

@njit(parallel=True)
def update_tau(Ci,C0,field,tau_field,E,sig,N,dim,NN):
    energie=0.
    for i in prange(NN):
        n=compute_n(i,N,dim)
        eps_i=E+field[n[0],n[1],n[2]]
        C_i=Ci[n[0],n[1],n[2]]
        tau_i=np.dot(C_i-C0,eps_i)
        tau_field[n[0],n[1],n[2]]=tau_i
        sig_i=np.dot(C_i,eps_i)
        sig+=sig_i/NN
        energie+=0.5*np.dot(sig_i,eps_i)/NN
        field[n[0],n[1],n[2]]=tau_i
    return energie,sig



def BS(Ej,N,L,C0,Cref,Ci,numThreads,prec,name):
    set_num_threads(12)
    dim=len(N)
    N=np.array(N)
    L=np.array(L)
    NN=1
    for i in range(dim):
        NN*=N[i]
    d=int(dim*(dim+1)/2)
    tau_field=np.zeros(tuple(N)+(d,))
    it=0
    err=2*prec
    err2=20*prec
    energie=-Cref[0,0]
    t0=time.time()
    print("init fft fields")
    field,field_fourier,fft,ifft,K = initialize_fft(N,dim,numThreads)
    eps_field_C,eps_field_fourier,fft_e,ifft_e,K = initialize_fft(N,dim,numThreads)
    sig_field_C,sig_field_fourier,fft_s,ifft_s,K = initialize_fft(N,dim,numThreads)
    print("init tau field")
    initialize_tau(Ci,Cref,tau_field,field,Ej,N,dim,NN)
    print("init Gamma")
    NK=1
    for i in range(dim):
        NK*=K[i]
    Gamma_field=np.zeros(tuple(K)+(d,d),dtype=np.complex128)
    initialize_Gamma(Cref,K,N,L,dim,NK,NN,Gamma_field)
    print("init done")
    #file=open('./data/'+name,'a')
    while True:#abs(err2)>prec:# or abs(err2)>10*prec:
        print("temps :",time.time()-t0)
        t0=time.time()
        field=spatial_convolution(field,field_fourier,fft,ifft,Gamma_field,dim,K,NK)
        res=residual(Ci,field,Ej,C0,Cref,N,dim,NN,sig_field_fourier,sig_field_C,eps_field_fourier,eps_field_C,fft_s,ifft_s,fft_e,ifft_e,Gamma_field,K,NK)
        prev_energ=energie
        sig=np.zeros((d,))
        energie,sig=update_tau(Ci,Cref,field,tau_field,Ej,sig,N,dim,NN)
        #file.write(str(it)+" "+str(2*energie)+"\n")
        it+=1
        err=np.abs((energie-prev_energ)/prev_energ)
        err2=np.abs(energie-prev_energ)
        if True:#(it/10==int(it/10) or it==2 or it==4 or it==6):
            print(it,err,err2,2*energie,sig)
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
C0=3*0.4*J+2*0.2*K
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
# c_i=100
# NN=N0**dim
# for i in np.ndindex(tuple(N)):  
# #for i in range(NN):
# #    n=compute_n(i,N,dim)
#     if phase[i]==0:
#         Ci[i]=C0
#     else:
#         cpt+=1
#         Ci[i]=c_i*C0
# print("frac :",cpt/NN)
print('compute')
#name='BS_hard_spheres_N0='+str(N0)+'_c='+str(c_i)+'_frac='+str(0.86)+'.txt'
#BS(Ej,N,L,(c_i+1)/2.*C0,Ci,12,1e-7,name)
#phase,N,NN=raffine(phase,N,dim,NN)
#cpt=0
#Ci=np.zeros(tuple(N)+(d,d))
#for i in np.ndindex(tuple(N)):  
#for i in range(NN):
#    n=compute_n(i,N,dim)
#    if phase[i]==0:
#        Ci[i]=C0
#    else:
#        cpt+=1
#        Ci[i]=1000*C0
#print("frac :",cpt/NN)
#plt.figure()
#plt.imshow(phase[:,:,0])
#plt.show()
#plt.close()
#BS(Ej,N,L,500.5*C0,Ci,15,1e-5,name)
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
# f0=1-2*frac
# kV=f0*k0+frac*kp+frac*kinc
# muV=f0*mu0+frac*mup+frac*muinc
# muHill=(4-5*nu0)/15/mu0/(1-nu0)
# kHill=(1-2*nu0)/6/mu0/(1-nu0)
# muPinc=1/(1/2/(muinc-mu0)+(1-frac)*muHill)
# muPp=1/(1/2/(mup-mu0)+(1-frac)*muHill)
# kPinc=1/(1/3/(kinc-k0)+(1-frac)*kHill)
# kPp=1/(1/3/(kp-k0)+(1-frac)*kHill)
# Pinc=kPinc*J+muPinc*K
# Pp=kPp*J+muPp*K
# print(C0[0,0]+frac*Pinc[0,0]+frac*Pp[0,0])
# kHS=k0+frac*(kPp+kPinc)/3
# muHS=mu0+frac*(muPp+muPinc)/2
# print("EHS",kHS+4/3*muHS)
# print("EV",kV+4/3*muV)
name='BS_double_inclusion_N0=32_c=1000000'
BS(Ej,N,L,C0,Cref,Ci,12,1e-10,name)