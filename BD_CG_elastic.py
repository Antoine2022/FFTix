import numpy as np
import time

from Operators_elastic import *
from microstructure_periodique import *
from voxelize import *

@njit(parallel=True)
def initialize_E(E_field,Ej,N,dim,NN):
    for i in prange(NN):
        n=compute_n(i,N,dim)
        E_field[n[0],n[1],n[2]]=Ej

@njit(parallel=True)
def initialize_Mi(Mi,Ci,C0,N,dim,NN):
    for i in prange(NN):
        n=compute_n(i,N,dim)
        Mi[n[0],n[1],n[2]]=np.linalg.inv(Ci[n[0],n[1],n[2]]-C0)

@njit(parallel=True)
def initialize_p(p_field,Ej,Mi,N,dim,NN):
    for i in prange(NN):
        n=compute_n(i,N,dim)
        p_field[n[0],n[1],n[2]]=np.dot(np.linalg.inv(Mi[n[0],n[1],n[2]]),Ej)
    
@njit(parallel=True)
def real_product(eps_field,Mi,tau_field,dim,N,NN):
    for i in prange(NN):
        n=compute_n(i,N,dim)
        eps_field[n[0],n[1],n[2]]=np.dot(Mi[n[0],n[1],n[2]],tau_field[n[0],n[1],n[2]])

@njit(parallel=True)
def update_tau(p,tau_field,N,dim,NN,d):
    for i in prange(NN):
        n=compute_n(i,N,dim)
        tau_field[n[0],n[1],n[2]]=p[i*d:(i+1)*d]

@njit(parallel=True)
def update_eps(eps_field,Gt_field,N,dim,NN):
    for i in prange(NN):
        n=compute_n(i,N,dim)
        eps_field[n[0],n[1],n[2]]=eps_field[n[0],n[1],n[2]]-Gt_field[n[0],n[1],n[2]]

def A_dot(Mi,eps_field,tau_field,tau_fourier,Gamma_field,fft,ifft,K,dim,NK,N,NN):
    print("A_dot...")
    real_product(eps_field,Mi,tau_field,dim,N,NN)
    Gt_field=spatial_convolution(tau_field,tau_fourier,fft,ifft,Gamma_field,dim,K,NK)
    update_eps(eps_field,Gt_field,N,dim,NN)
    return np.ravel(eps_field)


def BD_CG(Ej,N,L,C0,Ci,numThreads,prec,name):
    set_num_threads(15)
    dim=len(N)
    N=np.array(N)
    L=np.array(L)
    NN=1
    for i in range(dim):
        NN*=N[i]
    d=int(dim*(dim+1)/2)
    it=0
    err=2*prec
    err2=20*prec
    energie=-C0[0,0]
    t0=time.time()
    print("init fft fields")
    tau_field,tau_fourier,fft,ifft,K = initialize_fft(N,dim,numThreads)
    print("init Gamma")
    NK=1
    for i in range(dim):
        NK*=K[i]
    Gamma_field=np.zeros(tuple(K)+(d,d),dtype=np.complex128)
    initialize_Gamma(C0,K,N,L,dim,NK,NN,Gamma_field)
    print("init E field")
    E_field=np.zeros((tuple(N)+(d,)))
    eps_field=np.zeros((tuple(N)+(d,)))
    initialize_E(E_field,Ej,N,dim,NN)
    b=np.ravel(E_field)
    print("init Mi")
    Mi=np.zeros((tuple(N)+(d,d)))
    initialize_Mi(Mi,Ci,C0,N,dim,NN)
    print("init p0")
    p_field=np.zeros((tuple(N)+(d,)))
    initialize_p(p_field,Ej,Mi,N,dim,NN)
    p=np.ravel(p_field)
    print("init done")
    print("execute conjugate gradient")
    x=p
    Ap=A_dot(Mi,eps_field,tau_field,tau_fourier,Gamma_field,fft,ifft,K,dim,NK,N,NN)
    r=b-Ap
    p=r
    err=2*prec
    it=0
    CV=0.
    for i in np.ndindex(tuple(N)):
        CV+=Ci[i][0,0]/NN
    print(CV)
    energie=-CV
    #file=open('./data/'+name,'a')
    while err>prec:
        update_tau(p,tau_field,N,dim,NN,d)
        Ap=A_dot(Mi,eps_field,tau_field,tau_fourier,Gamma_field,fft,ifft,K,dim,NK,N,NN)
        alpha=np.dot(r,r)/np.dot(p,Ap)
        x+=alpha*p
        rk1=r-alpha*Ap
        print(np.linalg.norm(rk1))
        prev_energie=energie
        energie=compute_energie(Ci,eps_field,N,NN)
        err=np.abs((energie-prev_energie)/prev_energie)
        print(it,2*energie,err)
        #val=2*energie
        #file.write(str(it)+" "+str(val)+"\n")
        it+=1
        if err>prec:  
            beta=np.dot(rk1,rk1)/np.dot(r,r)
            p=rk1+beta*p
            r=rk1
    #file.flush()
    
    
    
    


Ej=np.array([1.,0.,0.,0.,0.,0.])
N0=128
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
phase=np.load("microstructures/micro_test_"+str(N0)+"_vox.npy")
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
BD_CG(Ej,N,L,(c_i+1)/2.*C0,Ci,15,1e-7,name)
