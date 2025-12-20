import numpy as np
import time
from Operators import *
from numba import set_num_threads


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



def Strain_BS(Ej,N,L,C0,Cref,Ci,numThreads,prec,name):
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

def Stress_BS(Ej,N,L,C0,Cref,Ci,numThreads,prec,name):
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