import numpy as np
from math import pi
from numba import jit, prange, njit
from utils import *


@jit
def Gamma_hat(k,C0,dim):
    C04=FromVoigt(C0,dim)
    k_=0.
    for i in range(dim):
        k_+=k[i]**2
    k_=np.sqrt(k_)
    nk=[]
    for i in range(dim):
        nk.append(k[i]/k_)
    nC0=nk[0]*C04[0,:,:,:]
    for i in range(1,dim):
        nC0+=nk[i]*C04[i,:,:,:]
    nC0n=nC0[:,:,0]*nk[0]
    for i in range(1,dim):
        nC0n+=nC0[:,:,i]*nk[i]
    inC0n= np.linalg.inv(nC0n)
    nnC0n= dyadic_g(nk,inC0n,dim)
    nnC0nn = dyadic_d(nnC0n,nk,dim)
    tens = sym(nnC0nn,dim)
    Gamma=ToVoigt(tens,dim)
    return Gamma


# Discretization of Gamma adopted by Moulinec Suquet 1994
@jit
def Gamma_hat_MS(C0,dim,n,N,L,Gamma):
    k=[]
    for i in range(dim):
        ki=n[i]/L[i]
        if n[i]>N[i]/2.:
            ki-=N[i]/L[i]
        k.append(2*np.pi*ki)
    Gamma+= Gamma_hat(k,C0,dim)
    return Gamma


# Discretization of Gamma adopted by Brisard Dormieux 2012
@jit
def Gamma_hat_BD(C0,dim,n,N,L,Gamma):
    p=[]
    for i in range(dim):
        p.append(-1)
    it=0
    while it<2**dim:
        ind=0
        val=it
        while ind<dim and val>0:
            q=int(val/2)
            r=val-2*q
            if r==1:
                p[ind]=0
                val=0
            else:
                p[ind]=-1
                ind+=1
                val=val/2
        k=[]
        for i in range(dim):
            k.append(2*np.pi*(n[i]+p[i]*N[i])/L[i])
        Gamma_h = Gamma_hat(k,C0,dim)
        G=1.
        for i in range(dim):
            G*=np.cos(L[i]/N[i]*k[i]/4)
        Gamma+=(G**2)*Gamma_h
        it+=1
    return Gamma

# Discretization of Gamma adopted by Willot 2015
@jit
def Gamma_hat_Wil(C0,dim,n,N,L,Gamma):
    k=[]
    for i in range(dim):
        k.append(N[i]/L[i]*np.tan(np.pi*n[i]/N[i]))
    Gamma+= Gamma_hat(k,C0,dim)
    return Gamma

# Green operator
@njit(parallel=True)
def initialize_Gamma(C0,N,L,dim,NN,Gamma_field,type):
    d=int(dim*(dim+1)/2)
    for i in prange(1,NN):
        n=compute_n(i,N,dim)
        Gamma=np.zeros((d,d),dtype=np.complex128)
        if type=="Willot":
            Gamma_field[n[0],n[1],n[2]]=Gamma_hat_Wil(C0,dim,np.array(n),N,L,Gamma)
        if type=="Moulinec-Suquet":
            Gamma_field[n[0],n[1],n[2]]=Gamma_hat_MS(C0,dim,np.array(n),N,L,Gamma)
        if type=="Brisard-Dormieux":
            Gamma_field[n[0],n[1],n[2]]=Gamma_hat_BD(C0,dim,np.array(n),N,L,Gamma)

# Stress Green operator
@njit(parallel=True)
def initialize_Delta(C0,N,L,dim,NN,Delta_field,type):
    d=int(dim*(dim+1)/2)
    Delta_field[0,0,0]=C0
    C0_cplx=C0.astype(np.complex128)
    for i in prange(1,NN):
        n=compute_n(i,N,dim)
        Gamma=np.zeros((d,d),dtype=np.complex128)
        if type=="Willot":
            Gamma=Gamma_hat_Wil(C0,dim,np.array(n),N,L,Gamma)
        if type=="Moulinec-Suquet":
            Gamma=Gamma_hat_MS(C0,dim,np.array(n),N,L,Gamma)
        if type=="Brisard-Dormieux":
            Gamma=Gamma_hat_BD(C0,dim,np.array(n),N,L,Gamma)
        Delta_field[n[0],n[1],n[2]]=C0_cplx-np.dot(C0_cplx,np.dot(Gamma,C0_cplx))


@njit(parallel=True)
def product(y,A,x,dim,N,NN):
    for i in prange(NN):
        n=compute_n(i,N,dim)
        y[n[0],n[1],n[2]]=np.dot(A[n[0],n[1],n[2]],x[n[0],n[1],n[2]])


def Fourier_convolution(x,Fourier_kernel,dim,N,NN):
    xfourier=np.fft.fftn(x,axes=range(dim))
    product(xfourier,Fourier_kernel,xfourier,dim,N,NN)
    return np.fft.ifftn(xfourier,axes=range(dim))

def norm2(x,d):
    x=x.reshape(-1,d)
    return np.trace(np.dot(np.transpose(x),x))
    


