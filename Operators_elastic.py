import time
import numpy as np
from math import pi
from numba import jit, prange, njit, set_num_threads
import pyfftw
from multiprocessing import Pool, Lock, Process, Manager
import matplotlib.pyplot as plt

@jit
def Indices(ii,dim):
    if ii==0:
        return 0,0
    if ii==1:
        return 1,1
    if ii==2:
        if dim==3:
            return 2,2
        if dim==2:
            return 0,1
    if ii==3 and dim==3:
        return 0,1
    if ii==4 and dim==3:
        return 0,2
    if ii==5 and dim==3:
        return 1,2
    else:
        return -1,-1
    
@jit
def VoigtIndex(i,j,dim):
    if i==0 and j==0:
        return 0
    if i==1 and j==1:
        return 1
    if (i==0 and j==1) or (i==1 and j==0):
        if dim==3:
            return 3
        if dim==2:
            return 2
    if i==2 and j==2 and dim==3:
        return 2
    if ((i==0 and j==2) or (i==2 and j==0)) and dim==3:
        return 4
    if ((i==1 and j==2) or (i==2 and j==1)) and dim==3:
        return 5
    else:
        return -1

# WARNING : A must have the minor symmetries
@jit 
def ToVoigt(A,dim):
    d=int(dim*(dim+1)/2)
    A_=np.zeros((d,d))
    for ii in range(d):
        for jj in range(d):
            i,j=Indices(ii,dim)
            k,l=Indices(jj,dim)
            A_[ii,jj]=A[i,j,k,l]
            if ii>=dim:
                A_[ii,jj]*=np.sqrt(2)
            if jj>=dim:
                A_[ii,jj]*=np.sqrt(2)
    return A_

@jit 
def FromVoigt(A,dim):
    A_=np.zeros((dim,dim,dim,dim))
    for i in range(dim):
        for j in range(i,dim):
            for k in range(dim):
                for l in range(k,dim):
                    ii=VoigtIndex(i,j,dim)
                    jj=VoigtIndex(k,l,dim)
                    fac=1.
                    if ii>=dim:
                        fac/=np.sqrt(2)
                    if jj>=dim:
                        fac/=np.sqrt(2)
                    A_[i,j,k,l]=fac*A[ii,jj]
                    A_[j,i,k,l]=fac*A[ii,jj]
                    A_[i,j,l,k]=fac*A[ii,jj]
                    A_[j,i,l,k]=fac*A[ii,jj]
    return A_

@jit
def dyadic_g(a,B,dim):
    d=np.zeros((dim,dim,dim))
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                d[i,j,k]=a[i]*B[j,k]
    return d 

@jit
def dyadic_d(A,b,dim):
    d=np.zeros((dim,dim,dim,dim))
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                for l in range(dim):
                    d[i,j,k,l]=A[i,j,k]*b[l]
    return d

@jit
def sym(A,dim):
    d=np.zeros((dim,dim,dim,dim))
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                for l in range(dim):
                    d[i,j,k,l]=(A[i,j,k,l]+A[j,i,k,l]+A[i,j,l,k]+A[j,i,l,k])/4.
    return d


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

@jit
def Gamma_hat_BD(C0,dim,n,N,L,Gamma):
    k=[]
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
        for i in range(dim):
            k.append(2*np.pi*(n[i]+p[i]*N[i])/L[i])
        Gamma_h = Gamma_hat(k,C0,dim)
        G=1.
        for i in range(dim):
            G*=np.cos(L[i]/N[i]*k[i]/4)
        Gamma+=G**2*Gamma_h
        it+=1
    return Gamma


@jit
def Gamma_hat_Wil(C0,dim,n,N,L,Gamma):
    k=[]
    for i in range(dim):
        k.append(N[i]/L[i]*np.tan(np.pi*n[i]/N[i]))
    Gamma+= Gamma_hat(k,C0,dim)
    return Gamma

@jit
def compute_n(i,N,dim):
    n=[]
    for p in range(dim):
        n.append(0)
    i_=i
    for ind in range(1,dim+1):
        q=i_
        for j in range(ind,dim):
            q/=N[j]
        q=int(q)
        n[ind-1]=q
        for j in range(ind,dim):
            q*=N[j]
        i_-=q
    return n



def initialize_fft(N,dim,numThreads):
    d=int(dim*(dim+1)/2)
    K=N.copy()
    K[-1] = N[-1]//2+1
    Npadded = N.copy()
    Npadded[-1] = K[-1]*2
    full_array = pyfftw.n_byte_align_empty(np.append(Npadded,d), pyfftw.simd_alignment,'float64')
    field = full_array[...,:N[-1],:]
    field_fourier = full_array.ravel().view('complex128').reshape(np.append(K,d))
    fft = pyfftw.FFTW(field, field_fourier, axes=range(dim),direction='FFTW_FORWARD',threads=numThreads)
    ifft = pyfftw.FFTW(field_fourier, field, axes=range(dim),direction='FFTW_BACKWARD',threads=numThreads)
    return field,field_fourier,fft,ifft,K

def initialize_fft_cplx(N,dim,numThreads):
    d=int(dim*(dim+1)/2)
    K=N.copy()
    K[-1] = N[-1]
    Npadded = N.copy()
    Npadded[-1] = K[-1]
    full_array = pyfftw.n_byte_align_empty(np.append(Npadded,d), pyfftw.simd_alignment,'complex128')
    field = full_array[...,:N[-1],:]
    field_fourier = full_array.ravel().view('complex128').reshape(np.append(K,d))
    fft = pyfftw.FFTW(field, field_fourier, axes=range(dim),direction='FFTW_FORWARD',threads=numThreads)
    ifft = pyfftw.FFTW(field_fourier, field, axes=range(dim),direction='FFTW_BACKWARD',threads=numThreads)
    return field,field_fourier,fft,ifft,K


@njit(parallel=True)
def initialize_Gamma(C0,K,N,L,dim,NK,NN,Gamma_field):
    d=int(dim*(dim+1)/2)
    for i in prange(1,NK):
        n=compute_n(i,K,dim)
        Gamma=np.zeros((d,d),dtype=np.complex128)
        Gamma_field[n[0],n[1],n[2]]=Gamma_hat_Wil(C0,dim,np.array(n),N,L,Gamma)

@njit(parallel=True)
def initialize_G(K,dim,NK,G_field,N,L,Cref,Crefr,I):
    d=int(dim*(dim+1)/2)
    G_field[0,0,0]=I
    for i in prange(1,NK):
        n=compute_n(i,K,dim)
        Gamma=np.zeros((d,d),dtype=np.complex128)
        G_field[n[0],n[1],n[2]]=I-Gamma_hat_Wil(I,dim,np.array(n),N,L,Gamma)

@njit(parallel=True)
def fourier_product(y,x,A_fourier,dim,K,NK):
    for i in prange(NK):
        n=compute_n(i,K,dim)
        y[n[0],n[1],n[2]]=A_fourier[n[0],n[1],n[2]].dot(x[n[0],n[1],n[2]])

@njit(parallel=True)
def fourier_product_with_rigid(y,x,phase,A_fourier,dim,K,NK):
    for i in prange(NK):
        n=compute_n(i,K,dim)
        if phase[n[0],n[1],n[2]]>=0:
            y[n[0],n[1],n[2]]=A_fourier[n[0],n[1],n[2]].dot(x[n[0],n[1],n[2]])
        else:
            y[n[0],n[1],n[2]]=x[n[0],n[1],n[2]]


@njit(parallel=True)
def fourier_product_bis(x,A_fourier,dim,K,NK):
    for i in prange(NK):
        n=compute_n(i,K,dim)
        x[n[0],n[1],n[2]]=((A_fourier[n[0],n[1],n[2]]).transpose()).dot(x[n[0],n[1],n[2]])


def spatial_convolution(x,yfourier,fft,ifft,Gamma_field,dim,K,NK):
    xfourier=fft(x)
    fourier_product(yfourier,xfourier,Gamma_field,dim,K,NK)
    return -ifft(yfourier)

def spectral_convolution(xfourier,ifft,Gamma_field,dim,K,NK):
    fourier_product(xfourier,xfourier,Gamma_field,dim,K,NK)
    return -ifft(xfourier)

def spectral_convolution_with_rigid(xfourier,phase,ifft,Gamma_field,dim,K,NK):
    fourier_product_with_rigid(xfourier,xfourier,phase,Gamma_field,dim,K,NK)
    return -ifft(xfourier)


def compute_energie(Ci,eps_field,N,NN):
    energie=0.
    for i in np.ndindex(tuple(N)):
        energie+=0.5*np.dot(eps_field[i],np.dot(Ci[i],eps_field[i]))/NN
    return energie

@njit(parallel=True)
def compute_sig(Ci,sig,eps_field,N,dim,NN):
    for i in prange(NN):
        n=compute_n(i,N,dim)
        sig+=np.dot(Ci[n[0],n[1],n[2]],eps_field[n[0],n[1],n[2]])/NN
    return sig


@njit(parallel=True)
def sub(field_0,field_1,field_2,N,dim,NN):
    for i in prange(NN):
        n=compute_n(i,N,dim)
        field_0[n[0],n[1],n[2]]=field_1[n[0],n[1],n[2]]-field_2[n[0],n[1],n[2]]

@njit(parallel=True)
def C0_norm2(C0,field,N,dim,NN):
    nor2=0.
    for i in prange(NN):
        n=compute_n(i,N,dim)
        nor2+=np.dot(field[n[0],n[1],n[2]],np.dot(C0,field[n[0],n[1],n[2]]))/NN
    return nor2


@njit(parallel=True)
def init_fields(Ci,field,eps_field,eps_field_C,sig_field_C,Ej,N,dim,NN,C0,Cref):
    S0=np.linalg.inv(C0)
    for i in prange(NN):
        n=compute_n(i,N,dim)
        eps_field_C[n[0],n[1],n[2]]=-np.dot(Cref,field[n[0],n[1],n[2]])
        sig_field_C[n[0],n[1],n[2]]=np.dot(Cref,np.dot(S0,np.dot(Ci[n[0],n[1],n[2]],Ej+field[n[0],n[1],n[2]])))
        eps_field[n[0],n[1],n[2]]=-field[n[0],n[1],n[2]]

@njit(parallel=True)
def init_fields_LF(Si,field,eps_field,eps_field_C,sig_field_C,Ej,N,dim,NN,C0,Cref):
    S0=np.linalg.inv(C0)
    for i in prange(NN):
        n=compute_n(i,N,dim)
        eps_field_C[n[0],n[1],n[2]]=np.dot(Cref,Ej-np.dot(Si[n[0],n[1],n[2]],field[n[0],n[1],n[2]]))
        sig_field_C[n[0],n[1],n[2]]=np.dot(Cref,np.dot(S0,field[n[0],n[1],n[2]]))
        eps_field[n[0],n[1],n[2]]=Ej-np.dot(Si[n[0],n[1],n[2]],field[n[0],n[1],n[2]])

def residual(Ci,field,Ej,C0,Cref,N,dim,NN,sig_field_fourier,sig_field_C,eps_field_fourier,eps_field_C,fft_s,ifft_s,fft_e,ifft_e,Gamma_field,K,NK):
    d=int(dim*(dim+1)/2)
    eps_field=np.zeros(tuple(N)+(d,))
    init_fields(Ci,field,eps_field,eps_field_C,sig_field_C,Ej,N,dim,NN,C0,Cref)
    sig_field_C=-spatial_convolution(sig_field_C,sig_field_fourier,fft_s,ifft_s,Gamma_field,dim,K,NK)
    nC2=C0_norm2(C0,sig_field_C,N,dim,NN)
    eps_field_C=-spatial_convolution(eps_field_C,eps_field_fourier,fft_e,ifft_e,Gamma_field,dim,K,NK)
    eps_field_S=np.zeros(tuple(N)+(d,))
    sub(eps_field_S,eps_field,eps_field_C,N,dim,NN)
    nS2=C0_norm2(C0,eps_field_S,N,dim,NN)
    return np.sqrt((nS2+nC2)/np.dot(Ej,np.dot(C0,Ej)))

def residual_EP(Ci,field,Ej,C0,Cref,N,dim,NN,sig_field_fourier,sig_field_C,eps_field_fourier,eps_field_C,fft_s,ifft_s,fft_e,ifft_e,Gamma_field,K,NK):
    d=int(dim*(dim+1)/2)
    Cref_=Cref
    Cref=np.zeros((d,d),dtype=np.complex128)
    for i in range(d):
        for j in range(d):
            Cref[i,j]=complex(Cref_[i,j])
    eps_field=np.zeros(tuple(N)+(d,),dtype=np.complex128)
    init_fields(Ci,field,eps_field,eps_field_C,sig_field_C,Ej,N,dim,NN,C0,Cref)
    sig_field_C=-spatial_convolution(sig_field_C,sig_field_fourier,fft_s,ifft_s,Gamma_field,dim,K,NK)
    nC2=C0_norm2(C0,sig_field_C,N,dim,NN)
    eps_field_C=-spatial_convolution(eps_field_C,eps_field_fourier,fft_e,ifft_e,Gamma_field,dim,K,NK)
    eps_field_S=np.zeros(tuple(N)+(d,),dtype=np.complex128)
    sub(eps_field_S,eps_field,eps_field_C,N,dim,NN)
    nS2=C0_norm2(C0,eps_field_S,N,dim,NN)
    return np.sqrt((nS2+nC2)/np.dot(Ej,np.dot(C0,Ej)))

def residual_LF(Si,field,Ej,C0,Cref,N,dim,NN,sig_field_fourier,sig_field_C,eps_field_fourier,eps_field_C,fft_s,ifft_s,fft_e,ifft_e,Gamma_field,K,NK):
    d=int(dim*(dim+1)/2)
    Cref_=Cref
    Cref=np.zeros((d,d),dtype=np.complex128)
    for i in range(d):
        for j in range(d):
            Cref[i,j]=complex(Cref_[i,j])
    eps_field=np.zeros(tuple(N)+(d,),dtype=np.complex128)
    init_fields_LF(Si,field,eps_field,eps_field_C,sig_field_C,Ej,N,dim,NN,C0,Cref)
    sig_field_C=-spatial_convolution(sig_field_C,sig_field_fourier,fft_s,ifft_s,Gamma_field,dim,K,NK)
    nC2=C0_norm2(C0,sig_field_C,N,dim,NN)
    eps_field_C=-spatial_convolution(eps_field_C,eps_field_fourier,fft_e,ifft_e,Gamma_field,dim,K,NK)
    eps_field_S=np.zeros(tuple(N)+(d,),dtype=np.complex128)
    sub(eps_field_S,eps_field,eps_field_C,N,dim,NN)
    nS2=C0_norm2(C0,eps_field_S,N,dim,NN)
    return np.sqrt((nS2+nC2)/np.dot(Ej,np.dot(C0,Ej)))