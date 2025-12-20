import numpy as np
from math import pi
from numba import jit

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