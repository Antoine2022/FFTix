import numpy as np
import time
from Operators import *

# For strain-based basic scheme (energie is optional)
def update_tau_Strain(Ci,Cref,field,eps_field,N,dim,NN):
    d=int(dim*(dim+1)/2)
    product(field,Ci,eps_field,dim,N,NN)
    sig_resh=field.reshape(NN,d)
    eps_resh=eps_field.reshape(NN,d)
    Creps= eps_resh @ Cref
    Creps=Creps.reshape(eps_field.shape)
    field=field- Creps
    energie=0.5*np.trace(np.dot(np.transpose(sig_resh),eps_resh))/NN
    return np.mean(sig_resh,axis=0),energie,field

# For stress-based basic scheme (energie is optional)
def update_tau_Stress(Si,Sref,field,sig_field,N,dim,NN):
    d=int(dim*(dim+1)/2)
    product(field,Si,sig_field,dim,N,NN)
    eps_resh=field.reshape(NN,d)
    sig_resh=sig_field.reshape(NN,d)
    Srsig=sig_resh @ Sref
    Srsig=Srsig.reshape(sig_field.shape)
    field=field- Srsig
    energie=0.5*np.trace(np.dot(np.transpose(sig_resh),eps_resh))/NN
    return np.mean(sig_resh,axis=0),energie,field

# For Eyre-Milton scheme (energie is optional)
def update_eps_EM(Ci,Mi,Cref,field,eps_field,E_field,N,dim,NN):
    d=int(dim*(dim+1)/2)
    field=field.reshape(NN,d)
    Crf = field @ Cref
    Crf = Crf.reshape(E_field.shape)
    field=field.reshape(E_field.shape)
    product(field,Mi,Crf,dim,N,NN)
    eps_field=eps_field+2*field
    eps_resh=eps_field.reshape(NN,d)
    Creps= eps_resh @ Cref
    Creps=Creps.reshape(eps_field.shape)
    product(field,Ci,eps_field,dim,N,NN)
    sig_resh=field.reshape(NN,d)
    field=field - Creps
    energie=0.5*np.trace(np.dot(np.transpose(sig_resh),eps_resh))/NN
    return np.mean(sig_resh,axis=0),energie,field,eps_field

# Strain-based basic scheme (see Moulinec Suquet 1994)
def Strain_BS(E_field,N,L,Cref,Ci,prec,name,type):
    dim=len(N)
    N=np.array(N)
    L=np.array(L)
    NN=1
    for i in range(dim):
        NN*=N[i]
    d=int(dim*(dim+1)/2)
    iteration=0
    print("init")
    Gamma_fourier=np.zeros(tuple(N)+(d,d),dtype=np.complex128)
    initialize_Gamma(Cref,N,L,dim,NN,Gamma_fourier,type)
    field=np.zeros(tuple(N)+(d,))
    eps_field=np.zeros(tuple(N)+(d,))
    normE=norm2(E_field,d)
    print("init done")
    #file=open('./data/'+name,'a')
    crit=prec+1
    while crit>prec:
        field=-Fourier_convolution(field,Gamma_fourier,dim,N,NN)
        field=field.astype("float64")
        res=eps_field-E_field-field
        crit=norm2(res,d)/normE
        eps_field=E_field+field
        sig,energie,field=update_tau_Strain(Ci,Cref,field,eps_field,N,dim,NN)
        #file.write(str(iteration)+" "+str(2*energie)+"\n")
        iteration+=1
        print(iteration,2*energie,sig,crit)
    #file.flush()

# Stress-based basic scheme (see Monchiet Bonnet 2012)
def Stress_BS(E_field,N,L,Sref,Si,prec,name,type):
    dim=len(N)
    N=np.array(N)
    L=np.array(L)
    NN=1
    for i in range(dim):
        NN*=N[i]
    d=int(dim*(dim+1)/2)
    iteration=0
    Cref=np.linalg.inv(Sref)
    print("init")
    Delta_field=np.zeros(tuple(N)+(d,d),dtype=np.complex128)
    initialize_Delta(Cref,N,L,dim,NN,Delta_field,type)
    field=np.zeros(tuple(N)+(d,))
    sig_field=np.zeros(tuple(N)+(d,))
    E_resh=E_field.reshape(NN,d)
    CrE= E_resh @ Cref
    CrE=CrE.reshape(E_field.shape)
    normCE=norm2(CrE,d)
    print("init done")
    #file=open('./data/'+name,'a')
    crit=prec+1
    while crit>prec:
        field=-Fourier_convolution(field,Delta_field,dim,N,NN)
        field=field.astype("float64")
        res=sig_field-CrE-field
        crit=norm2(res,d)/normCE
        sig_field=CrE+field
        sig,energie,field=update_tau_Stress(Si,Sref,field,sig_field,N,dim,NN)
        #file.write(str(iteration)+" "+str(2*energie)+"\n")
        iteration+=1
        print(iteration,2*energie,sig,crit)
    #file.flush()

# Eyre-Milton scheme (see Michel Moulinec Suquet 2001)
def EM(E_field,N,L,Cref,Ci,Mi,prec,name,type):
    dim=len(N)
    N=np.array(N)
    L=np.array(L)
    NN=1
    for i in range(dim):
        NN*=N[i]
    d=int(dim*(dim+1)/2)
    iteration=0
    print("init")
    Gamma_field=np.zeros(tuple(N)+(d,d),dtype=np.complex128)
    initialize_Gamma(Cref,N,L,dim,NN,Gamma_field,type)
    eps_field=np.zeros(tuple(N)+(d,))
    field=np.zeros(tuple(N)+(d,))
    normE=norm2(E_field,d)
    print("init done")
    #file=open('./data/'+name,'a')
    crit=prec+1
    while crit>prec:
        field=-Fourier_convolution(field,Gamma_field,dim,N,NN)
        field=field.astype("float64")
        res=E_field+field-eps_field
        crit=norm2(res,d)/normE
        field=E_field+field-eps_field
        sig,energie,field,eps_field=update_eps_EM(Ci,Mi,Cref,field,eps_field,E_field,N,dim,NN)
        #file.write(str(iteration)+" "+str(2*energie)+"\n")
        iteration+=1
        print(iteration,2*energie,sig,crit)
    #file.flush()