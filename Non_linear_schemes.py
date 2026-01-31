import numpy as np
from mgis import ThreadPool
import mgis.behaviour as mgis_bv
from Operators import *

integration_type = mgis_bv.IntegrationType.IntegrationWithConsistentTangentOperator

def init_state(behaviours,d):
    zero=np.zeros((d,))
    nph=len(behaviours)
    for r in range(nph):
        behaviours[r].s0.gradients[:] = zero
        behaviours[r].s0.thermodynamic_forces[:] = zero

def constitutive_update(Cref,field,eps_resh,micro_flat,behaviours,dim,NN,dt_):
    d=int(dim*(dim+1)/2)
    shape=field.shape
    field = field.reshape(NN,d)
    nph=len(behaviours)
    for r in range(nph):
        behaviours[r].s1.gradients[:]=eps_resh[micro_flat==r,:]
        p1 = ThreadPool(10)
        rs = mgis_bv.integrate(p1,behaviours[r], integration_type, dt_)
        assert rs > 0, "Behaviour integration has failed."
        field[micro_flat==r,:]= behaviours[r].s1.thermodynamic_forces[:]
    sig=np.mean(field,axis=0)
    field = field.reshape(shape)
    Creps= eps_resh @ Cref
    Creps=Creps.reshape(shape)
    field = field - Creps
    return sig,field

def dual_constitutive_update(Sref,field,sig_resh,micro_flat,behaviours,dim,NN,dt_):
    d=int(dim*(dim+1)/2)
    shape=field.shape
    field = field.reshape(NN,d)
    nph=len(behaviours)
    for r in range(nph):
        behaviours[r].s1.gradients[:]=sig_resh[micro_flat==r,:]
        p1 = ThreadPool(10)
        rs = mgis_bv.integrate(p1,behaviours[r], integration_type, dt_)
        assert rs > 0, "Behaviour integration has failed."
        field[micro_flat==r,:]= behaviours[r].s1.thermodynamic_forces[:]
    eps=np.mean(field,axis=0)
    field = field.reshape(shape)
    Srsig= sig_resh @ Sref
    Srsig= Srsig.reshape(shape)
    field = field - Srsig
    return eps,field

# The strain basic scheme is the initial scheme from Moulinec and Suquet.
# It is strain-piloted: we provide an history E_history which is a uniform remote strain which evolves in time.
def Non_linear_Strain_BS(E_history,dt,N,L,phases,behaviours,Cref,prec,name,type):
    nph=len(behaviours)
    dim=len(N)
    N=np.array(N)
    L=np.array(L)
    NN=1
    for i in range(dim):
        NN*=N[i]
    d=int(dim*(dim+1)/2)
    Nt=len(dt)
    print("initialize")
    Gamma_fourier=np.zeros(tuple(N)+(d,d),dtype=np.complex128)
    initialize_Gamma(Cref,N,L,dim,NN,Gamma_fourier,type)
    field=np.zeros(tuple(N)+(d,))
    eps_field=np.zeros(tuple(N)+(d,))
    eps_resh=eps_field.reshape(NN,d)
    init_state(behaviours,d)
    t1=0.
    micro_flat=phases.flatten()
    E_field=np.zeros(tuple(N)+(d,))
    print("initialized")
    file=open(name+'.txt','a')
    file.write("0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0."+"\n")
    for i_t in range(1,Nt+1):
        print("pas de temps ",i_t)
        E_field[:]=E_history[i_t-1]
        normE=norm2(E_field,d)
        iteration=0
        t1+=dt[i_t-1]
        crit=prec[i_t-1]+1
        while crit>prec[i_t-1]:
            sig_moy,field=constitutive_update(Cref,field,eps_resh,micro_flat,behaviours,dim,NN,dt[i_t-1])  
            field=-Fourier_convolution(field,Gamma_fourier,dim,N,NN)
            field=field.astype("float64")
            res=eps_field-E_field-field
            crit=norm2(res,d)/normE
            eps_field=E_field+field
            eps_resh=eps_field.reshape(NN,d)
            iteration+=1
            print("t ",t1," it ",iteration," conv ",crit," sig ",sig_moy)
        eps_moy=np.mean(eps_resh,axis=0)
        file.write(str(t1))
        for jj in range(6):
            file.write(" "+str(eps_moy[jj]))
        for jj in range(6):
            file.write(" "+str(sig_moy[jj]))
        file.write("\n")
        for r in range(nph):
            behaviours[r].update()
        file.flush()


# The stress basic scheme (Monchiet Bonnet) consists in working on stress field, with the stress Green operator 'Delta'.
# It also means that the behaviour must takes the increment of stress as an argument and return the strain.
# Here, as the strain basic scheme above, it is strain-piloted.
def Non_linear_Stress_BS(E_history,dt,N,L,phases,behaviours,Sref,prec,name,type):
    Cref=np.linalg.inv(Sref)
    nph=len(behaviours)
    dim=len(N)
    N=np.array(N)
    L=np.array(L)
    NN=1
    for i in range(dim):
        NN*=N[i]
    d=int(dim*(dim+1)/2)
    Nt=len(dt)
    print("initialize")
    Delta_fourier=np.zeros(tuple(N)+(d,d),dtype=np.complex128)
    initialize_Delta(Cref,N,L,dim,NN,Delta_fourier,type)
    field=np.zeros(tuple(N)+(d,))
    sig_field=np.zeros(tuple(N)+(d,))
    sig_resh=np.reshape(sig_field,(NN,d))
    init_state(behaviours,d)
    t1=0.
    micro_flat=phases.flatten()
    E_field=np.zeros(tuple(N)+(d,))
    print("initialized")
    file=open(name+'.txt','a')
    file.write("0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0."+"\n")
    for i_t in range(1,Nt+1):
        print("pas de temps ",i_t)
        E_field[:]=E_history[i_t-1]
        E_resh=E_field.reshape(NN,d)
        CrE= E_resh @ Cref
        CrE=CrE.reshape(E_field.shape)
        normCE=norm2(CrE,d)
        iteration=0
        t1+=dt[i_t-1]
        crit=prec[i_t-1]+1
        while crit>prec[i_t-1]:
            eps_moy,field=dual_constitutive_update(Sref,field,sig_resh,micro_flat,behaviours,dim,NN,dt[i_t-1]) 
            field=-Fourier_convolution(field,Delta_fourier,dim,N,NN)
            field=field.astype("float64")
            res=sig_field-CrE-field
            crit=norm2(res,d)/normCE
            sig_field=CrE+field
            sig_resh=np.reshape(sig_field,(NN,d))
            sig_moy=np.mean(sig_resh,axis=0)
            iteration+=1
            print("t ",t1," it ",iteration," conv ",crit," sig ",sig_moy)
        file.write(str(t1))
        for jj in range(6):
            file.write(" "+str(eps_moy[jj]))
        for jj in range(6):
            file.write(" "+str(sig_moy[jj]))
        file.write("\n")
        for r in range(nph):
            behaviours[r].update()
        file.flush()
