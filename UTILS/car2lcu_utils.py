import os
import sys
import math
import warnings
import numpy as np
import scipy.linalg as la
import scipy.optimize as opt
from scipy.linalg import null_space

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

DECIMALS = 6

def TBT_to_L1opt_LCU (tbt, norb, solmtd='l1ip', pout=True): 
    solvers = {'l1rs'  :'revised simplex', 
               'l1ip'  :'interior-point',
               'l1hi'  :'highs',
               'l1hds' :'highs-ds',
               'l1hpm' :'highs-ipm',
               'l1ip'  :'interior-point'}

    if (solmtd == 'l1ip' or 'l1rs' or 'l1hi' or 'l1hds' or 'l1hpm'): 
        resout = L1opt_LCU (tbt, norb, solvers[solmtd], pout)
        return resout
    else: 
        return print ('Unknown solmtd in solv_rcsa.') 


def OBTTBT_to_L1opt_LCU (obt, tbt, norb, solmtd='l1ip', pout=True): 
    solvers = {'l1rs'  :'revised simplex', 
               'l1ip'  :'interior-point',
               'l1hi'  :'highs',
               'l1hds' :'highs-ds',
               'l1hpm' :'highs-ipm',
               'l1ip'  :'interior-point'}

    if (solmtd == 'l1ip' or 'l1rs' or 'l1hi' or 'l1hds' or 'l1hpm'): 
        resout = obttbt_L1opt_LCU (obt, tbt, norb, solvers[solmtd], pout)
        return resout
    else: 
        return print ('Unknown solmtd in solv_rcsa.') 


def L1opt_LCU (tbt, norb, optmtd='interior-point', pout=True):
#FUNCTION: Construct the constrains of L1-norm minimization
    cvar = coeff_builder(norb)
    ccon = tbt2ccon(tbt, norb)
    pcon = pcon_builder(norb)
    res = opt.linprog(cvar, A_ub=pcon, b_ub=ccon, bounds=(None,None), method=optmtd)
    opteval = pol_eval (res, tbt, norb, optmtd, pout)
    return opteval


def obttbt_L1opt_LCU (obt, tbt, norb, optmtd='interior-point', pout=True):
#FUNCTION: Construct the constrains of L1-norm minimization
    cvar = coeff_builder(norb)
    ccon = obttbt2ccon(obt, tbt, norb)
    pcon = pcon_builder(norb)
    res = opt.linprog(cvar, A_ub=pcon, b_ub=ccon, bounds=(None,None), method=optmtd)
    opteval = obttbt_pol_eval (res, obt, tbt, norb, optmtd, pout)
    return opteval

def coeff_builder (norb):
    nproj, npair, nfree = cpar_nproj(norb)
    return np.block([np.zeros(nfree), np.ones(nproj)]).reshape(nfree+nproj,1)

def nCk (n,k):
    f = math.factorial 
    return int(f(n) / f(k) / f(n-k))

def cpar_nproj (norb): 
#FUNCTION: number of projectors, pairs and free parameters
    if (norb == 2): 
        npair = nCk(norb,2)
        ntrpl = 0
        nquad = 0
    elif (norb == 3): 
        npair = nCk(norb, 2)
        ntrpl = nCk(norb, 3)
        nquad = 0
    elif (norb >= 4):
        npair = nCk(norb,2)
        ntrpl = nCk(norb,3)
        nquad = nCk(norb,4)
    nproj = norb + (5 * npair) + (16 * ntrpl) + (12 * nquad)
    nfree = nproj - (npair + norb)
    return nproj, npair, nfree

def tbt2ccon (tbt, norb):
    cpar = tbt2cpar (tbt, norb)
    return np.block([[-1*cpar],[cpar]])

def obttbt2ccon (obt, tbt, norb):
    cpar = obttbt2cpar (obt, tbt, norb)
    return np.block([[-1*cpar],[cpar]])

def tbt2cpar (tbt, norb):
#FUNCTION: Transfer tbt to cpar, a particular solution of lin-dep equations
    num = int(0.5 *norb *(norb+1) )
    nproj, npair, nfree = cpar_nproj(norb)
    cpar = np.zeros([nproj,1])
    lam = tbt2lam (tbt, norb)
    cpar[0:num,:] = -0.5 * lam
    return cpar

def obttbt2cpar (obt, tbt, norb):
#FUNCTION: Transfer tbt to cpar, a particular solution of lin-dep equations
    num = int(0.5 *norb *(norb+1) )
    nproj, npair, nfree = cpar_nproj(norb)
    cpar = np.zeros([nproj,1])
    lam = obttbt2lam (obt, tbt, norb)
    cpar[0:num,:] = -0.5 * lam
    return cpar

def tbt2lam (tbt, norb):
#FUNCTION: Transfer tbt to the vector form (useful in debug)
    num = int(0.5 *norb *(norb+1) )
    lam = np.zeros([num,1])
    for iorb in range(norb):
        #lam[iorb] = -0.5 * tbt[iorb, iorb, iorb, iorb]
        lam[iorb] = tbt[iorb, iorb, iorb, iorb]
    ijorb=0
    for iorb in range(norb):
        for jorb in range(iorb+1, norb):
            #lam[norb+ijorb] = -0.5 * tbt[iorb, iorb, jorb, jorb]
            lam[norb+ijorb] = tbt[iorb, iorb, jorb, jorb]
            ijorb += 1
    return lam 

def obttbt2lam (obt, tbt, norb):
#FUNCTION: Transfer tbt to the vector form (useful in debug)
    num = int(0.5 *norb *(norb+1) )
    lam = np.zeros([num,1])
    for iorb in range(norb):
        #lam[iorb] = -0.5 * tbt[iorb, iorb, iorb, iorb]
        lam[iorb] = tbt[iorb, iorb, iorb, iorb] + obt[iorb, iorb]
    ijorb=0
    for iorb in range(norb):
        for jorb in range(iorb+1, norb):
            #lam[norb+ijorb] = -0.5 * tbt[iorb, iorb, jorb, jorb]
            lam[norb+ijorb] = tbt[iorb, iorb, jorb, jorb]
            ijorb += 1
    return lam 

def pcon_builder (norb=2):
    kerp  = null_space(pmat_construction(norb))
    dfree = kerp.shape[0]
    return np.block([[kerp, -1.0*np.eye(dfree)],[-1*kerp, -1.0*np.eye(dfree)]])

def pmat_construction (norb):
#FUNCTION: Construct the projector for n-orbital tbt+obt to reflection polynomials
    if (norb == 1): 
        print('norb=1, Use pen and paper, not a quantum computer!')
        return

    nproj, npair, nfree = cpar_nproj(norb)
    pmat = np.zeros([npair+norb, nproj])
    # Keep the cartan-polynoms at the beginning for easy construction of cpar

    iprj = 0
    for iorb in range(1, norb+1):
        #orb index
        indx = iorb-1

        # P(1,1) : ni
        pmat[indx,iprj] =  1.0 ; iprj+=1

    for iorb in range(1, norb+1):
        for jorb in range(iorb+1, norb+1):
            #pair index
            ijndx = norb + npair - int(0.5 * (norb-iorb+1) * (norb-iorb)) + (jorb - iorb - 1)

            # P(2,1) : nij
            pmat[ijndx,iprj] =  1.0 ; iprj+=1

    for iorb in range(1, norb+1):
        for jorb in range(iorb+1, norb+1):
            #pair index
            ijndx = norb + npair - int(0.5 * (norb-iorb+1) * (norb-iorb)) + (jorb - iorb - 1)

            #orb index
            indx = iorb-1
            jndx = jorb-1

            # P(2,2) : ni-nij
            pmat[indx,iprj]  =  1.0
            pmat[ijndx,iprj] = -1.0 ; iprj+=1
            # P(2,2) : nj-nij 
            pmat[jndx,iprj]  =  1.0
            pmat[ijndx,iprj] = -1.0 ; iprj+=1
            
            # P(2,3) : ni + nj - 2 * nij
            pmat[indx,iprj]  =  1.0
            pmat[jndx,iprj]  =  1.0  
            pmat[ijndx,iprj] = -2.0 ; iprj+=1

            # P(2,4) : ni + nj - nij
            pmat[indx,iprj]  =  1.0
            pmat[jndx,iprj]  =  1.0  
            pmat[ijndx,iprj] = -1.0 ; iprj+=1

            for korb in range(jorb+1, norb+1):
                #pair index
                ikndx = norb + npair - int(0.5 * (norb-iorb+1) * (norb-iorb)) + (korb - iorb - 1)
                jkndx = norb + npair - int(0.5 * (norb-jorb+1) * (norb-jorb)) + (korb - jorb - 1)
  
                #orb index
                kndx = korb-1

                # P(3,1) : ni + njk - nik
                # 3-orbitals asymmetric 2-pairs projectors
                pmat[indx,iprj]  =  1.0
                pmat[jkndx,iprj] =  1.0
                pmat[ikndx,iprj] = -1.0 ; iprj+=1
 
                pmat[indx,iprj] =   1.0
                pmat[jkndx,iprj] =  1.0
                pmat[ijndx,iprj] = -1.0 ; iprj+=1
  
                pmat[jndx,iprj]  =  1.0
                pmat[ikndx,iprj] =  1.0
                pmat[ijndx,iprj] = -1.0 ; iprj+=1
  
                pmat[jndx,iprj]  =  1.0
                pmat[ikndx,iprj] =  1.0
                pmat[jkndx,iprj] = -1.0 ; iprj+=1
  
                pmat[kndx,iprj]  =  1.0
                pmat[ijndx,iprj] =  1.0
                pmat[ikndx,iprj] = -1.0 ; iprj+=1
  
                pmat[kndx,iprj]  =  1.0
                pmat[ijndx,iprj] =  1.0
                pmat[jkndx,iprj] = -1.0 ; iprj+=1

                # P(3,3) : ni + njk - nik
                # 3-orbitals symmetric 2-pairs projectors
                pmat[indx,iprj]  =  1.0
                pmat[jndx,iprj]  =  1.0
                pmat[ijndx,iprj] = -1.0
                pmat[ikndx,iprj] = -1.0 ; iprj+=1

                pmat[indx,iprj]  =  1.0
                pmat[kndx,iprj]  =  1.0
                pmat[ijndx,iprj] = -1.0
                pmat[ikndx,iprj] = -1.0 ; iprj+=1
 
                pmat[jndx,iprj]  =  1.0
                pmat[indx,iprj]  =  1.0
                pmat[ijndx,iprj] = -1.0
                pmat[jkndx,iprj] = -1.0 ; iprj+=1

                pmat[jndx,iprj]  =  1.0
                pmat[kndx,iprj]  =  1.0
                pmat[ijndx,iprj] = -1.0
                pmat[jkndx,iprj] = -1.0 ; iprj+=1

                pmat[kndx,iprj]  =  1.0
                pmat[indx,iprj]  =  1.0
                pmat[ikndx,iprj] = -1.0
                pmat[jkndx,iprj] = -1.0 ; iprj+=1

                pmat[kndx,iprj]  =  1.0
                pmat[jndx,iprj]  =  1.0
                pmat[ikndx,iprj] = -1.0
                pmat[jkndx,iprj] = -1.0 ; iprj+=1
 
                # P(3,2) : ni + njk - nik - nij
                # 3-orbitals asymmetric-3-pairs projectors
                pmat[indx,iprj]  =  1.0
                pmat[jkndx,iprj] =  1.0
                pmat[ijndx,iprj] = -1.0
                pmat[ikndx,iprj] = -1.0 ; iprj+=1
 
                pmat[jndx,iprj]  =  1.0
                pmat[ikndx,iprj] =  1.0
                pmat[ijndx,iprj] = -1.0
                pmat[jkndx,iprj] = -1.0 ; iprj+=1
  
                pmat[kndx,iprj]  =  1.0
                pmat[ijndx,iprj] =  1.0
                pmat[ikndx,iprj] = -1.0
                pmat[jkndx,iprj] = -1.0 ; iprj+=1
 
                # 3-orbitals symmetric-3-pairs projectors

                # 20)  
                pmat[jndx,iprj]  =  1.0
                pmat[kndx,iprj]  =  1.0
                pmat[ijndx,iprj] = -1.0
                pmat[ikndx,iprj] = -1.0
                pmat[jkndx,iprj] = -1.0 ; iprj+=1



                for lorb in range(korb+1, norb+1):
                    ilndx = norb + npair - int(0.5 * (norb-iorb+1) * (norb-iorb)) + (lorb - iorb - 1)
                    jlndx = norb + npair - int(0.5 * (norb-jorb+1) * (norb-jorb)) + (lorb - jorb - 1)
                    klndx = norb + npair - int(0.5 * (norb-korb+1) * (norb-korb)) + (lorb - korb - 1)

                    lndx = korb-1
                    #4-orbitals non-symmetric-4-pairs projectors
                    # print("lorb=", lorb, "iprj=", iprj)

                    pmat[kndx,iprj]  =  1.0
                    pmat[lndx,iprj]  =  1.0
                    pmat[ijndx,iprj] =  1.0
                    pmat[ikndx,iprj] = -1.0
                    pmat[jlndx,iprj] = -1.0
                    pmat[klndx,iprj] = -1.0 ; iprj+=1 #1

                    pmat[jndx,iprj]  =  1.0
                    pmat[lndx,iprj]  =  1.0
                    pmat[ikndx,iprj] =  1.0
                    pmat[ijndx,iprj] = -1.0
                    pmat[jlndx,iprj] = -1.0
                    pmat[klndx,iprj] = -1.0 ; iprj+=1 #2

                    pmat[indx,iprj]  =  1.0
                    pmat[kndx,iprj]  =  1.0
                    pmat[jlndx,iprj] =  1.0
                    pmat[ijndx,iprj] = -1.0
                    pmat[ikndx,iprj] = -1.0
                    pmat[klndx,iprj] = -1.0 ; iprj+=1 #3

                    pmat[indx,iprj]  =  1.0
                    pmat[jndx,iprj]  =  1.0
                    pmat[klndx,iprj] =  1.0
                    pmat[ijndx,iprj] = -1.0
                    pmat[ikndx,iprj] = -1.0
                    pmat[jlndx,iprj] = -1.0 ; iprj+=1 #4

                    pmat[kndx,iprj]  =  1.0
                    pmat[lndx,iprj]  =  1.0
                    pmat[ijndx,iprj] =  1.0
                    pmat[ilndx,iprj] = -1.0
                    pmat[jkndx,iprj] = -1.0
                    pmat[klndx,iprj] = -1.0 ; iprj+=1 #5

                    pmat[jndx,iprj]  =  1.0
                    pmat[kndx,iprj]  =  1.0
                    pmat[ilndx,iprj] =  1.0
                    pmat[ijndx,iprj] = -1.0
                    pmat[jkndx,iprj] = -1.0
                    pmat[klndx,iprj] = -1.0 ; iprj+=1 #6

                    pmat[indx,iprj]  =  1.0
                    pmat[lndx,iprj]  =  1.0
                    pmat[jkndx,iprj] =  1.0
                    pmat[ijndx,iprj] = -1.0
                    pmat[ilndx,iprj] = -1.0
                    pmat[klndx,iprj] = -1.0 ; iprj+=1 #7

                    pmat[indx,iprj]  =  1.0
                    pmat[jndx,iprj]  =  1.0
                    pmat[klndx,iprj] =  1.0
                    pmat[ijndx,iprj] = -1.0
                    pmat[ilndx,iprj] = -1.0
                    pmat[jkndx,iprj] = -1.0 ; iprj+=1 #8

                    pmat[jndx,iprj]  =  1.0
                    pmat[lndx,iprj]  =  1.0
                    pmat[ikndx,iprj] =  1.0
                    pmat[ilndx,iprj] = -1.0
                    pmat[jkndx,iprj] = -1.0
                    pmat[jlndx,iprj] = -1.0 ; iprj+=1 #9

                    pmat[jndx,iprj]  =  1.0
                    pmat[kndx,iprj]  =  1.0
                    pmat[ilndx,iprj] =  1.0
                    pmat[ikndx,iprj] = -1.0
                    pmat[jkndx,iprj] = -1.0
                    pmat[jlndx,iprj] = -1.0 ; iprj+=1 #10

                    pmat[indx,iprj]  =  1.0
                    pmat[lndx,iprj]  =  1.0
                    pmat[jkndx,iprj] =  1.0
                    pmat[ikndx,iprj] = -1.0
                    pmat[ilndx,iprj] = -1.0
                    pmat[jlndx,iprj] = -1.0 ; iprj+=1 #11

                    pmat[indx,iprj]  =  1.0
                    pmat[kndx,iprj]  =  1.0
                    pmat[jlndx,iprj] =  1.0
                    pmat[ikndx,iprj] = -1.0
                    pmat[ilndx,iprj] = -1.0
                    pmat[jkndx,iprj] = -1.0 ; iprj+=1 #12
    #factor of -2.0 is included to build non-trivial part of reflections 
    #return -2.0 * pmat
    return -2.0 * pmat

def pol_eval (res, tbt, norb, optmtd, pout=True):
    nproj, npair, nfree = cpar_nproj(norb)
    pmat = pmat_construction(norb)
    copt = res.x[:nfree].reshape(nfree,1)
    pker = null_space(pmat)
    cpar = tbt2cpar(tbt, norb)
    cobj = cpar + np.matmul(pker, copt)
    prdim = np.count_nonzero(np.around(cobj, decimals=DECIMALS))
    lam   = tbt2lam (tbt, norb)
    lam_resmax = np.around(np.max(np.absolute(lam - np.matmul(pmat, cobj) )), decimals=DECIMALS)
    csa_l1 = np.sum(np.absolute(lam), axis=0)[0]

    opteval = {'solver':optmtd, 'lamdim':npair, 'poldim':prdim, 
    'csa_l1': csa_l1, 'lcu_l1': res.fun, 'maxdlam':lam_resmax, 'cobj': cobj, 'copt':copt}   

    if (pout):
        print ('solver:', optmtd, 'csa_dim:', (norb+npair) , 'lcu_dim:', prdim, 
        'csa_l1:', csa_l1, 'lcu_l1:', res.fun, 'MAD:', lam_resmax) 

    return opteval


def obttbt_pol_eval (res, obt, tbt, norb, optmtd, pout=True):
    nproj, npair, nfree = cpar_nproj(norb)
    pmat = pmat_construction(norb)
    copt = res.x[:nfree].reshape(nfree,1)
    pker = null_space(pmat)
    cpar = obttbt2cpar(obt, tbt, norb)
    cobj = cpar + np.matmul(pker, copt)
    prdim = np.count_nonzero(np.around(cobj, decimals=DECIMALS))
    lam   = obttbt2lam (obt, tbt, norb)
    lam_resmax = np.around(np.max(np.absolute(lam - np.matmul(pmat, cobj) )), decimals=DECIMALS)
    csa_l1 = np.sum(np.absolute(lam), axis=0)[0]

    opteval = {'solver':optmtd, 'lamdim':npair, 'poldim':prdim, 
    'csa_l1': csa_l1, 'lcu_l1': res.fun, 'maxdlam':lam_resmax, 'cobj': cobj, 'copt':copt}   

    if (pout):
        print ('solver:', optmtd, 'csa_dim:', (norb+npair) , 'lcu_dim:', prdim, 
        'csa_l1:', csa_l1, 'lcu_l1:', res.fun, 'MAD:', lam_resmax) 

    return opteval

