import numpy as np
import os, os.path, glob
import nibabel as nib
from scipy.optimize import least_squares
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from multiprocessing import Pool


def fun_exp(coeffs, y, te):
    return y - model(te, coeffs)

def model(te, coeffs):
    return coeffs[0] * np.exp(-te * coeffs[1])

def fit_nls(data, S0, T2, TE):
    S0_final = np.zeros_like(S0)
    invT2_final = np.zeros_like(T2)
    R2_final = np.zeros_like(T2)
    scale = [np.max(S0), 50]
    for i in range(len(S0)):
        #if i % 30000 == 0:
        #    print(i)
        nls = least_squares(fun_exp, (S0[i], T2[i]), bounds=([0, 0.1], [np.inf,np.inf]), method='trf', x_scale=scale, max_nfev=20, args=(data[i, :], TE))
        S0_final[i] = nls.x[0]
        invT2_final[i] = nls.x[1]
        R2_final[i] = r2_score(data[i, :], model(TE, nls.x))

    return S0_final, invT2_final, R2_final

def _fit_nls_helper(args):
    x, TE = args
    return fit_nls(x[0], x[1], x[2], TE)

def fit_nls_by_multiprocessing(data, S0, T2, TE, workers):
    pool = Pool(processes=workers)
    result = pool.map(_fit_nls_helper, [(x,TE) for x in zip(np.array_split(data,workers),
                                                            np.array_split(S0,workers),
                                                            np.array_split(T2,workers))])
    pool.close()
    S0_final = np.concatenate([x[0] for x in result])
    invT2_final = np.concatenate([x[1] for x in result])
    R2_final = np.concatenate([x[2] for x in result])

    return S0_final, invT2_final, R2_final


# In[5]:


#Load data
def estimate_T2star(path,subj,num,num_workers=1):
    file_e1 = os.path.join(path,subj,num,subj+'_'+num+'_GRE_e1_reg_Warped.nii.gz')
    file_e2 = os.path.join(path,subj,num,subj+'_'+num+'_GRE_e2_reg_Warped.nii.gz')
    file_e3 = os.path.join(path,subj,num,subj+'_'+num+'_GRE_e3_reg_Warped.nii.gz')
    file_e4 = os.path.join(path,subj,num,subj+'_'+num+'_GRE_e4_reg_Warped.nii.gz')
    file_e5 = os.path.join(path,subj,num,subj+'_'+num+'_GRE_e5_reg_Warped.nii.gz')
    file_e6 = os.path.join(path,subj,num,subj+'_'+num+'_GRE_e6_reg_Warped.nii.gz')
    file_e7 = os.path.join(path,subj,num,subj+'_'+num+'_GRE_e7_reg_Warped.nii.gz')
    file_bm = os.path.join(path,subj,num,subj+'_'+num+'_MPRAGEPre_reg_monstr.nii.gz')

    data_raw_e1 = nib.load(file_e1)
    data_e1 = data_raw_e1.get_fdata()
    data_e2 = nib.load(file_e2).get_fdata()
    data_e3 = nib.load(file_e3).get_fdata()
    data_e4 = nib.load(file_e4).get_fdata()
    data_e5 = nib.load(file_e5).get_fdata()
    data_e6 = nib.load(file_e6).get_fdata()
    data_e7 = nib.load(file_e7).get_fdata()
    bm = nib.load(file_bm).get_fdata()


    #Stack echos
    data_conc = np.stack((data_e1,data_e2,data_e3,data_e4,data_e5,data_e6,data_e7),axis=3)
    n = data_conc.shape
    TE = np.array((0.006, 0.011, 0.016, 0.021, 0.026, 0.031, 0.036))

    data_conc_flat = data_conc.reshape(np.prod(n[0:3]), n[3])
    bm_flat = bm.reshape(np.prod(n[0:3]))

    data_flat_ss = data_conc_flat[bm_flat==1]
    #thr1 = 40
    #thr2 = 40

    #Ordinary least squares
    X = np.ones((n[3],2))
    X[:,0] = TE
    XTXinvXT = np.dot(np.linalg.inv(np.dot(X.T,X)),X.T)
    beta_hat = np.dot(XTXinvXT,np.log(data_flat_ss.T+1e-9))
    logS0_ss = beta_hat[1,:]#.reshape(n[0:3])
    negT2inv_ss = beta_hat[0,:]#.reshape(n[0:3])
    #negT2inv[negT2inv<1e-9] = 1e-9
    S0_ss = np.exp(logS0_ss)
    invT2_ss = -negT2inv_ss

    invT2_ss[invT2_ss<0.5] = 0.5#np.percentile(invT2_ss,5)
    invT2_ss[invT2_ss>np.percentile(invT2_ss,95)] = np.percentile(invT2_ss,95)

    #By row
    #S0_ss_final = np.zeros_like(S0_ss)
    #T2_ss_final = np.zeros_like(T2_ss)
    #for c in range(n[1]):
    #    print(c)

    #for i in range(len(S0_ss)):
    #    if i%1000==0:
    #        print(i)
    #    nls = least_squares(fun_exp,(S0_ss[i],T2_ss[i]),method='lm',max_nfev=20,args=(data_flat_ss[i,:],TE))
    #    S0_ss_final[i] = nls.x[0]
    #    T2_ss_final[i] = nls.x[1]
    #S0_ss_final, T2_ss_final = fit_nls(data_flat_ss[0:10,:], S0_ss[0:10], T2_ss[0:10], TE)
    #scale = np.max(S0_ss)
    if num_workers > 1:
        S0_ss_final, invT2_ss_final, R2_ss_final = fit_nls_by_multiprocessing(data_flat_ss, S0_ss, invT2_ss, TE, num_workers)
    else:
        S0_ss_final, invT2_ss_final, R2_ss_final = fit_nls(data_flat_ss, S0_ss, invT2_ss, TE)

    S0 = np.zeros(np.prod(n[0:3]))
    S0[bm_flat==1] = S0_ss_final
    S0 = S0.reshape(n[0:3])
    
    T2 = np.zeros(np.prod(n[0:3]))
    T2[bm_flat==1] = 1/invT2_ss_final
    T2 = T2.reshape(n[0:3])
    
    R2 = np.zeros(np.prod(n[0:3]))
    R2[bm_flat == 1] = R2_ss_final
    R2 = R2.reshape(n[0:3])

    file_S0 = os.path.join(path,subj,num,subj+'_'+num+'_GRE_S0.nii.gz')
    file_T2 = os.path.join(path,subj,num,subj+'_'+num+'_GRE_T2.nii.gz')
    file_R2 = os.path.join(path,subj,num,subj+'_'+num+'_GRE_R2.nii.gz')

    nib.Nifti1Image(S0, data_raw_e1.affine, data_raw_e1.header).to_filename(file_S0)
    nib.Nifti1Image(T2, data_raw_e1.affine, data_raw_e1.header).to_filename(file_T2)
    nib.Nifti1Image(R2, data_raw_e1.affine, data_raw_e1.header).to_filename(file_R2)

