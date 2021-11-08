import numpy as np
import os, os.path, glob
import nibabel as nib
from scipy.optimize import least_squares
from sklearn.metrics import r2_score
from multiprocessing import Pool


# Define fun_exp and model for later functions

#    parser.add_argument('--reorient', choices=[None, 'RAS', 'LAS', 'RPS', 'LPS', 'RAI', 'LAI', 'RPI', 'LPI'],
#                        default='RAS', help='Reorient T2 images to specified orientation before registering')
def _fun_exp(coeffs, y, te):
    return y - _model(te, coeffs)


def _model(te, coeffs):
    return coeffs[0] * np.exp(-te * coeffs[1])


def fit_nls(data, s0, t2, te):
    '''
    fit_nls - Estimates s0 and t2star using trust region
    :param data: flattened signals for given te values
    :param s0: initial estimates for signal for te=0
    :param t2: initial estimates for t2star
    :param te: te values
    :return:
    '''
    s0_final = np.zeros_like(s0)
    inv_t2_final = np.zeros_like(t2)
    r2_final = np.zeros_like(t2)
    scale = [np.max(s0), 50]
    for i in range(len(s0)):
        print(i)
        nls = least_squares(_fun_exp,
                            (s0[i], t2[i]),
                            bounds=([0, 0.1], [np.inf, np.inf]),
                            method='trf',
                            x_scale=scale,
                            max_nfev=20,
                            #loss='soft_l1',
                            args=(data[i, :], te))
        s0_final[i] = nls.x[0]
        inv_t2_final[i] = nls.x[1]
        r2_final[i] = r2_score(data[i, :], _model(te, nls.x))

    return s0_final, inv_t2_final, r2_final


def _fit_nls_helper(args):
    x, te = args
    return fit_nls(x[0], x[1], x[2], te)


def fit_nls_by_multiprocessing(data, s0, t2, te, workers):
    '''
    fit_nls_by_multiprocessing - Estimates s0 and t2star with multiprocessing
    :param data: flattened signals for given te values
    :param s0: initial estimates for signal for te=0
    :param t2: initial estimates for t2star
    :param te: te values
    :param workers: number of thread for multiprocessing
    :return:
    '''
    pool = Pool(processes=workers)
    result = pool.map(_fit_nls_helper, [(x, te) for x in zip(np.array_split(data, workers),
                                                             np.array_split(s0, workers),
                                                             np.array_split(t2, workers))])
    pool.close()
    s0_final = np.concatenate([x[0] for x in result])
    inv_t2_final = np.concatenate([x[1] for x in result])
    r2_final = np.concatenate([x[2] for x in result])

    return s0_final, inv_t2_final, r2_final


def estimate_t2star(t2star_files, brainmask_file, te_list, output_dir=None, output_prefix='EstT2star_', num_workers=1):
    '''
    estimate_t2star - Load multiecho t2star files and estimate t2star
    :param t2star_files: List of paths to t2star files
    :type t2star_files: list[str]
    :param brainmask_file: Path to brainmask file in same space as the t2star files
    :type brainmask_file: str
    :param te: List of te
    :type te: list[float]
    :param output_dir: Output directory. Default is current working directory.
    :type output_dir: str
    :param output_prefix: Output prefix. Default is 'EstT2star_'
    :type output_prefix: str
    :param num_workers: Number of workers. If >1, uses multiprocessing. Default is 1.
    :type num_workers: int
    :return:
    '''
    num_echos = len(t2star_files)
    if len(te_list) is not num_echos:
        raise ValueError('Length of TE and number of t2star files should be equal.')

    # Load first echo to get image size and header info
    data_raw_e1 = nib.load(t2star_files[0])
    data_e1 = data_raw_e1.get_fdata()

    # Create empty array to store all echos
    data = np.zeros(data_e1.shape[0:3] + (num_echos,))
    print(data.shape)
    data_shape = data.shape
    data[:, :, :, 0] = np.squeeze(data_e1)

    # Load remaining echos
    for idx, f in enumerate(t2star_files[1:]):
        data[:, :, :, idx+1] = np.squeeze(nib.load(f).get_fdata()) # Offset by 1 since we start at element 1

    bm = np.squeeze(nib.load(brainmask_file).get_fdata())

    #Stack echos
    #data_conc = np.stack((data_e1,data_e2,data_e3,data_e4,data_e5,data_e6,data_e7),axis=3)
    #n = data_conc.shape
    #TE = np.array((0.006, 0.011, 0.016, 0.021, 0.026, 0.031, 0.036))
    te = np.array(te_list)
    data_flat = data.reshape(np.prod(data_shape[0:3]), data_shape[3])
    bm_flat = bm.reshape(np.prod(data_shape[0:3]))

    data_flat_mask = data_flat[bm_flat == 1]

    # Initial s0 and t2star values from ordinary least squares
    x = np.ones((data_shape[3], 2))
    x[:, 0] = te
    xtrans_xinv_xtrans = np.dot(np.linalg.inv(np.dot(x.T, x)), x.T)
    beta_hat = np.dot(xtrans_xinv_xtrans, np.log(data_flat_mask.T + 1e-9))
    log_s0_mask = beta_hat[1, :]#.reshape(n[0:3])
    neg_t2inv_mask = beta_hat[0, :]#.reshape(n[0:3])
    #negT2inv[negT2inv<1e-9] = 1e-9
    s0_mask = np.exp(log_s0_mask)
    t2inv_mask = -neg_t2inv_mask

    t2inv_mask[t2inv_mask < 0.5] = 0.5 # np.percentile(invT2_ss,5)
    t2inv_mask[t2inv_mask > np.percentile(t2inv_mask, 95)] = np.percentile(t2inv_mask, 95)

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
        s0_mask, t2inv_mask, r2_mask = fit_nls_by_multiprocessing(data_flat_mask, s0_mask, t2inv_mask, te, num_workers)
    else:
        s0_mask, t2inv_mask, r2_mask = fit_nls(data_flat_mask, s0_mask, t2inv_mask, te)

    s0 = np.zeros(np.prod(data_shape[0:3]))
    s0[bm_flat == 1] = s0_mask
    s0 = s0.reshape(data_shape[0:3])
    
    t2 = np.zeros(np.prod(data_shape[0:3]))
    t2[bm_flat == 1] = 1/t2inv_mask
    t2 = t2.reshape(data_shape[0:3])
    
    r2 = np.zeros(np.prod(data_shape[0:3]))
    r2[bm_flat == 1] = r2_mask
    r2 = r2.reshape(data_shape[0:3])
    
    if output_dir is None:
        output_dir = ''

    file_s0 = os.path.join(output_dir, output_prefix + '_S0.nii.gz')
    file_t2 = os.path.join(output_dir, output_prefix + '_T2star.nii.gz')
    file_r2 = os.path.join(output_dir, output_prefix + '_R2.nii.gz')
    print(file_s0)
    print(file_t2)
    print(file_r2)

    nib.Nifti1Image(s0, data_raw_e1.affine, data_raw_e1.header).to_filename(file_s0)
    nib.Nifti1Image(t2, data_raw_e1.affine, data_raw_e1.header).to_filename(file_t2)
    nib.Nifti1Image(r2, data_raw_e1.affine, data_raw_e1.header).to_filename(file_r2)

