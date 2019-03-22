import numpy as np
import tensorflow as tf
import os, sys, math, glob, random, cv2


# =============================================================
# =               Model Info Functions                        =
# =============================================================

def count_number_trainable_params():
    '''
    Counts the number of trainable variables.
    '''
    total_nb_params = 0
    total_btypes = 0
    for trainable_variable in tf.trainable_variables():
        shape = trainable_variable.get_shape() # e.g [D,F] or [W,H,C]
        current_nb_params = get_nb_params_shape(shape)
        variable_type = trainable_variable.dtype
        total_nb_params = total_nb_params + current_nb_params
        total_btypes = total_btypes + current_nb_params * params_bytes(variable_type)
    
    print "Model size: {0}K, Space usage: {1}KB ({2:6.2f}MB)".format(total_nb_params/1000, total_btypes/1000, total_btypes/1000000.0)
    # return total_nb_params, total_size
    return

def params_bytes(vtype):
    if vtype == 'float32_ref':
        return 32 / 8
    if vtype == 'float64_ref':
        return 64 / 8

def get_nb_params_shape(shape):
    '''
    Computes the total number of params for a given shap.
    Works for any number of shapes etc [D,F] or [W,H,C] computes D*F and W*H*C.
    '''
    nb_params = 1
    for dim in shape:
        nb_params = nb_params*int(dim)
    return nb_params


def check_img4d(sess, img4d):
    
    multichannel = True
    if img4d.shape[-1] == 1:
        multichannel = False
    
    label_patches, label_patches_indices = img4d_splitpatches(img4d)
    reconstruct_patches = np.zeros_like(label_patches)
    uint8_labelpatches = np.zeros_like(label_patches)
    for num in range(len(label_patches)):
        raw = label_patches[num].reshape([-1] + [k for k in label_patches.shape[1:]])
        mos, fake = sess.run([model.downscaled, model.gene_outputs], feed_dict={x: raw, is_training: False})
        reconstruct_patches[num] = img4duint8(fake)
        
        uint8_labelpatches[num] = img4duint8(raw)
    
    reconstruct_img4d = shaved_img4d_reconstruct(reconstruct_patches, label_patches_indices)
    original_img4d = shaved_img4d_reconstruct(uint8_labelpatches, label_patches_indices)
    
    recon_center = reconstruct_img4d[0,:,:,3,3,:]
    orig_center  = original_img4d[0,:,:,3,3,:]
    print 'center image: image[{0}]  psnr: {1}, mse: {2}, ssim: {3}'.format(i, psnr(recon_center, orig_center), 
                                                                            mse(recon_center, orig_center), 
                                                                            ssim(recon_center, orig_center,
                                                                                multichannel=multichannel))
    return reconstruct_img4d, original_img4d

def cal_metrics(predicts, labels):
    ''' Calculate the PSNR and SSIM '''
    
    # print 'calculate the psnr, ssim and mse:'
    multichannel = True
    if predicts.shape[-1] == 1:
        multichannel = False
    
    for i in range(len(predicts)):
        pred_ = predicts[i]
        pred_ = np.uint8(Normal(pred_)*255.)
#         if (pred_.shape[-1] != 1) and (len(pred_.shape) == 3):
#             pred_ = cv2.cvtColor(pred_, cv2.COLOR_BGR2RGB)
        
        labels_ = labels[i]
        labels_ = np.uint8(Normal(labels_)*255.)
#         if (pred_.shape[-1] != 1) and (len(pred_.shape) == 3):
#             labels_ = cv2.cvtColor(labels_, cv2.COLOR_BGR2RGB)

        if pred_.shape[-1] == 1:
            pred_ = np.squeeze(pred_)
            labels_ = np.squeeze(labels_)
        print 'image[{0}]  psnr: {1}, mse: {2}, ssim: {3}'.format(i, psnr(pred_, labels_), 
                                                                  mse(pred_, labels_), 
                                                                  ssim(pred_, labels_, multichannel=multichannel))
    return



#########################
## upsampling function ##
#########################
def Upsampling4d(data, size, method='linear'):
    ''' The quard-linear upsampling methods for light field upsampling
        
        input :       data: (ndarray) The 4-dimension light field data 
                    method: (string)  The string describe the method used for upsampling, 'linear', 'nearest'
        
        output:  upsampled: (ndarray) The upsampled light field data
        
        reference: [1] Weiser, Alan, and Sergio E. Zarantonello. ''A note on piecewise linear and multilinear
                   table interpolation in many dimensions.'' MATH. COMPUT. 50.181 (1988): 189-196.
                   http://www.ams.org/journals/mcom/1988-50-181/S0025-5718-1988-0917826-0/S0025-5718-1988-0917826-0.pdf
                   [2] http://nbviewer.jupyter.org/github/pierre-haessig/stodynprog/blob/master/stodynprog/linear_interp_benchmark.ipynb
                   [3] https://scicomp.stackexchange.com/questions/19137/what-is-the-preferred-and-efficient-approach-for-interpolating-multidimensional?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
    '''
    from scipy.interpolate import RegularGridInterpolator
    
    width, height, sview, tview = size[0], size[1], size[2], size[3]

    data = np.squeeze(data)
    w,h,m,n = data.shape   # m = n = 3

    x = np.linspace(0, w-1, w, dtype=np.float64)
    y = np.linspace(0, h-1, h, dtype=np.float64)
    s = np.linspace(0, m-1, m, dtype=np.float64)  # [0., 2., 4.]
    t = np.linspace(0, n-1, n, dtype=np.float64)  # [0., 2., 4.]

    fn = RegularGridInterpolator((x,y,s,t), data, method='linear')

    x_up = np.linspace(0, w-1, width, dtype=np.float64)
    y_up = np.linspace(0, h-1, height, dtype=np.float64)
    s_up = np.linspace(0, m-1, sview, dtype=np.float64)  # [ 0., 1., 2., 3., 4.]
    t_up = np.linspace(0, n-1, tview, dtype=np.float64)  # [ 0., 1., 2., 3., 4.]
    indices = np.array([[i,j,k,l] for i in x_up for j in y_up for k in s_up for l in t_up])

    upsampled = fn(indices)
    upsampled = upsampled.reshape(width, height, sview, tview)

    return upsampled


## downsampling by nearest neighbor and upsampling by quardlinear
def batch_upsampling(data, K=2, method='linear'):

    data_upsampled = np.zeros_like(data, dtype=np.float32)
    data = np.squeeze(data)
    if len(data.shape) == 4:
        data = np.expand_dims(data, axis=0)
    
    for i in range(len(data)):
        item = data[i]
        item_down = item[::K,::K,::K,::K]
        item_up = Upsampling4d(item_down, item.shape, method)
        data_upsampled[i] = np.expand_dims(item_up, axis=4)

    return data_upsampled


def lfpsnrs(truth4d, recons4d):
    ''' The truth4d represents for a single 4d-patches of light field
        
        truth4d  = [height, width, sview, tview, channels]
        recons4d = [height, width, sview, tview, channels]
    '''
    assert truth4d.shape == recons4d.shape, 'The prediction and label should be same size.'
    
    h,w,s,t,c = truth4d.shape
    lfpsnr = np.zeros([s,t])
    for i in range(s):
        for j in range(t):
            truth = truth4d[:,:,i,j,:]
            truth = np.squeeze(truth)
            recons = recons4d[:,:,i,j,:]
            recons = np.squeeze(recons)
            lfpsnr[i,j] = psnr(truth, recons)
    meanpsnr = np.mean(lfpsnr)
    return lfpsnr, meanpsnr