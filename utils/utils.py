import numpy as np
import scipy
import scipy.ndimage

from scipy.misc import imread, imresize, imsave
import glob, math, random
import os, re, sys
import tensorflow as tf
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d
from scipy import signal
from scipy import ndimage
# from skimage.measure import compare_ssim as ssim
# from skimage.measure import compare_psnr as psnr
# from skimage.measure import compare_mse as mse

##############################################################
######                previous function                 ######
##############################################################
def gaussian2(size, sigma):
    """Returns a normalized circularly symmetric 2D gauss kernel array
    
    f(x,y) = A.e^{-(x^2/2*sigma^2 + y^2/2*sigma^2)} where
    
    A = 1/(2*pi*sigma^2)
    
    as define by Wolfram Mathworld 
    http://mathworld.wolfram.com/GaussianFunction.html
    """
    A = 1/(2.0*np.pi*sigma**2)
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = A*np.exp(-((x**2/(2.0*sigma**2))+(y**2/(2.0*sigma**2))))
    return g

def fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()

def get_center(self, x, channels):
    b,h,w,s,t,c = x.get_shape().as_list()
    print x.get_shape().as_list(), 'get the center:', s/2, t/2
    center = x[:,:,:,s/2,t/2,:]
    center = tf.expand_dims(center, axis=3)
    center = tf.expand_dims(center, axis=4)
    return center

def extend_center(self, x):
    x = tf.tile(x, [1,1,1,3,3,1])
    return x
##############################################################
######          statistic the info of network           ######
##############################################################
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
    
    return "Model size: {0}K, Space usage: {1}KB ({2:6.2f}MB)".format(total_nb_params/1000, total_btypes/1000, total_btypes/1000000.0)

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


##############################################################
######               downsampling methods               ######
##############################################################
def get_gauss_filter(shape=(7,7),sigma=1.2):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def blur(hrlf, psf):
    '''
        HR_LF = [batchsize, height, width, sview, tview, channel]
        
        
        return blurred_lfimgs
    '''
    blurred_lfimgs = np.zeros_like(hrlf)
    ws = psf.shape[0]
    t = (ws-1) / 2
    
    hrlf = np.concatenate([hrlf[:,:t,:], hrlf, hrlf[:,-t:,:]], axis=1)
    hrlf = np.concatenate([hrlf[:t,:,:], hrlf, hrlf[-t:,:,:]], axis=0)
    
    if hrlf.shape[2] == 3:
        blurred_lfimgs[:,:,0] = convolve2d(hrlf[:,:,0], psf, 'valid')
        blurred_lfimgs[:,:,1] = convolve2d(hrlf[:,:,1], psf, 'valid')
        blurred_lfimgs[:,:,2] = convolve2d(hrlf[:,:,2], psf, 'valid')
    else:
        blurred_lfimgs = convolve2d(np.squeeze(hrlf), psf, 'valid')
        blurred_lfimgs = np.expand_dims(blurred_lfimgs, axis=2)
    
    return blurred_lfimgs


def downsampling(HR_LF, K1=2, K2=2, nSig=1.2, spatial_only=False):
    '''@change: delete noise
    HR_LF = [batchsize, height, width, sview, tview, channels]

    
    '''
    
    b,h,w,s,t,c = HR_LF.shape
    psf = get_gauss_filter(shape=(7,7), sigma=nSig)
    LR_LF = np.zeros_like(HR_LF)
    for n in range(b):
        for i in range(s):
            for j in range(t):
                LR_LF[n,:,:,i,j,:] = blur(HR_LF[n,:,:,i,j,:], psf)
    
    LR_LF = LR_LF[:,::K1,::K1,:,:,:]
    if not spatial_only:
        LR_LF = LR_LF[:,:,:,::K2,::K2,:]
    return LR_LF

def downsampling2D(HR, K1=2, nSig=1.2, spatial_only=False):
    '''@change: delete noise
    HR_LF = [height, width, channels]

    
    '''
    
    h,w,c = HR.shape
    psf = get_gauss_filter(shape=(7,7), sigma=nSig)
    LR = np.zeros_like(HR)
    LR = blur(HR, psf)
    
    LR = LR[::K1,::K1,:]
    return LR

##############################################################
######                  measure method                  ######
##############################################################

def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def lfpsnrs(truth4d, recons4d):
    ''' The truth4d represents for a single 4d-patches of light field
        
        truth4d  = [height, width, sview, tview, channels]
        recons4d = [height, width, sview, tview, channels]
    '''
    assert truth4d.shape == recons4d.shape, 'The prediction and label should be same size.'
    assert truth4d.dtype == 'uint8', 'The ground truth should be uint8 format within range (0, 255)'
    assert recons4d.dtype == 'uint8', 'The inputs should be uint8 format within range (0, 255)'

    h,w,s,t = np.squeeze(truth4d).shape
    lfpsnr = np.zeros([s,t])
    for i in range(s):
        for j in range(t):
            truth = truth4d[:,:,i,j]
            truth = np.squeeze(truth)
            recons = recons4d[:,:,i,j]
            recons = np.squeeze(recons)
            lfpsnr[i,j] = psnr(truth, recons)
    meanpsnr = np.mean(lfpsnr)
    return lfpsnr, meanpsnr


def batchmeanpsnr(truth, pred):
    ''' This function calculate the average psnr over a batchsize of light field patches and
        the size of each batch equal to the length of inputs
        The inputs should be uint8 format within range (0, 1)
    '''
    # assert pred.dtype == 'uint8', 'The inputs should be uint8 format within range (0, 255)'
    
    batchmean_psnr = 0
    for i in range(len(pred)):
        _, meanpsnr = lfpsnrs(np.uint8(truth[i]*255.), np.uint8(pred[i]*255.))
        batchmean_psnr += meanpsnr
    batchmean_psnr = batchmean_psnr / len(pred)
    
    return batchmean_psnr


def ssim_exact(img1, img2, sd=1.5, C1=0.01**2, C2=0.03**2):

    mu1 = gaussian_filter(img1, sd)
    mu2 = gaussian_filter(img2, sd)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = gaussian_filter(img1 * img1, sd) - mu1_sq
    sigma2_sq = gaussian_filter(img2 * img2, sd) - mu2_sq
    sigma12 = gaussian_filter(img1 * img2, sd) - mu1_mu2

    ssim_num = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))

    ssim_den = ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    ssim_map = ssim_num / ssim_den
    return np.mean(ssim_map)

def lfssims(truth4d, recons4d):
    ''' The truth4d represents for a single 4d-patches of light field
        
        truth4d  = [height, width, sview, tview, channels]
        recons4d = [height, width, sview, tview, channels]
    '''
    assert truth4d.shape == recons4d.shape, 'The prediction and label should be same size.'
    
    h,w,s,t,c = truth4d.shape
    lfssim = np.zeros([s,t])
    for i in range(s):
        for j in range(t):
            truth = truth4d[:,:,i,j,:]
            truth = np.squeeze(truth)
            recons = recons4d[:,:,i,j,:]
            recons = np.squeeze(recons)
            lfssim[i,j] = ssim_exact(truth/255., recons/255.)
    meanssim = np.mean(lfssim)
    return lfssim, meanssim


def batchmeanssim(truth, pred):
    ''' This function calculate the average psnr over a batchsize of light field patches and
        the size of each batch equal to the length of inputs
        The inputs should be uint8 format within range (0, 1)
    '''
    # assert pred.dtype == 'uint8', 'The inputs should be uint8 format within range (0, 255)'
    
    batchmean_ssim = 0
    for i in range(len(pred)):
        _, meanssim = lfssims(np.uint8(truth[i]*255.), np.uint8(pred[i]*255.))
        batchmean_ssim += meanssim
    batchmean_ssim = batchmean_ssim / len(pred)
    
    return batchmean_ssim


##############################################################
######       split patches using overlap strides        ######
##############################################################
def img4d_splitpatches(img4d, patchsize=96, stride=30):
    ''' img4d = [batchsize, height, width, s_view, t_view, channels]
    '''
    _, height, width, sSize, tSize, channels = img4d.shape
    nh = (height - patchsize) // stride + 1
    nw = (width  - patchsize) // stride + 1
    
    patch_squence = []
    indices_squence = []
    for y in range(0, height-patchsize+1, stride):
        for x in range(0, width-patchsize+1, stride):
            patch = img4d[:, y:y+patchsize, x:x+patchsize,:,:,:]
            indices = np.array([y,y+patchsize,x,x+patchsize])
            patch_squence.append(patch)
            indices_squence.append(indices)
        patch = img4d[:,y:y+patchsize, width-patchsize:, :,:,:]
        indices = np.array([y,y+patchsize, width-patchsize, width])
        patch_squence.append(patch)
        indices_squence.append(indices)
    
    for x in range(0, width-patchsize+1, stride):
        patch = img4d[:,height-patchsize:, x:x+patchsize, :,:,:]
        indices = np.array([height-patchsize, height, x,x+patchsize])
        patch_squence.append(patch)
        indices_squence.append(indices)
    
    patch = img4d[:,height-patchsize:, width-patchsize:, :,:,:]
    indices = np.array([height-patchsize, height, width-patchsize, width])
    patch_squence.append(patch)
    indices_squence.append(indices)
            
    return patch_squence, np.array(indices_squence)


def shaved_img4d_reconstruct(patch_squence, indices_squence, border=[3,3]):
    
    _, height, _, width = indices_squence[-1]
    batchsize, _, _, s, t, channels = patch_squence[0].shape
    
    img4d = np.zeros([1,height, width, s, t, channels], dtype=np.float32)
    for i in range(len(patch_squence)):
        h_start, h_off, w_start, w_off = indices_squence[i]
        img4d[0,h_start+border[0]:h_off-border[0],
              w_start+border[1]:w_off-border[1], ...] = shave4d(patch_squence[i], border)
    
    # print shave4d(patch_squence[i], border).shape
    
    img4d = shave4d(img4d, border)
    return img4d






############################################################################
######                    read lightfield images                      ######
############################################################################

def imread(path, is_grayscale=False):
    """
    Read image using its path.
    Default value is gray-scale, and image is read by YCbCr format as the paper said.
    """
    if is_grayscale:
        return scipy.misc.imread(path, mode='YCbCr').astype(np.float32)[:,:,:1]
    else:
        return scipy.misc.imread(path, mode='RGB').astype(np.float32)


def read_single_lf4d(lf_folder, viewSize=5, channels=3, normalize=False):
    img_list = glob.glob(os.path.join(lf_folder, '*.jpg'))
    suffix = [re.findall('([0-9]+x[0-9]+)', i)[0] for i in img_list]
    sorted_suffix = sorted(suffix, key=lambda x: (int(x.split('x')[0]), int(x.split('x')[1])))
    # print 'sorted_suffix: ', sorted_suffix
    
    ori_view = np.sqrt(len(img_list)).astype(np.int)
    v_start = (ori_view - viewSize) / 2
    
    height, width, _ = imread(img_list[0]).shape
    if normalize == True:
        lf = np.zeros([1, height, width, viewSize, viewSize, channels], dtype=np.float32)
    else:
        lf = np.zeros([1, height, width, viewSize, viewSize, channels], dtype=np.uint8)
    is_grayscale = True
    if channels == 3:
        is_grayscale = False
    
    for i in range(viewSize):
        for j in range(viewSize):
            select_suf = sorted_suffix[v_start+(i+v_start)*ori_view+j]
            img_ = imread(img_list[suffix.index(select_suf)], is_grayscale)
            if normalize == True:
                lf[0,:,:,i,j,:] = Normal(img_)
            else:
                lf[0,:,:,i,j,:] = img_
    return lf


def read_lf4d(filenames, viewSize=5, is_grayscale=False):
    assert not isinstance(filenames, basestring), 'The 1st input is not a list'
    
    batchNum = len(filenames)
    if is_grayscale:
        channels = 1
    else:
        channels = 3
    # lfimgs = np.zeros([batchNum, imageSize, imageSize, viewSize, viewSize, channels], dtype=np.float32)
    for i in range(batchNum):
        tmpimgs = read_single_lf4d(filenames[i], viewSize=viewSize, channels=channels)
        if i == 0:
            lfimgs = tmpimgs
        else:
            lfimgs = np.vstack([lfimgs, tmpimgs])
    return lfimgs


def read_batch_lf4d(filenames, viewSize=5, is_grayscale=False, iter=0, 
                    batch_size=30, normalize=True):
    # assert not isinstance(filenames, basestring), 'The 1st input is not a list'

    if filenames is None:
        raise ValueError('The input filelist is empty.')
    
    ## choose batchsize of filelist
    num_files = len(filenames)
    if batch_size is None:
        batch_size = num_files
    
    filestart = iter * batch_size % num_files
    fileend   = (iter+1) * batch_size % num_files
    if filestart >= fileend:
        batch_files = filenames[filestart:] + filenames[:fileend]
    else:
        batch_files = filenames[filestart:fileend]

    batchNum = len(batch_files)
    if is_grayscale:
        channels = 1
    else:
        channels = 3
    
    templf = read_single_lf4d(filenames[0], viewSize=viewSize, channels=channels)
    _,h,w,s,t,c = templf.shape
    lfimgs = np.zeros([batch_size, h, w, s, t, c], dtype=np.float32)

    for i in range(batchNum):
        tmpimgs = read_single_lf4d(filenames[i], viewSize=viewSize, channels=channels)[:,:h,:w,...]
        lfimgs[i] = tmpimgs

    return lfimgs.astype(np.uint8)




############################################################################
######                      processing functions                      ######
############################################################################

def rgb2gray4d(rgb):

    r, g, b = rgb[...,0], rgb[...,1], rgb[...,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    gray = gray.reshape([i for i in gray.shape] + [-1])
    
    return gray

def modcrop(image, scale=3):
    """
    To scale down and up the original image, first thing to do is to have no remainder while scaling operation.
    
    We need to find modulo of height (and width) and scale factor.
    Then, subtract the modulo from height (and width) of original image size.
    There would be no remainder even after scaling operation.
    """
    if len(image.shape) == 3:
        h, w, _ = image.shape
        h = h - np.mod(h, scale)
        w = w - np.mod(w, scale)
        image = image[0:h, 0:w, :]
    else:
        h, w = image.shape
        h = h - np.mod(h, scale)
        w = w - np.mod(w, scale)
        image = image[0:h, 0:w]
    return image


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h*size[0], w*size[1], 1), dtype=np.float32)
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img


def Normal(_data):
    ''' This function return the normalization of image data

        input :  image:               _data

        output:  normalized_image:    data
    '''
    data = _data.copy()
    data = data.astype(np.float32)
    data = (data-np.min(data)) / (np.max(data) - np.min(data))
    return data


def shave4d(img4d, border=[3,3]):
    h_border, w_border = border
    if (h_border != 0) and (w_border != 0):
        img4d_shaved = img4d[:,h_border:-h_border, w_border:-w_border, :,:,:]
    elif (h_border != 0) and (w_border == 0):
        img4d_shaved = img4d[:,h_border:-h_border,:,:,:,:]
    elif (h_border == 0) and (w_border != 0):
        img4d_shaved = img4d[:,:,w_border:-w_border,:,:,:]
    elif (h_border == 0) and (w_border == 0):
        img4d_shaved = img4d[:,:,:,:,:,:]
    return img4d_shaved


def save_img(imgs, label, epoch, batch, channels=3):
    batch_size = imgs[0].shape[0]
    for i in range(batch_size):
        fig = plt.figure()
        for j, img in enumerate(imgs):
            im = np.uint8(Normal(img[i])*255.)
            if channels == 1:
                im = np.squeeze(im)
            # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            fig.add_subplot(1, len(imgs), j+1)
            plt.imshow(im)
            plt.tick_params(labelbottom='off')
            plt.tick_params(labelleft='off')
            plt.gca().get_xaxis().set_ticks_position('none')
            plt.gca().get_yaxis().set_ticks_position('none')
            plt.xlabel(label[j])
        seq_ = "{0:09d}".format(i+1)
        epoch_ = "{0:09d}".format(epoch)
        path = os.path.join('result', seq_, 'epoch_{0}_batch_{1}_{2:02f}.png'.format(epoch_, batch, np.random.random()))
        if os.path.exists(os.path.join('result', seq_)) == False:
            os.makedirs(os.path.join('result', seq_))
        plt.savefig(path)
        plt.close()



def img4duint8(img4d):
    ''' img4d = [batchsize=1, height, width, sview, tview, channels]
    '''
    batchsize, height, width, sview, tview, channels = img4d.shape
    uint8img4d = np.zeros_like(img4d, dtype=np.float32)
    for i in range(sview):
        for j in range(tview):
            uint8img4d[0,:,:,i,j,:] = np.uint8(img4d[0,:,:,i,j,:]*255.)
    return uint8img4d




################################################################################
######                            reading data                            ######
################################################################################



##################################### center crop #######################################

def crop_center(img, cropsize):
    '''
    img : [H, W, C]
    '''
    h,w,c = img.shape
    return img[(h-cropsize)/2:(h+cropsize)/2, (w-cropsize)/2:(w+cropsize)/2, :]


def read_lf(lf_folder, imageSize=96, viewSize=5, channels=3):
    '''read single light field images'''
    
    img_list = glob.glob(os.path.join(lf_folder, '*.jpg'))
    suffix = [re.findall('([0-9]+x[0-9]+)', i)[0] for i in img_list]
    sorted_suffix = sorted(suffix, key=lambda x: (int(x.split('x')[0]), int(x.split('x')[1])))
    # print 'sorted_suffix: ', sorted_suffix
    
    ori_view = np.sqrt(len(img_list)).astype(np.int)
    v_start = (ori_view - viewSize) / 2
    
    
    lf_image = np.zeros([channels, imageSize, imageSize, viewSize, viewSize], dtype=np.float32)
    
    if channels == 3:
        for i in range(viewSize):
            for j in range(viewSize):
                select_suf = sorted_suffix[v_start+(i+v_start)*ori_view+j]
                img_ = scipy.misc.imread(img_list[suffix.index(select_suf)])

                # img_ = crop_center(img_, imageSize)
                img_ = crop_center(img_, 240)
                img_ = imresize(img_, (imageSize,imageSize))
                lf_image[:,:,:,i,j] = Normal(img_.transpose(2,0,1))
    
    elif channels == 1:
        for i in range(viewSize):
            for j in range(viewSize):
                select_suf = sorted_suffix[v_start+(i+v_start)*ori_view+j]
                img_ = scipy.misc.imread(img_list[suffix.index(select_suf)], mode = 'YCbCr')

                # img_ = crop_center(img_, imageSize)
                img_ = crop_center(img_, 240)
                img_ = imresize(img_, (imageSize,imageSize))[:,:,:1]
                lf_image[:,:,:,i,j] = Normal(img_.transpose(2,0,1))
    
    return lf_image



def get_inputs(filenames=None, imageSize=96, viewSize=9, channels=3, iter=0, batch_size=None):
    
    
    # Read each Lightfield file
    if filenames is None:
        raise ValueError('The input filelist is empty.')
    
    ## choose batchsize of filelist
    num_files = len(filenames)
    if batch_size is None:
        batch_size = num_files
    
    filestart = iter * batch_size % num_files
    fileend   = (iter+1) * batch_size % num_files
    if filestart >= fileend:
        batch_files = filenames[filestart:] + filenames[:fileend]
    else:
        batch_files = filenames[filestart:fileend]
    
    
    ## read images
    batchNum = len(batch_files)
    labels = np.zeros([batchNum, channels, imageSize, imageSize, viewSize, viewSize], dtype=np.float32)
    
    # print 'start reading images ...'
    for i in range(batchNum):
        labels[i] = read_lf(batch_files[i], imageSize=imageSize, 
                            viewSize=viewSize, channels=channels)
    # print 'finished.'
    # features = labels[:,:,::spacial_rate,::spacial_rate,::view_rate,::view_rate]
    
    labels = labels.transpose(0,2,3,4,5,1)      # reshape the data to : [batchsize, img_h, img_w, view_s, view_t, channels]
    # features = features.transpose(0,2,3,4,5,1)  # reshape the data to : [batchsize, img_h, img_w, view_s, view_t, channels]
    
    return labels

##################################### center crop #######################################
# ===================================================================================== #




##################################### random crop #######################################


def read_lf_randcrop(lf_folder, imageSize=96, viewSize=5, channels=3):
    '''read single light field images'''
    
    img_list = glob.glob(os.path.join(lf_folder, '*.jpg'))
    suffix = [re.findall('([0-9]+x[0-9]+)', i)[0] for i in img_list]
    sorted_suffix = sorted(suffix, key=lambda x: (int(x.split('x')[0]), int(x.split('x')[1])))
    # print 'sorted_suffix: ', sorted_suffix
    
    ori_view = np.sqrt(len(img_list)).astype(np.int)
    v_start = (ori_view - viewSize) / 2
    
    lf_image = np.zeros([channels, imageSize, imageSize, viewSize, viewSize], dtype=np.float32)
    height, width, _ = scipy.misc.imread(img_list[0]).shape
    h_offset = random.randint(60, height-imageSize-60)
    w_offset = random.randint(80, width-imageSize-80)
    
    if channels == 3:
        for i in range(viewSize):
            for j in range(viewSize):
                select_suf = sorted_suffix[v_start+(i+v_start)*ori_view+j]
                img_ = scipy.misc.imread(img_list[suffix.index(select_suf)])

                # img_ = crop_center(img_, imageSize)
                # img_ = crop_center(img_, 240)
                # img_ = imresize(img_, (imageSize,imageSize))
                img_ = img_[h_offset:h_offset+imageSize, w_offset:w_offset+imageSize, :]
                lf_image[:,:,:,i,j] = Normal(img_.transpose(2,0,1))
    
    elif channels == 1:
        for i in range(viewSize):
            for j in range(viewSize):
                select_suf = sorted_suffix[v_start+(i+v_start)*ori_view+j]
                img_ = scipy.misc.imread(img_list[suffix.index(select_suf)], mode='YCbCr')

                # img_ = crop_center(img_, imageSize)
                # img_ = crop_center(img_, 240)
                # img_ = imresize(img_, (imageSize,imageSize))
                img_ = img_[h_offset:h_offset+imageSize, w_offset:w_offset+imageSize, :1]
                lf_image[:,:,:,i,j] = Normal(img_.transpose(2,0,1))
    
    return lf_image



def get_inputs_randcrop(filenames=None, imageSize=96, viewSize=9, channels=3, 
                        iter=0, batch_size=None):
    
    
    # Read each Lightfield file
    if filenames is None:
        raise ValueError('The input filelist is empty.')
    
    ## choose batchsize of filelist
    num_files = len(filenames)
    if batch_size is None:
        batch_size = num_files
    
    filestart = iter * batch_size % num_files
    fileend   = (iter+1) * batch_size % num_files
    if filestart >= fileend:
        batch_files = filenames[filestart:] + filenames[:fileend]
    else:
        batch_files = filenames[filestart:fileend]
    
    
    ## read images
    batchNum = len(batch_files)
    labels = np.zeros([batchNum, channels, imageSize, imageSize, viewSize, viewSize], dtype=np.float32)
    
    # print 'start reading images ...'
    for i in range(batchNum):
        labels[i] = read_lf_randcrop(batch_files[i], imageSize=imageSize, viewSize=viewSize, 
                                     channels=channels)
    # print 'finished.'
    
    labels = labels.transpose(0,2,3,4,5,1)      # reshape the data to : [batchsize, img_h, img_w, view_s, view_t, channels]

    return labels

##################################### random crop #######################################
# ===================================================================================== #



################################################################################
######                           check and test                           ######
################################################################################

def check_img4d(sess, img4d):
    
    multichannel = True
    if img4d.shape[-1] == 1:
        multichannel = False
    
    label_patches, label_patches_indices = img4d_splitpatches(img4d)
    reconstruct_patches = np.zeros_like(label_patches, dtype=np.float32)
    uint8_labelpatches = np.zeros_like(label_patches, dtype=np.float32)
    for num in tqdm(range(len(label_patches))):
        raw = label_patches[num].reshape([-1] + [k for k in label_patches.shape[1:]])
        mos, fake = sess.run([model.downscaled, model.gene_outputs], feed_dict={x: raw, is_training: False})
        reconstruct_patches[num] = img4duint8(fake)
        
        uint8_labelpatches[num] = img4duint8(raw)
    
    reconstruct_img4d = shaved_img4d_reconstruct(reconstruct_patches, label_patches_indices)
    original_img4d = shaved_img4d_reconstruct(uint8_labelpatches, label_patches_indices)
    
    recon_center = reconstruct_img4d[0,:,:,2,2,:]
    orig_center  = original_img4d[0,:,:,2,2,:]
    print 'center image: image[{0}]  psnr: {1}, mse: {2}, ssim: {3}'.format(i, psnr(recon_center, orig_center), 
                                                                            mse(recon_center, orig_center), 
                                                                            ssim(recon_center, orig_center,
                                                                                multichannel=multichannel))
    return reconstruct_img4d, original_img4d


def checkimg4d(sess, img4d, stride=60, channels=3):
    patches, indices = img4d_splitpatches(img4d, stride=stride)

    recons_patches = []
    for num in tqdm(range(len(patches))):
#         if
#         print '[{0}/{1}]'.format(num, len(patches)),
        raw = patches[num].reshape([-1] + [k for k in patches[0].shape[1:]])
        mos, fake = sess.run([model.downscaled, model.gene_outputs], feed_dict={x: raw, is_training: False})
        recons_patches.append(fake)
        # cal_metrics(fake[:,:,:,3,3,:], raw[:,:,:,3,3,:])
    
    raw4d = shaved_img4d_reconstruct(patches, indices)
    recons4d = shaved_img4d_reconstruct(recons_patches, indices)
    
    label_center = raw4d[0,:,:,2,2,:].copy()
    recons_center = recons4d[0,:,:,2,2,:].copy()
    
    PSNR = psnr(np.uint8(Normal(label_center)*255.), np.uint8(Normal(recons_center)*255.))
    SSIM = ssim(np.uint8(Normal(label_center)*255.), np.uint8(Normal(recons_center)*255.), multichannel=True)
    MSE  = mse(np.uint8(Normal(label_center)*255.), np.uint8(Normal(recons_center)*255.))
    print 'psnr:', PSNR
    print 'ssim:', SSIM
    print 'mse :', MSE
    
    
    ### compared with other methods
    methods = ['nearest', 'lanczos', 'bilinear', 'bicubic', 'cubic']
    img = np.uint8(Normal(raw4d[0,:,:,2,2,:])*255.)
    for med in methods:
        if channels == 3:
            cal_metrics_3channels(np.uint8(Normal(img)*255.), method=med, K=2)
        elif channels == 1:
            cal_metrics_1channel(np.uint8(Normal(img)*255.), method=med, K=2)
    
    return PSNR, SSIM