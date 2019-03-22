import numpy as np
from skimage.color import rgb2gray

def rgb2gray(rgb):

    r, g, b = rgb[...,0], rgb[...,1], rgb[...,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    gray = gray.reshape([i for i in gray.shape] + [-1])
    
    return gray


def readlfbatches(filespath, imageSize = 96, viewSize = 5, channels = 3):
    '''
        inputs:  filespath  --  list of path strings
        
        outputs: batchlfimgs
    '''

    batchsize = len(filespath)
    batchlfimgs = np.zeros([batchsize, imageSize, imageSize, viewSize, viewSize, channels], dtype=np.uint8)

    for i in range(batchsize):
        # read imgs
        print i,
        lfimg = np.load(filespath[i])
        s, t, height, width, c = lfimg.shape

        h_offset = random.randint(60, height-imageSize-60)
        w_offset = random.randint(80, width-imageSize-80)

        s_start = (s - viewSize) / 2
        s_end   = (s + viewSize) / 2
        t_start = (t - viewSize) / 2
        t_end   = (t + viewSize) / 2
        if channels == 3:
            # print 'color'
            lfimg = lfimg[s_start:s_end, t_start:t_end, 
                          h_offset:h_offset+imageSize, 
                          w_offset:w_offset+imageSize, :]
        elif channels == 1:
            # print 'gray'
            lfimg = rgb2gray4d(lfimg)
            lfimg = lfimg[s_start:s_end, t_start:t_end, 
                          h_offset:h_offset+imageSize, 
                          w_offset:w_offset+imageSize, :]

        batchlfimgs[i] = lfimg.transpose(2,3,0,1,4)
        # print lfimg.shape
    return batchlfimgs


def readlfiteration(filespath=None, batchsize=100, imageSize = 96, viewSize = 5, channels = 3, iteration=0):
    # Read each Lightfield file
    if filespath is None:
        raise ValueError('The input filelist is empty.')
    
    ## choose batchsize of filelist
    num_files = len(filespath)
    if batchsize is None:
        batchsize = num_files
    
    filestart = iteration * batchsize % num_files
    fileend   = (iteration+1) * batchsize % num_files
    if filestart >= fileend:
        batch_files = filespath[filestart:] + filespath[:fileend]
    else:
        batch_files = filespath[filestart:fileend]
    
    lfbatch = readlfbatches(batch_files, imageSize = imageSize, 
                            viewSize = viewSize, channels = channels)
    return lfbatch