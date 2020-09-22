from utils.layers import *
from utils.convolve4d import *
from vgg19.vgg19 import VGG19
import pdb


def ConvBNLReLU(x, in_channels, out_channels, kernel_size_1=3, kernel_size_2=3, padding="SAME", trainable=True, is_training=True, name="Conv4DBNLReLU"):
    with tf.variable_scope(name):
        x = conv4d(x, in_channels, out_channels, kernel_size_1=kernel_size_1, kernel_size_2=kernel_size_2, padding=padding, trainable=trainable)
        x = batch_normalize(x, is_training)
        out = leakyrelu(x)
    return out


def ConvLReLU(x, in_channels, out_channels, kernel_size_1=3, kernel_size_2=3, padding="SAME", trainable=True, name="Conv4DLReLU"):
    with tf.variable_scope(name):
        x = conv4d(x, in_channels, out_channels, kernel_size_1=kernel_size_1, kernel_size_2=kernel_size_2, padding=padding, trainable=trainable)
        out = leakyrelu(x)
    return out


def ConvBN(x, in_channels, out_channels, kernel_size_1=3, kernel_size_2=3, padding="SAME", trainable=True, is_training=True, name="Conv4DBN"):
    with tf.variable_scope(name):
        x = conv4d(x, in_channels, out_channels, kernel_size_1=kernel_size_1, kernel_size_2=kernel_size_2, padding=padding, trainable=trainable)
        out = batch_normalize(x, is_training)
    return out


def HRB(x, in_channels, out_channels, kernel_size_1=3, kernel_size_2=3, padding="SAME", trainable=True, is_training=True, name="HRB"):
    with tf.variable_scope(name):
        out = ConvBNLReLU(x, in_channels, out_channels, kernel_size_1=kernel_size_1, kernel_size_2=kernel_size_2, padding=padding, trainable=trainable, is_training=is_training)
        out = ConvBN(out, in_channels, out_channels, kernel_size_1=kernel_size_1, kernel_size_2=kernel_size_2, padding=padding, trainable=trainable, is_training=is_training)
        out = tf.add(out, x)
    return out


def SpatialPixelShuffle(x, sr):
    batchsize, in_height, in_width, in_sview, in_tview, channels = x.get_shape().as_list()
    out_channels = channels / (sr ** 2)
    out_height = in_height * sr
    out_width = in_width * sr
    out_sview = in_sview
    out_tview = in_tview
    
    out = tf.transpose(x, [0, 5, 1, 2, 3, 4])
    out = tf.reshape(out, [batchsize, out_channels, sr, sr, in_height, in_width, in_sview, in_tview])
    out = tf.transpose(out, [0, 1, 4, 2, 5, 3, 6, 7])
    out = tf.reshape(out, [batchsize, out_channels, out_height, out_width, out_sview, out_tview])
    out = tf.transpose(out, [0, 2, 3, 4, 5, 1])
    return out


def AngularPixelShuffle(x, ar):
    batchsize, in_height, in_width, in_sview, in_tview, channels = x.get_shape().as_list()
    out_channels = channels / (ar ** 2)
    out_sview = in_sview * ar
    out_tview = in_tview * ar
    
    out = tf.transpose(x, [0, 1, 2, 5, 3, 4])
    out = tf.reshape(out, [batchsize, in_height, in_width, out_channels, ar, ar, in_sview, in_tview])
    out = tf.transpose(out, [0, 1, 2, 3, 6, 4, 7, 5])
    out = tf.reshape(out, [batchsize, in_height, in_width, out_channels, out_sview, out_tview])
    return out


def Upscaling_Spatial_and_Angular(x, ar=2, sr=2):
    sh = x.get_shape().as_list()
    dim = len(sh[1:-1])
    out = tf.reshape(x, [-1] + sh[-dim:])
    for i in range(dim, 2, -1):
        out = tf.concat([out, out], i)
    out_size = [-1] + [s for s in sh[1:3]] + [s*ar for s in sh[3:-1]] + [sh[-1]]
    out = tf.reshape(out, out_size)
    
    if tf.shape(out)[4] % 2 == 0:
        out = out[:, :, :, 1:, 1:, :]
    
    out = SpatialPixelShuffle(out, sr)
    return out
    

def UpNet(x, sr=2, ar=2):
    if ar == 1:
        if sr == 2 or sr == 3:
            x = conv4d(x, 64, 64*sr*sr, kernel_size_1=3, kernel_size_2=3, padding="SAME", trainable=True)
            out = SpatialPixelShuffle(x, sr)
        if sr == 4:
            with tf.variable_scope("4x_01"):
                x = conv4d(x, 64, 64*2*2, kernel_size_1=3, kernel_size_2=3, padding="SAME", trainable=True)
                x = SpatialPixelShuffle(x, 2)
            with tf.variable_scope("4x_02"):
                x = conv4d(x, 64, 64*2*2, kernel_size_1=3, kernel_size_2=3, padding="SAME", trainable=True)
                out = SpatialPixelShuffle(x, 2)
    elif sr == 1:
        out = AngularPixelShuffle(x, ar)
    else:
        out = Upscaling_Spatial_and_Angular(x, ar, sr)
    return out


# ==================================================================================== #
#                                        LFWGAN                                        #
# ==================================================================================== #

class LFGAN(object):
    '''
    The LFGAN Model (Tensorflow >= 1.8)
    '''

    def __init__(self, inputs, targets, is_training, args):
        super(LFGAN, self).__init__()
        self.gamma_A = args.gamma_A
        self.gamma_S = args.gamma_S
        self.channels = args.Channels
        self.gene_variables = None
        self.disc_variables = None
        self.num_GRL_HRB = args.num_GRL_HRB
        self.num_SRe_HRB = args.num_SRe_HRB
        self.GRLfeats = []
        self.SRefeats = []
        self.vgg = VGG19(None, None, None)
        self.PreRecons, self.Recons = self.generator(inputs, is_training, reuse=False)
        self.Disc_real = self.discriminator(targets, is_training, False)
        self.Disc_fake = self.discriminator(self.Recons, is_training, True)
        self.g_loss, self.d_loss = self.compute_loss(targets, self.Recons, self.PreRecons, self.Disc_real, self.Disc_fake)

    # =================================================== Generator ================================================== #
    def generator(self, x, is_training, reuse):
        with tf.variable_scope('LFGAN_Gene', reuse=reuse):
            with tf.variable_scope('SFEfeat_1'):
                x = conv4d(x, self.channels, 64, kernel_size_1=3, kernel_size_2=3, padding='SAME', trainable=True)
                x = leakyrelu(x)
            SFEfeat1 = x

            # =============================  Geometric Representation Learning Net (GRLNet) ========================== #
            with tf.variable_scope("GRLNet"):
                for i in range(self.num_GRL_HRB):
                    out = HRB(x, 64, 64, kernel_size_1=3, kernel_size_2=3, padding="SAME", trainable=True, is_training=is_training, name="RDB_{0:02d}".format(i))
                    x = x + out
                    self.GRLfeats.append(out)
            x = tf.add(SFEfeat1, x)
             
            with tf.variable_scope("UPNet"):
                x = UpNet(x, self.gamma_S, self.gamma_A)
            
            with tf.variable_scope("PreRecons"):
                x = ConvLReLU(x, 64, 64, kernel_size_1=3, kernel_size_2=3, padding="SAME", trainable=True, name="Conv4DReLU")
                x = conv4d(x, 64, self.channels, kernel_size_1=1, kernel_size_2=1, padding="SAME", trainable=True)
            PreRecons = x
            
            # ==================================== Spatial Refine Network (SReNet) =================================== #
            with tf.variable_scope("SFENet_2"):
                x = conv4d(x, self.channels, 64, kernel_size_1=3, kernel_size_2=5, padding="SAME", trainable=True)
            SFEfeat2 = x
            
            with tf.variable_scope("SReNet"):
                for i in range(self.num_SRe_HRB):
                    out = HRB(x, 64, 64, kernel_size_1=3, kernel_size_2=3, padding="SAME", trainable=True, is_training=is_training, name="RDB_{0:02d}".format(i))
                    x = x + out
                    self.SRefeats.append(out)
            x = tf.add(SFEfeat2, x)
            
            with tf.variable_scope("Output"):
                Recons = conv4d(x, 64, self.channels, kernel_size_1=3, kernel_size_2=3, padding='SAME', trainable=True)
                
        self.gene_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='LFGAN_Gene')
        return PreRecons, Recons

    # ================================================= Discriminator ================================================ #
    def discriminator(self, x, is_training, reuse):
        with tf.variable_scope('LFGAN_Disc', reuse=reuse):
            with tf.variable_scope('conv4d1'):
                x = conv4d(x, self.channels, 64, kernel_size_1=3, kernel_size_2=3, padding='SAME', trainable=True)
                x = leakyrelu(x)
            with tf.variable_scope('conv4d2'):
                x = conv4d(x, 64, 64, kernel_size_1=3, kernel_size_2=3, padding='SAME', trainable=True)
                x = leakyrelu(x)
                x = batch_normalize(x, is_training)
            with tf.variable_scope('conv4d3'):
                x = conv4d(x, 64, 128, kernel_size_1=3, kernel_size_2=3, stride_1=2, stride_2=2, padding='SAME', trainable=True)
                x = leakyrelu(x)
                x = batch_normalize(x, is_training)
            with tf.variable_scope('conv4d4'):
                x = conv4d(x, 128, 128, kernel_size_1=3, kernel_size_2=3, padding='SAME', trainable=True)
                x = leakyrelu(x)
                x = batch_normalize(x, is_training)
            with tf.variable_scope('conv4d5'):
                x = conv4d(x, 128, 256, kernel_size_1=3, kernel_size_2=3, stride_1=2, stride_2=1, padding='SAME', trainable=True)
                x = leakyrelu(x)
                x = batch_normalize(x, is_training)
            with tf.variable_scope('conv4d6'):
                x = conv4d(x, 256, 256, kernel_size_1=3, kernel_size_2=3, padding='SAME', trainable=True)
                x = leakyrelu(x)
                x = batch_normalize(x, is_training)
            with tf.variable_scope('conv4d7'):
                x = conv4d(x, 256, 512, kernel_size_1=3, kernel_size_2=3, stride_1=2, stride_2=1, padding='SAME', trainable=True)
                x = leakyrelu(x)
                x = batch_normalize(x, is_training)
            with tf.variable_scope('conv4d8'):
                x = conv4d(x, 512, 512, kernel_size_1=3, kernel_size_2=3, padding='SAME', trainable=True)
                x = leakyrelu(x)
                x = batch_normalize(x, is_training)
            with tf.variable_scope('conv4d9'):
                x = conv4d(x, 512, 1, kernel_size_1=3, kernel_size_2=3, padding='SAME', trainable=True)
            with tf.variable_scope('mean'):
                x = mean_layer(x)

        self.disc_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='LFGAN_Disc')
        return x

    # ============================================= Auxiliary Functions ============================================= #
    def views_flatten(self, x):
            batch, w, h, s, t, c = x.get_shape().as_list()
            x = tf.transpose(x, [0, 3, 4, 1, 2, 5])
            x_flatten = tf.reshape(x, (batch*s*t, w, h, c))
            return x_flatten

    def compute_loss(self, labels, recons, pre_recons, disc_real_output, disc_fake_output):

        def inference_content_loss(x, gene_output):
            _, x_phi = self.vgg.build_model(
                x, tf.constant(False), False)  # First
            _, gene_output_phi = self.vgg.build_model(
                gene_output, tf.constant(False), True)  # Second

            content_loss = None
            for i in range(len(x_phi)):
                l2_loss = tf.nn.l2_loss(x_phi[i] - gene_output_phi[i])
                if content_loss is None:
                    content_loss = l2_loss
                else:
                    content_loss = content_loss + l2_loss
            return tf.reduce_mean(content_loss)

        def inference_adversarial_loss(disc_real_output, disc_fake_output):
            # Adversarial loss of Generator
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake_output,
                                                                    labels=tf.ones_like(disc_fake_output))
            gene_adversarial_loss  = tf.reduce_mean(cross_entropy, name='gene_ce_loss')
            
            # Adversarial loss of Discriminator
            cross_entropy_real = tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real_output,
                                                                         labels=tf.ones_like(disc_real_output))
            disc_real_loss = tf.reduce_mean(cross_entropy_real, name='disc_real_loss')

            cross_entropy_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake_output,
                                                                         labels=tf.zeros_like(disc_fake_output))
            disc_fake_loss = tf.reduce_mean(cross_entropy_fake, name='disc_fake_loss')
            
            disc_adversarial_loss = tf.add(disc_real_loss, disc_fake_loss, name='disc_loss')
            
            return gene_adversarial_loss, disc_adversarial_loss
            
        def inference_spacial_loss(x, gene_output):
            spacial_loss = tf.reduce_mean(tf.abs(gene_output - x), name='spacial_loss')
            return spacial_loss

        def inference_angular_loss(x, gene_output):
            batchsize, h, w, s, t, c = x.get_shape().as_list()
            epi_x = tf.reshape(tf.transpose(x, [0,3,4,1,2,5]), [batchsize, h*s, w*t, -1])
            epi_gene_output = tf.reshape(tf.transpose(gene_output, [0,3,4,1,2,5]), [batchsize, h*s, w*t, -1])
            angular_loss = tf.reduce_mean(tf.abs(epi_x - epi_gene_output))
            return angular_loss


        self.angular_loss = tf.reduce_mean(tf.keras.losses.MSE(labels, pre_recons))

        self.content_loss = inference_content_loss(tf.tile(self.views_flatten(labels), (1, 1, 1, 3)),
                                                   tf.tile(self.views_flatten(recons), (1, 1, 1, 3)))
        self.gene_adversarial_loss, self.disc_loss = inference_adversarial_loss(disc_real_output, disc_fake_output)

        self.spatial_loss = 1e-5 * self.content_loss

        gene_losses = self.spatial_loss + self.angular_loss + self.gene_adversarial_loss

        disc_losses = self.disc_loss

        return gene_losses, disc_losses















