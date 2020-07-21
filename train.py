import argparse
import os

if __name__ == "__main__":

    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # General parameters
    parser.add_argument('--save_path', type = str, default = './models', help = 'saving path that is a folder')
    parser.add_argument('--sample_path', type = str, default = './samples', help = 'training samples path that is a folder')
    parser.add_argument('--gan_type', type = str, default = 'WGAN', help = 'the type of GAN for training')
    parser.add_argument('--multi_gpu', type = bool, default = False, help = 'nn.Parallel needs or not')
    parser.add_argument('--gpu_ids', type = str, default = "0", help = 'gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--cudnn_benchmark', type = bool, default = True, help = 'True for unchanged input data type')
    parser.add_argument('--checkpoint_interval', type = int, default = 1, help = 'interval between model checkpoints')
    parser.add_argument('--load_name_g', type = str, default = './models/deepfillv2_WGAN_epoch1_batchsize4.pth', help = 'load model name')#./models/deepfillv2_WGAN_epoch2_batchsize4.pth
    parser.add_argument('--load_name_d', type=str, default='./models/discriminator_WGAN_epoch1_batchsize4.pth', help='load model name')#./models/discriminator_WGAN_epoch2_batchsize4.pth
    # Training parameters
    parser.add_argument('--epochs', type = int, default = 40, help = 'number of epochs of training')
    parser.add_argument('--batch_size', type = int, default = 4, help = 'size of the batches')
    parser.add_argument('--lr_g', type = float, default = 1e-4, help = 'Adam: learning rate')
    parser.add_argument('--lr_d', type = float, default = 4e-4, help = 'Adam: learning rate')
    parser.add_argument('--weight_decay', type = float, default = 0, help = 'Adam: weight decay')
    parser.add_argument('--lr_decrease_epoch', type = int, default = 10, help = 'lr decrease at certain epoch and its multiple')
    parser.add_argument('--lr_decrease_factor', type = float, default = 0.5, help = 'lr decrease factor, for classification default 0.1')
    parser.add_argument('--lambda_l1', type = float, default = 256, help = 'the parameter of L1Loss')
    parser.add_argument('--lambda_perceptual', type = float, default = 100, help = 'the parameter of FML1Loss (perceptual loss)')
    parser.add_argument('--num_workers', type = int, default = 16, help = 'number of cpu threads to use during batch generation')
    # Network parameters
    parser.add_argument('--latent_channels', type = int, default = 32, help = 'latent channels')
    parser.add_argument('--pad_type', type = str, default = 'replicate', help = 'the padding type')
    parser.add_argument('--activation', type = str, default = 'elu', help = 'the activation type')
    parser.add_argument('--norm1', type = str, default = 'in', help = 'normalization type')
    parser.add_argument('--norm', type=str, default='in', help='normalization type')
    parser.add_argument('--init_type', type = str, default = 'kaiming', help = 'the initialization type')
    parser.add_argument('--init_gain', type = float, default = 0.2, help = 'the initialization gain')
    # Dataset parameters
    parser.add_argument('--baseroot', type = str, default = './dataset/data_large', help = 'the training folder: val_256, test_large, data_256')
    parser.add_argument('--mask_type', type = str, default = 'free_form', help = 'mask type')
    parser.add_argument('--imgsize', type = int, default = 512, help = 'size of image')
    parser.add_argument('--margin', type = int, default = 10, help = 'margin of image')
    parser.add_argument('--bbox_shape', type = int, default = 30, help = 'margin of image for bbox mask')
    opt = parser.parse_args()
    print(opt)
    
    '''
    # ----------------------------------------
    #       Choose CUDA visible devices
    # ----------------------------------------
    if opt.multi_gpu == True:
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    '''
    
    # Enter main function
    import trainer
    if opt.gan_type == 'WGAN':
        trainer.WGAN_trainer(opt)
    if opt.gan_type == 'LSGAN':
        trainer.LSGAN_trainer(opt)
    