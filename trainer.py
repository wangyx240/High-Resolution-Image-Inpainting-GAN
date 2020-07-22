import itertools
import os
import time
import datetime
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.nn import functional as F

#import network
import inpainting_network
#import resnet_version
import dataset
import utils
import torch.autograd as autograd
from torch.autograd import Variable


def gradient_penalty (netD, real_data, fake_data, mask):
    alpha = torch.rand(1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netD.forward(interpolates, mask)

    gradients = autograd.grad(
        outputs=disc_interpolates, inputs=interpolates, grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()*10
    return gradient_penalty

def WGAN_trainer(opt):
    # ----------------------------------------
    #      Initialize training parameters
    # ----------------------------------------

    # cudnn benchmark accelerates the network
    cudnn.benchmark = opt.cudnn_benchmark
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)

    # configurations
    save_folder = opt.save_path
    sample_folder = opt.sample_path
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if not os.path.exists(sample_folder):
        os.makedirs(sample_folder)

    # Build networks
    generator = utils.create_generator(opt)
    discriminator = utils.create_discriminator(opt)
    perceptualnet = utils.create_perceptualnet()

    # To device
    if opt.multi_gpu == True:
        generator = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        perceptualnet = nn.DataParallel(perceptualnet)
        perceptualnet = perceptualnet.cuda()
    else:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        perceptualnet = perceptualnet.cuda()

    # Loss functions
    L1Loss = nn.L1Loss()#reduce=False, size_average=False)
    RELU = nn.ReLU()

    # Optimizers
    optimizer_g1 = torch.optim.Adam(generator.coarse.parameters(), lr=opt.lr_g)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=opt.lr_g)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr = opt.lr_d)

    # Learning rate decrease
    def adjust_learning_rate(lr_in, optimizer, epoch, opt):
        """Set the learning rate to the initial LR decayed by "lr_decrease_factor" every "lr_decrease_epoch" epochs"""
        lr = lr_in * (opt.lr_decrease_factor ** (epoch // opt.lr_decrease_epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    # Save the model if pre_train == True
    def save_model(net, epoch, opt, batch=0, is_D=False):
        """Save the model at "checkpoint_interval" and its multiple"""
        if is_D==True:
            model_name = 'discriminator_WGAN_epoch%d_batch%d.pth' % (epoch + 1, batch)
        else:
            model_name = 'deepfillv2_WGAN_epoch%d_batch%d.pth' % (epoch+1, batch)
        model_name = os.path.join(save_folder, model_name)
        if opt.multi_gpu == True:
            if epoch % opt.checkpoint_interval == 0:
                torch.save(net.module.state_dict(), model_name)
                print('The trained model is successfully saved at epoch %d batch %d' % (epoch, batch))
        else:
            if epoch % opt.checkpoint_interval == 0:
                torch.save(net.state_dict(), model_name)
                print('The trained model is successfully saved at epoch %d batch %d' % (epoch, batch))
    
    # ----------------------------------------
    #       Initialize training dataset
    # ----------------------------------------

    # Define the dataset
    trainset = dataset.InpaintDataset(opt)
    print('The overall number of images equals to %d' % len(trainset))

    # Define the dataloader
    dataloader = DataLoader(trainset, batch_size = opt.batch_size, shuffle = True, num_workers = opt.num_workers, pin_memory = True)
    
    # ----------------------------------------
    #            Training and Testing
    # ----------------------------------------

    # Initialize start time
    prev_time = time.time()

    # Training loop
    for epoch in range(opt.epochs):
        print("Start epoch ", epoch+1, "!")
        for batch_idx, (img, mask) in enumerate(dataloader):

            # Load mask (shape: [B, 1, H, W]), masked_img (shape: [B, 3, H, W]), img (shape: [B, 3, H, W]) and put it to cuda
            img = img.cuda()
            mask = mask.cuda()

            # Generator output
            first_out, second_out = generator(img, mask)

            # forward propagation
            first_out_wholeimg = img * (1 - mask) + first_out * mask        # in range [0, 1]
            second_out_wholeimg = img * (1 - mask) + second_out * mask      # in range [0, 1]

            for wk in range(1):
                optimizer_d.zero_grad()
                fake_scalar = discriminator(second_out_wholeimg.detach(), mask)
                true_scalar = discriminator(img, mask)
                #W_Loss = -torch.mean(true_scalar) + torch.mean(fake_scalar)#+ gradient_penalty(discriminator, img, second_out_wholeimg, mask)
                hinge_loss = torch.mean(RELU(1-true_scalar)) + torch.mean(RELU(fake_scalar+1))
                loss_D = hinge_loss
                loss_D.backward(retain_graph=True)
                optimizer_d.step()

            ### Train Generator
            # Mask L1 Loss
            first_MaskL1Loss = L1Loss(first_out_wholeimg, img)
            second_MaskL1Loss = L1Loss(second_out_wholeimg, img)
            # GAN Loss
            fake_scalar = discriminator(second_out_wholeimg, mask)
            GAN_Loss = - torch.mean(fake_scalar)

            optimizer_g1.zero_grad()
            first_MaskL1Loss.backward(retain_graph=True)
            optimizer_g1.step()

            optimizer_g.zero_grad()

            # Get the deep semantic feature maps, and compute Perceptual Loss
            img_featuremaps = perceptualnet(img)  # feature maps
            second_out_wholeimg_featuremaps = perceptualnet(second_out_wholeimg)
            second_PerceptualLoss = L1Loss(second_out_wholeimg_featuremaps, img_featuremaps)

            loss = 0.5*opt.lambda_l1 * first_MaskL1Loss + opt.lambda_l1 * second_MaskL1Loss + GAN_Loss + second_PerceptualLoss * opt.lambda_perceptual
            loss.backward()

            optimizer_g.step()

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + batch_idx
            batches_left = opt.epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()
            # Print log
            print("\r[Epoch %d/%d] [Batch %d/%d] [first Mask L1 Loss: %.5f] [second Mask L1 Loss: %.5f]" %
                  ((epoch + 1), opt.epochs, (batch_idx+1), len(dataloader), first_MaskL1Loss.item(),
                   second_MaskL1Loss.item()))
            print("\r[D Loss: %.5f] [Perceptual Loss: %.5f] [G Loss: %.5f] time_left: %s" %
                  (loss_D.item(), second_PerceptualLoss.item(), GAN_Loss.item(), time_left))



            if (batch_idx + 1) % 100 ==0:

                # Generate Visualization image
                masked_img = img * (1 - mask) + mask
                img_save = torch.cat((img, masked_img, first_out, second_out, first_out_wholeimg, second_out_wholeimg),3)
                # Recover normalization: * 255 because last layer is sigmoid activated
                img_save = F.interpolate(img_save, scale_factor=0.5)
                img_save = img_save * 255
                # Process img_copy and do not destroy the data of img
                img_copy = img_save.clone().data.permute(0, 2, 3, 1)[0, :, :, :].cpu().numpy()
                #img_copy = np.clip(img_copy, 0, 255)
                img_copy = img_copy.astype(np.uint8)
                save_img_name = 'sample_batch' + str(batch_idx+1) + '.png'
                save_img_path = os.path.join(sample_folder, save_img_name)
                img_copy = cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR)
                cv2.imwrite(save_img_path, img_copy)
            if (batch_idx + 1) % 5000 == 0:
                save_model(generator, epoch, opt, batch_idx+1)
                save_model(discriminator, epoch, opt, batch_idx+1, is_D=True)


        #Learning rate decrease
        adjust_learning_rate(opt.lr_g, optimizer_g, (epoch + 1), opt)
        adjust_learning_rate(opt.lr_d, optimizer_d, (epoch + 1), opt)

        # Save the model
        save_model(generator, epoch, opt)
        save_model(discriminator, epoch , opt, is_D=True)

        ### Sample data every epoch
        # if (epoch + 1) % 1 == 0:
        #     masked_img = img * (1 - mask) + mask
        #     img_save = torch.cat((img, masked_img, first_out, second_out, first_out_wholeimg, second_out_wholeimg), 3)
        #     img_save = img_save * 255
        #     img_copy = img_save.clone().data.permute(0, 2, 3, 1)[0, :, :, :].cpu().numpy()
        #     img_copy = np.clip(img_copy, 0, 255)
        #     img_copy = img_copy.astype(np.uint8)
        #     # Save to certain path
        #     save_img_name = 'epoch' + str(epoch + 1) + 'sample' + '.png'
        #     save_img_path = os.path.join(sample_folder, save_img_name)
        #     img_copy = cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR)
        #     cv2.imwrite(save_img_path, img_copy)

