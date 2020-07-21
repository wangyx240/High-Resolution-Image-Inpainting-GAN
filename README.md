# High Resolution Image Inpainting 
Unofficial Pytorch Re-implementation of "<a href="https://arxiv.org/abs/2005.09704">Contextual Residual Aggregation for Ultra High-Resolution Image Inpainting</a>"(CVPR 2020 Oral).

The code is based on implementation of <a href="https://github.com/zhaoyuzhi/deepfillv2">deepfillv2</a>. Thanks for the great job.

The project is still in progress, please feel free to contact me if there is any problem.

## Implementation
Besides Contextual Residual Aggregation(CRA) and Light-Weight GatedConvolution in the paper, also add Residual network structure, SN-PatchGAN in this project.
### Preparing
Before running, please ensure the environment is `Python 3.6` and `PyTorch 1.2.0`.

Dataset: <a href="http://places2.csail.mit.edu/download.html">Places365-Standard</a>

### Training
```bash
python train.py     --epochs 40
                    --lr_g 0.0001
                    --batch_size 4
                    --lambda_perceptual 100
                    --lambda_l1 300 [feel free to change during training]
                    --baseroot [the path of training set]
                    --mask_type 'free_form' [or 'single_bbox' or 'bbox']
                    --imgsize 512
```
```bash
if you have more than one GPU, please change following codes:
python train.py     --multi_gpu True
                    --gpu_ids [the ids of your multi-GPUs]
```
Default training process uses hinge loss as the D_loss, also provide Wgan-GP in the code.

Modify train.py to change other training parameters. Modify dataset.py to change the size of free-form mask.

### Testing
To do
### Pre-trained model
Still in training
### Sample images
To do
### Acknowledgement

```bash
@misc{yi2020contextual,
    title={Contextual Residual Aggregation for Ultra High-Resolution Image Inpainting},
    author={Zili Yi and Qiang Tang and Shekoofeh Azizi and Daesik Jang and Zhan Xu},
    year={2020},
    eprint={2005.09704},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
@inproceedings{yu2019free,
  title={Free-form image inpainting with gated convolution},
  author={Yu, Jiahui and Lin, Zhe and Yang, Jimei and Shen, Xiaohui and Lu, Xin and Huang, Thomas S},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={4471--4480},
  year={2019}
}
```
