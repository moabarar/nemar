# NeMAR - Neural Multimodal Adversarial Registration
This is the official implementation of:<br>
Unsupervised Multi-Modal Image Registration via Geometry Preserving Image-to-Image Translation.
### [\[Paper\]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Arar_Unsupervised_Multi-Modal_Image_Registration_via_Geometry_Preserving_Image-to-Image_Translation_CVPR_2020_paper.pdf)   [\[Supplemental Materials\]](http://openaccess.thecvf.com/content_CVPR_2020/supplemental/Arar_Unsupervised_Multi-Modal_Image_CVPR_2020_supplemental.pdf)

<p align="center">
    <img src='https://raw.githubusercontent.com/moabarar/nemar/nemar_deploy/teaser.gif' alt='missing' />
    <br>Registration output during 50 training epochs 
</p>

## Getting started
This repository is based on the implementation of [Pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). We recommened getting yourself familiar with the former framework. Here we provide a basic guidline on how to use our code.

---

### (Step 1) Preparing your data
You need to implement your own dataset. The dataset should implement the template proivded by in [base_dataset.py](data/base_dataset.py). Invoking '__getitem__' of your dataset should return the following template dictionay:<br>

    {'A': tensor_image_modality_a,
     'B': tensor_image_modality_b}
 
Where tensor_image_modality_[a/b] is the tensor of the image from modality A and modality B respectively.<br>
The name convention use is [whatever-name-you-want]_dataset.py - this is important in order to be able to define your dataset using strings.

---
 
### (Step 2) Train NeMAR model
Here we provide a list of flags that could be used during the training of our model. The flags are categorized into 
STN related flags and general training flags<br>
* --stn_type: train an affine based registration ('affine') or a deformation field based registration network ('unet').
* --stn_cfg: you can define the network architecture via a configuration string (default is 'A').
 See example configuration in [unet_stn.py](models/stn/unet_stn.py) and in [affine_stn.py](models/stn/affine_stn.py).
* --stn_no_identity_init: set if you WANT to start with a random transformation.
* --stn_bilateral_alpha: the alpha value used in the bilateral filtering term (see paper).

Training related flags (beside the base flags used by Pix2Pix/CycleGAN):
* --multi_res: you can train NeMAR with multi-resoultion discriminators (similar to [pix2pixHD](https://arxiv.org/pdf/1711.11585.pdf)). 
We believe this could be use-full when working with high-resolution images.
* --lambda_smooth: the lambda used for the regularization term of the stn.
* --lambda_recon: the lambda used for the reconstruction loss.
* --lambda_GAN: the lambda used for the GAN loss

---

## Enabling Tensorboard
We provide an option to write training stats using tensorboard. To enable tensorboard visualizer, you need to set the flag --enable_tbvis.
This will create a tensboard log-file in the directory "<checkpoints_dir>/<exp_name>/<exp_name>_tensorboard_logs". The tensorboard visualizer
class reports (1) mean transformation offsets in x,y direction, (2) network weights, (3) losses. These values are written to tensorboard logfile each epoch. 
The following flags can be used when tbvis is enabled<br>
* --tbvis_iteration_update_rate if you want to write in iteration resolution set --tbvis_iteration_update_rate to positive number.
* --tbvis_disable_report_offsets: set if you don't want to save the mean transformation offsets.
* --tbvis_disable_report_weights: set if you don't want to save the network weights. 


---

### Citation
If you use this repository - please cite:

    @InProceedings{Arar_2020_CVPR,
    author = {Arar, Moab and Ginger, Yiftach and Danon, Dov and Bermano, Amit H. and Cohen-Or, Daniel},
    title = {Unsupervised Multi-Modal Image Registration via Geometry Preserving Image-to-Image Translation},
    booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2020}
    }

Since this repository is based on Pix2Pix & CycleGan framework - make sure you cite these two awesome papers:

    @inproceedings{CycleGAN2017,
     title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networkss},
     author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
     booktitle={Computer Vision (ICCV), 2017 IEEE International Conference on},
     year={2017}
     }


    @inproceedings{isola2017image,
     title={Image-to-Image Translation with Conditional Adversarial Networks},
     author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
     booktitle={Computer Vision and Pattern Recognition (CVPR), 2017 IEEE Conference on},
     year={2017}
    }
   
