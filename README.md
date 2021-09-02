# HGM

**Paper**: High-dimensional Assisted Generative Model for Color Image Restoration


**Authors**: Kai Hong, Chunhua Wu, Cailian Yang, Minghui Zhang, Yancheng Lu, Yuhao Wang, Qiegen Liu


Date : 8/2021
Version : 1.0   
The code and the algorithm are for non-comercial use only. 
Copyright 2021, Department of Electronic Information Engineering, Nanchang University.  

This work presents an unsupervised deep learning scheme that exploiting high-dimensional assisted score-based generative model for color image restoration tasks. Con-sidering that the sample number and internal dimension in score-based generative model have key influence on estimating the gradients of data distribution, two differ-ent high-dimensional ways are proposed: The chan-nel-copy transformation increases the sample number and the pixel-scale transformation decreases feasible space dimension. Subsequently, a set of high-dimensional tensors represented by these transformations are used to train the network through denoising score matching. Then, sampling is performed by annealing Langevin dy-namics and alternative data-consistency update. Fur-thermore, to alleviate the difficulty of learning high-dimensional representation, a progressive strategy is proposed to leverage the performance. The proposed unsupervised learning and iterative restoration algo-rithm, which involves a pre-trained generative network to obtain prior, has transparent and clear interpretation compared to other data-driven approaches. Experimental results on demosaicking and inpainting conveyed the re-markable performance and diversity of our proposed method.
if you want to train the code, please train the code to attain the model

```bash 
demoscaicking    python3.5 separate_ImageNet.py --model ncsn --runner Anneal_runner_train_6ch_demoscaicking/Anneal_runner_train_12ch_demoscaicking --config anneal.yml --doc your-save-path
inpainting       python3.5 separate_ImageNet.py --model ncsn --runner train_6ch_rgbrgb/train_12ch_pool --config anneal.yml --doc your-save-path
```

## Test
if you want to test the code, please 

```bash
not-p
python3.5 separate_ImageNet.py --model ncsn --runner Test_6ch_inpainting_rgbrgb_256 --config anneal_bedroom_6ch_256.yml --doc your-checkpoint --test --image_folder your-save-path

or
python3.5 separate_ImageNet.py --model ncsn --runner Test_3ch_inpainting_rgbrgb_256 --config anneal_bedroom_3ch_256.yml --doc your-checkpoint --test --image_folder your-save-path
python3.5 separate_ImageNet.py --model ncsn --runner Test_6ch_inpainting_rgbrgb_256 --config anneal_bedroom_6ch_256.yml --doc your-checkpoint --test --image_folder your-save-path
....
```
key number is "HGM " 

## Graphical representation
## Visualization of the HGM.
 <div align="center"><img src="https://github.com/yqx7150/HGM/blob/main/img/1.jpg" >  </div>
The pipeline of HGM. The blue box is the training process for prior learning, the yellow box is the procedure for IR, and the green box is the detailed structure diagram of HGM. In the training pass, we learn the score of perturbed low-resolution high-dimensional images via  transformation and add a series of noises . At the stage of restoration, the Gaussian noise is gradually approximated to the real score and is sampled by annealed Langevin dynamics. The DC is the Lagrangian term to enforce the mixture constraint.
 
 ##  Visualization of the representation error bound and the relationship between sample number and dimension 
 <div align="center"><img src="https://github.com/yqx7150/JGM/blob/main/image/2.jpg" >  </div>
 Visualization of the representation error bound and the relationship between sample number.
 Visualization of the representation error bound and the relationship between sample dimension.

 
## Train Data
inpainting: We choose LSUN(bedroom and church) dataset for experiments
demosaicking :  We choose three MSR dataset for experiments
## Test Data
inpainting: We randomly select 100 bedrooms dataset, the size is 256x256.
demosaicking: We select Kodak dataset and McM dataset respectively, the size is 500x500.

### Other Related Projects

  * Homotopic Gradients of Generative Density Priors for MR Image Reconstruction  
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/abstract/document/9435335)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/HGGDP)   [<font size=5>**[PPT]**</font>](https://github.com/yqx7150/HGGDP/tree/master/Slide)   [<font size=5>**[数学图像联盟会议交流PPT]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/Slide)

 * Iterative Reconstruction for Low-Dose CT using Deep Gradient Priors of Generative Model  
[<font size=5>**[Paper]**</font>](https://arxiv.org/abs/2009.12760)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/EASEL)   [<font size=5>**[PPT]**</font>](https://github.com/yqx7150/HGGDP/tree/master/Slide)

 * Progressive Colorization via Interative Generative Models  
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/document/9258392)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/iGM)   [<font size=5>**[PPT]**</font>](https://github.com/yqx7150/HGGDP/tree/master/Slide)   [<font size=5>**[数学图像联盟会议交流PPT]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/Slide)


 * Wavelet Transform-assisted Adaptive Generative Modeling for Colorization
[<font size=5>**[Paper]**</font>](https://arxiv.org/abs/2107.04261)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/WACM)   [<font size=5>**[数学图像联盟会议交流PPT]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/Slide)
