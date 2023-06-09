# DeepSimNets

Official repository for **_DeepSim-Nets: Deep Similarity Networks for Stereo Image Matching_** [paper :page_facing_up:](https://openaccess.thecvf.com/content/CVPR2023W/EarthVision/papers/Chebbi_DeepSim-Nets_Deep_Similarity_Networks_for_Stereo_Image_Matching_CVPRW_2023_paper.pdf) accepted for EarthVision2023.

The paper relative code will be uploaded soon ! 

<p align="center">
  <img width="900" height="200" src="https://user-images.githubusercontent.com/28929267/230094222-a7dc3348-3474-47cc-9074-cbbb68605f4e.png">
 </p>
<p align="center">
Overall training pipeline
</p>


<p align="center">
  <img width="900" height="400" src="https://user-images.githubusercontent.com/28929267/230093358-41c5f835-079d-4ead-9727-f3e8f927ebb3.png">  
 </p>
  <em> &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp  &nbsp  &nbsp &nbsp &nbsp   Epipolar &nbsp &nbsp  &nbsp  &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp  Our MS-AFF &nbsp  &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp  PSMNet &nbsp &nbsp  &nbsp  &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp &nbsp Normalized Cross Correlation</em>&nbsp &nbsp 
         
   
  
  
*We propose to learn dense similarities by training three multi-scale learning architectures on wider images tiles. To enable robust self-supervised contrastive learning, a sample mining is developed. Our main idea lies in relying on wider suppport regions to leverage pixel-level similarity-aware embeddings. Then, the whole set of pixel embeddings of a reference image are matched to their corresponding ones at once. Our approach alleviates the block matching distinctiveness shotcomings by exploiting the image wider context. We therefore leverage quite distinctive similarity measures that outcome standard hand-crafted correlation (NCC) and deep learning patch based approaches (MC-CNN). Compared to end-to-end methods, our DeepSim-Nets are highly versatile and readily suited for standard mutli resolution and large scale stereo matching pipelines.* 

# Multi-Scale Attentional Feature Fusion (MS-AFF) 
*We additionally propose a lightweight architecture baptized MS-AFF where inputs are 4 multi-scale or multi-resolution tiles as highlighted below. The generated multi-scale features are iteratively fused based on an adpated attention mechanism from [Attentional Feature Fusion](https://openaccess.thecvf.com/content/WACV2021/papers/Dai_Attentional_Feature_Fusion_WACV_2021_paper.pdf). Here is the architecture together with the multi-scale attention module.*

<p align="center">
  <img width="600" height="200" src="https://user-images.githubusercontent.com/28929267/230161014-aa12a227-52d4-4bbd-93f6-5db6428c9eb5.png">
 </p>


# Training 

DeepSim-Nets are trained on Aerial data from Dublin dataset on 4 GPUs. The following summarizes the training environment:
- Ubuntu 18.04.6 LTS/CentOS Linux release 7.9.2009
- Python 3.9.12 
- PyTorch 1.11.0
- pytorch_lightning 1.6.3
- CUDA 10.2, 11.2 and 11.4
- NVIDIA V100 32G/ NVIDIA A100 40G
- 64G RAM

# Evaluation 

To evaluate our classifiers performance, we estimate joint distributions of matching and non-similarity random variables on test data. Details will be given soon.

# Inference

## Models

After training, models are scripted and arranged so that similarities could be computed by:
- normalized dot product between embeddings : This relies on feature extractor ouptput feature maps. 
- learned similarity function from the MLP decision network (feature extractor+ MLP).

| Model name | Dataset | Joint_Probability(JP)  | :floppy_disk: | :point_down: |
|---|:---:|:---:|:---:|:---:|
| MS-AFF feature |  Dublin/Vaihingen/Enschede | -- | 4 M | [link](-)  |
| MS-AFF MLP |  Dublin/Vaihingen/Enschede | 89.6 | 1,4 M | [link](-)  |
| Unet32  |  Dublin/Vaihingen/Enschede | -- | 31,4 M | [link](-)  |
| Unet32 MLP |  Dublin/Vaihingen/Enschede | 88.6 | 1,4 M | [link](-)  |
| Unet Attention  |  Dublin/Vaihingen/Enschede | -- | 38,1 M | [link](-)  |
| Unet Attention MLP |  Dublin/Vaihingen/Enschede | 88.9 | 1,4 M | [link](-)  |

Inference requires an SGM implementation for cost volume regularization. Our similarty models are scripted and fed to our C++ implementation under the [![MicMac](<img src="https://user-images.githubusercontent.com/28929267/230158064-57c90a2a-e906-4d72-b238-1d168f0cca58.png" width="50" height="10">)](https://github.com/micmacIGN/micmac) photogrammetry software. The main C++ production code is located at *MMVII/src/LearningMatching*.
Our approach is embedded into the MicMac multi-resolution image matching pipeline and can be parametrized using a MicMac compliant xml file. The figure below illustrates 
![image](https://user-images.githubusercontent.com/28929267/230213458-4b43d162-2259-4808-8e4a-d66657473ad7.png)


For real world stereo matching on larger images, a jupyter notebook will be provided ! 


# Docker Image 

Alternatively, a docker image will be prepared with a precompiled version within the MicMac Open Source photogrammetry software. 
