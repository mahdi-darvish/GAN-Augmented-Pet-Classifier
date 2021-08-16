# GANs Augmented Pet Classifier
<div style="text-align:center"><img src="Figures/training_sequence.gif" /></div>

###Towards Fine-grained Image Classification withGenerative Adversarial Networks and FacialLandmark Detection
Mahdi Darvish, Mahsa Pouramini, Hamid Bahador

arixv ;llinkkk

Abstract: *Fine-grained   classification   remains   a   challengingtask  because  distinguishing  categories  needs  learning  complexand  local  differences.  Diversity  in  the  pose,  scale,  and  positionof  objects  in  an  image  makes  the  problem  even  more  difficult.Although  the  recent  Vision  Transformer  models  achieve  highperformance,  they  need  an  extensive  volume  of  input  data.  Toencounter this problem, we made the best use of GAN-based dataaugmentation  to  generate  extra  dataset  instances.  Oxford-IIITPets  was  our  dataset  of  choice  for  this  experiment.  It  consistsof  37  breeds  of  cats  and  dogs  with  variations  in  scale,  poses,and  lighting,  which  intensifies  the  difficulty  of  the  classificationtask.  Furthermore,  we  enhanced  the  performance  of  the  recentGenerative Adversarial Network (GAN), StyleGAN2-ADA modelto generate more realistic images while preventing overfitting tothe  training  set.  We  did  this  by  training  a  customized  versionof  MobileNetV2  to  predict  animal  facial  landmarks;  then,  wecropped  images  accordingly.  Lastly,  we  combined  the  syntheticimages  with  the  original  dataset  and  compared  our  proposedmethod with standard GANs augmentation and no augmentationwith  different  subsets  of  training  data.  We  validated  our  workby  evaluating  the  accuracy  of  fine-grained  image  classificationon the recent Vision Transformer (ViT) Model. *

# Results

the evaluated RMSE of the trained MobileNetV2 model with
and without landmark normalization:

<div style="text-align:center"><img src="Figures/RMSE.PNG" /></div>

The measured accuracy of the used model and FID for three different dataset conditions (Original, augmented, and augmented-cropped) in data regimes of 10, 50, and 100 percent:

<div style="text-align:center"><img src="Figures/FID.PNG" /></div>

Comparison between synthetic and authentic images. This figure show (a) the original data,(b) and (c) generated images on
the whole dataset, cropped and uncropped, respectively. (d) cropped images on 50%, (e) uncropped images generated on 50%
subset and finally (f) and (g), cropped and uncropped images result of training on only 10% of the data. These qualitative
visualizations prove the effectiveness and the interpretability of the method.

<div style="text-align:center"><img src="Figures/result's pic.PNG" /></div>

Finally, the charts explain the accuracy of the used model and FID for three different dataset conditions (Original, augmented, and cropped-augmented ) in data regimes of 10, 50, and 100 percent:

<div style="text-align:center"><img src="Figures/charts.PNG" /></div>

# Pre-Trained Models

### StyleGAN2-ADA trained on cropped pets dataset 



| Subset | Kimg | FID  | Acc on Vit | Model link | TFRecords |
|--------|------|------|------------|------------|-----------|
| 10%    | 5120 | 49.4 | 68.55      | 250 Mb     |           |
| 50%    | 5120 | 22.3 | 91.73      | 250 Mb     |           |
| 100%   | 5120 | 14.1 | 96.28      | 250 Mb     |           |


### StyleGAN2-ADA trained on not cropped pets dataset 

| Subset | Kimg | FID  | Acc on Vit | Model link | TFRecords |
|--------|------|------|------------|------------|-----------|
| 10%    | 5120 | 71.1 | 63.32      | 250 Mb     |           |
| 50%    | 5120 | 36.4 | 88.70      | 250 Mb     |           |
| 100%   | 5120 | 20.7 | 94.93      | 250 Mb     |           |

# Getting started
## Dataset

The official dataset can be reached from:

[Oxford-IIIT Pet dataset.](https://www.robots.ox.ac.uk/~vgg/data/pets/)

The cropped dataset iis given in:

[GDRIVE.](https://drive.google.com/drive/u/7/my-drive)

## StyleGAN2-ADA Installation
### Running Localy
#### Data Preperation
#### Training
#### Generating Images with Pre-trained Models
 ### Running through Google Colab
## Landmark Detection
# Citation

# Acknowledgement
