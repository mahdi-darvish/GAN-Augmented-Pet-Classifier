# GANs Augmented Pet Classifier
<div style="text-align:center"><img src="Figures/training_sequence.gif" /></div>

###Towards Fine-grained Image Classification withGenerative Adversarial Networks and FacialLandmark Detection
Mahdi Darvish, Mahsa Pouramini, Hamid Bahador

arixv ;llinkkk

Abstract: *Fine-grained   classification   remains   a   challengingtask  because  distinguishing  categories  needs  learning  complexand  local  differences.  Diversity  in  the  pose,  scale,  and  positionof  objects  in  an  image  makes  the  problem  even  more  difficult.Although  the  recent  Vision  Transformer  models  achieve  highperformance,  they  need  an  extensive  volume  of  input  data.  Toencounter this problem, we made the best use of GAN-based dataaugmentation  to  generate  extra  dataset  instances.  Oxford-IIITPets  was  our  dataset  of  choice  for  this  experiment.  It  consistsof  37  breeds  of  cats  and  dogs  with  variations  in  scale,  poses,and  lighting,  which  intensifies  the  difficulty  of  the  classificationtask.  Furthermore,  we  enhanced  the  performance  of  the  recentGenerative Adversarial Network (GAN), StyleGAN2-ADA modelto generate more realistic images while preventing overfitting tothe  training  set.  We  did  this  by  training  a  customized  versionof  MobileNetV2  to  predict  animal  facial  landmarks;  then,  wecropped  images  accordingly.  Lastly,  we  combined  the  syntheticimages  with  the  original  dataset  and  compared  our  proposedmethod with standard GANs augmentation and no augmentationwith  different  subsets  of  training  data.  We  validated  our  workby  evaluating  the  accuracy  of  fine-grained  image  classificationon the recent Vision Transformer (ViT) Model. *

# Results


# Pre-Trained Models
# Getting started
## Dataset
## StyleGAN2-ADA Installation
### Running Localy
#### Data Preperation
#### Training
#### Generating Images with Pre-trained Models
 ### Running through Google Colab
## Landmark Detection
# Citation

# Acknowledgement
