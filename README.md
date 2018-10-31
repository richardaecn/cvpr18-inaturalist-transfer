# Large Scale Fine-Grained Categorization and Domain-Specific Transfer Learning

Tensorflow **code and models** for the paper:

[*Large Scale Fine-Grained Categorization  and Domain-Specific Transfer Learning*](https://arxiv.org/abs/1806.06193)\
[Yin Cui](http://www.cs.cornell.edu/~ycui/), [Yang Song](https://ai.google/research/people/author38270), [Chen Sun](http://chensun.me/), Andrew Howard, [Serge Belongie](http://blogs.cornell.edu/techfaculty/serge-belongie/)\
CVPR 2018

This repository contains code and pre-trained models used in the paper and 2 demos to demonstrate: 1) the importance of pre-training data on transfer learning; 2) how to calculate domain similarity between source domain and target domain.

Notice that we used a mini validation set (./inat_minival.txt) contains 9,697 images that are randomly selected from the original iNaturalist 2017 validation set. The rest of valdiation images were combined with the original training set to train our model in the paper. There are 665,473 training images in total.


## Dependencies:
+ Python (3.5)
+ Tensorflow (1.11)
+ [pyemd](https://pypi.org/project/pyemd/)
+ [scikit-learn](http://scikit-learn.org/stable/)
+ [scikit-image](https://scikit-image.org/)


## Preparation:
+ Clone the repo with recursive
```bash
git clone --recursive https://github.com/richardaecn/cvpr18-inaturalist-transfer.git
```
+ Install dependencies. Please refer to TensorFlow, pyemd, scikit-learn and scikit-image official websites for installation guide.
+ Download [data](https://drive.google.com/file/d/1-FJlSj0Qa8pYvp_5PbMKw4piu7Qt5qfE/) and [feature](https://drive.google.com/file/d/1vOHKuqt7XgROo9t5cblJvGf0kGpiFaF1/) and unzip them into the same directory as the cloned repo. You should have two folders './data' and './feature' in the repo's directory.


## Datasets (optional):
In the paper, we used data from 9 publicly available datasets:
+ [ImageNet (ILSVRC2012)](http://www.image-net.org/challenges/LSVRC/2012/)
+ [iNaturalist 2017](https://github.com/visipedia/inat_comp/tree/master/2017)
+ [Aircraft](http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/)
+ [CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)
+ [Oxford Flower 102](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
+ [Food 101](https://www.vision.ee.ethz.ch/datasets_extra/food-101/)
+ [NABirds](http://dl.allaboutbirds.org/nabirds)
+ [Stanford Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)
+ [Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/)

We provide [a download link](https://drive.google.com/file/d/1-FJlSj0Qa8pYvp_5PbMKw4piu7Qt5qfE/) that includes the entire CUB-200-2011 dataset and data splits for the rest of 8 datasets. The provided link contains sufficient data for this repo. If you would like to use other 8 datasets, please download them from the official websites and put them in the corresponding subfolders under './data'.


## Pre-trained Models (optional):
The models were trained using [TensorFlow-Slim](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim). We implemented [Squeeze-and-Excitation Networks (SENet)](https://arxiv.org/abs/1709.01507) under './slim'. The pre-trained models can be downloaded from the following links:

| Network                | Pre-trained Data    | Input Size      | Download Link |
|------------------------|---------------------|-----------------|---------------|
| Inception-V3           | ImageNet            | 299             |[link](https://drive.google.com/open?id=1Djydji-QnJOQ93dWYw-4yVLSP4-TAXHy)
| Inception-V3           | iNat2017            | 299             |[link](https://drive.google.com/open?id=1g3bsmBrKPRbah4EDNnC_IgDk8N4uAZaY)
| Inception-V3           | iNat2017            | 448             |[link](https://drive.google.com/open?id=1RWr2I1mOV6l5VNIZrIhwnAMtggfMaeMf)
| Inception-V3           | iNat2017            | 299 -> 560 FT<sup>1</sup> |[link](https://drive.google.com/open?id=1ocJALCYcsM6Ym2DWV-B6GYmUHVshd8lF)
| Inception-V3           | ImageNet + iNat2017 | 299             |[link](https://drive.google.com/open?id=1EUNR4o77lNt0fN5Bi4lKxTZnFhghILRw)
| Inception-V3 SE        | ImageNet + iNat2017 | 299             |[link](https://drive.google.com/open?id=11T9ogOdoG0Qu2rBNKQ1hNcHq3rAQ1F-Q)
| Inception-V4           | iNat2017            | 448             |[link](https://drive.google.com/open?id=1zV0n1qAQ9rDoNMHvRP488mes018BOqs0)
| Inception-V4           | iNat2017            | 448 -> 560 FT<sup>2</sup> |[link](https://drive.google.com/open?id=13IkVGfjxQKMpN6OFHpqLoIiM92FMJ-o8)
| Inception-ResNet-V2    | ImageNet + iNat2017 | 299             |[link](https://drive.google.com/open?id=1l-7ZidrlG8UeS8E21tcWzke5JmaTR7mF)
| Inception-ResNet-V2 SE | ImageNet + iNat2017 | 299             |[link](https://drive.google.com/open?id=1XvF8rDCJEeOMowy6JlRaNDOUD51KajpR)
| ResNet-V2 50           | ImageNet + iNat2017 | 299             |[link](https://drive.google.com/open?id=1F9e5kFhwthf-h7CSMKW3aatPcRY0o4qs)
| ResNet-V2 101          | ImageNet + iNat2017 | 299             |[link](https://drive.google.com/open?id=1eoOfsMRdgrLxkoEw0e3Za0aPvSBzkCE2)
| ResNet-V2 152          | ImageNet + iNat2017 | 299             |[link](https://drive.google.com/open?id=1VvMGQIDNsxs1H9lnfTWXiikbcF1EOQ7N)

<sup>1</sup> This model was trained with 299 input size on train + 90% val and then fine-tuned with 560 input size on 90% val.

<sup>2</sup> This model was trained with 448 input size on train + 90% val and then fine-tuned with 560 input size on 90% val.

[TensorFlow Hub](https://www.tensorflow.org/hub/) also provides a pre-trained Inception-V3 299 on iNat2017 original training set [here](https://tfhub.dev/google/inaturalist/inception_v3/feature_vector/1).


## Featrue Extraction (optional):
Run the following Python script to extract feature:
```
python feature_extraction.py
```
To run this script, you need to download the checkpoint of [Inception-V3 299 trained on iNat2017](https://drive.google.com/open?id=1g3bsmBrKPRbah4EDNnC_IgDk8N4uAZaY). The dataset and pre-trained model can be modified in the script.

We provide [a download link](https://drive.google.com/file/d/1vOHKuqt7XgROo9t5cblJvGf0kGpiFaF1/) that includes features used in the domos of this repo.


## Demos
1. Linear logistic regression on extracted features:

This demo shows the importance of pre-training data on transfer learning. Based on features extracted from an Inception-V3 pre-trained on iNat2017, we are able to achieve **89.9%** classification accuracy on CUB-200-2011 with the simple logistic regression, outperforming most state-of-the-art methods.
```
LinearClassifierDemo.ipynb
```

2. Calculating domain similarity by Earth Mover's Distance (EMD):
This demo gives an example to calculate the domain similarity proposed in the paper. Results correspond to part of the Fig. 5 in the original paper.
```
DomainSimilarityDemo.ipynb
```


## Citation
If you find our work helpful in your research, please cite it as:
```latex
@inproceedings{Cui2018iNatTransfer,
  title = {Large Scale Fine-Grained Categorization and Domain-Specific Transfer Learning},
  author = {Yin Cui, Yang Song, Chen Sun, Andrew Howard, Serge Belongie},
  booktitle={CVPR},
  year={2018}
}
```
