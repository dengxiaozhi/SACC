# Strongly Augmented Contrastive Clustering (SACC)

By Xiao-Zhi Deng, Dong Huang, Ding-Hua Chen

This is a Pytorch implementation of the paper.

![network](.\figures\network.png)

## Performance

The representation encoder of the proposed SACC is ResNet34.

|    Dataset    | NMI  | ACC  | ARI  |
| :-----------: | :--: | :--: | :--: |
|   CIFAR-10    | 76.5 | 85.1 | 72.4 |
|   CIFAR-100   | 44.8 | 44.3 | 28.2 |
|    STL-10     | 69.1 | 75.9 | 62.6 |
|  ImageNet-10  | 87.7 | 90.5 | 84.3 |
| ImageNet-dogs | 45.5 | 43.7 | 28.5 |

## Dependency

- python>=3.7
- pytorch>=1.6.0
- torchvision>=0.8.1
- munkres>=1.1.4
- numpy>=1.19.2
- opencv-python>=4.4.0.46
- pyyaml>=5.3.1
- scikit-learn>=0.23.2
- cudatoolkit>=11.0

## Configuration

There is a configuration file "config/config.yaml", where one can edit both the training and test options.

## Acknowledgment for reference repos

- [CC](https://github.com/Yunfan-Li/Contrastive-Clustering)

- [CLSA](https://github.com/maple-research-lab/CLSA)