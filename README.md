# Capsule-Expert Routing UNet: A Hybrid 2.5D Convolution-Attention Architecture with Mixture-of-Experts for Efficient 3D Medical Segmentation
![](https://i.imgur.com/waxVImv.png)


<hr />

![main figure](media/intro_fig.png)
> **Abstract:** *Recent advances in 3D medical image segmentation have been driven by hybrid CNN-Transformer
architectures that capture long-range dependencies at the cost of heavy parameters and latency.
This paper introduces Capsule-Expert Routing UNet (CER-UNet), a novel encoder–decoder model
that achieves strong global context modeling with substantially lower computational params. CER-
UNet integrates two complementary contributions: (1) a statistical attention module that performs
computationally efficient long-range interaction via low-rank covariance pooling and channel-wise
statistics, coupled with a 2.5D hybrid convolutional design featuring Inception-style multi-scale
depthwise-separable kernels, (2) a Capsule-Expert Mixture-of-Experts (CapMoE) mechanism that
introduces dynamic feature routing across hierarchical scales, enabling lightweight multi-scale
fusion and expert specialization while avoiding the instability of full attention-based routers. CER-
UNet preserves the strong context modeling of recent UNet-like CNN-Transformer hybrids but
surpasses them in accuracy-efficiency balance. On ACDC, CER-UNet achieves 92.52% average
Dice, and on Synapse, it attains 86.64% Dice with only 33M parameters, outperforming competitive
Transformer baselines and conventional 2D/2.5D segmentation networks. Extensive experiments
across multiple 3D medical segmentation benchmarks demonstrate that CER-UNet delivers robust
state-of-the-art performance with significantly lower trainable params. * 
<hr />


![Architecture overview](media/UNETR++_Block_Diagram.jpg)

<hr />



### ACDC Dataset
Qualitative comparison on the ACDC dataset. We compare our UNETR++ with existing methods: UNETR and nnFormer. It is noticeable that the existing methods struggle to correctly segment different organs (marked in red dashed box). Our UNETR++ achieves favorable segmentation performance by accurately segmenting the organs.  Our UNETR++ achieves promising segmentation performance by accurately segmenting the organs.
![ACDC Qual Results](media/acdc_vs_unetr_suppl.jpg)


<hr />




## Dataset
We follow the same dataset preprocessing as in [nnFormer](https://github.com/282857341/nnFormer). We conducted extensive experiments on five benchmarks: Synapse, BTCV, ACDC, BRaTs, and Decathlon-Lung. 

The dataset folders for Synapse should be organized as follows: 

```
./DATASET_Synapse/
  ├── unetr_pp_raw/
      ├── unetr_pp_raw_data/
           ├── Task02_Synapse/
              ├── imagesTr/
              ├── imagesTs/
              ├── labelsTr/
              ├── labelsTs/
              ├── dataset.json
           ├── Task002_Synapse
       ├── unetr_pp_cropped_data/
           ├── Task002_Synapse
 ```
 
 The dataset folders for ACDC should be organized as follows: 

```
./DATASET_Acdc/
  ├── unetr_pp_raw/
      ├── unetr_pp_raw_data/
           ├── Task01_ACDC/
              ├── imagesTr/
              ├── imagesTs/
              ├── labelsTr/
              ├── labelsTs/
              ├── dataset.json
           ├── Task001_ACDC
       ├── unetr_pp_cropped_data/
           ├── Task001_ACDC
 ```
 

Please refer to [Setting up the datasets](https://github.com/282857341/nnFormer) on nnFormer repository for more details.

## Training
The following scripts can be used for training our UNETR++ model on the datasets:
```shell
bash training_scripts/run_training_synapse.sh
bash training_scripts/run_training_acdc.sh
bash training_scripts/run_training_lung.sh
bash training_scripts/run_training_tumor.sh
```

<hr />


## Acknowledgement
This repository is built based on [nnFormer](https://github.com/282857341/nnFormer) repository.



## Contact
Should you have any question, please create an issue on this repository or contact me at abdelrahman.youssief@mbzuai.ac.ae.
