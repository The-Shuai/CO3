# Collaborative Consortium of Foundation Models for Open-World Few-Shot Learning
This paper has been accepted by **AAAI 2024**. We are profoundly grateful for the significant insights provided by [CaFo](https://arxiv.org/pdf/2303.02151.pdf).

## Abstract
Open-World Few-Shot Learning (OFSL) is a crucial research field dedicated to accurately identifying target samples in scenarios where data is limited and labels are unreliable. 
This research holds significant practical implications and is highly relevant to real-world applications. 
Recently, the advancements in foundation models like CLIP and DINO have showcased their robust representation capabilities even in resource-constrained settings with scarce data. 
This realization has brought about a transformative shift in focus, moving away from “building models from scratch” towards “effectively harnessing the potential of foundation models to extract pertinent prior knowledge suitable for OFSL and utilizing it sensibly”. 
Motivated by this perspective, we introduce the **C**ollaborative C**o**nsortium of F**o**undation M**o**dels (**CO3**), which leverages CLIP, DINO, GPT-3, and DALLE to collectively address the OFSL problem. CO3 comprises four key blocks: 
(1) the Label Correction Block (LC-Block) corrects unreliable labels, (2) the Data Augmentation Block (DA-Block) enhances available data, 
(3) the Feature Extraction Block (FE-Block) extracts multi-modal features, and (4) the Text-guided Fusion Adapter (TeFu-Adapter) integrates multiple features while mitigating the impact of noisy labels through semantic constraints. 
Only the adapter’s parameters are adjustable, while the others remain frozen.
Through collaboration among these foundation models, CO3 effectively unlocks their potential and unifies their capabilities to achieve state-of-the-art performance on 11 datasets.

## Get Started
1. Create a conda environment and install dependencies.
```
pip install -r requirements.txt
```
2. Download the "cache" folder and place it in the root directory.
3. Download the DINO pre-trained model and place it in the "dino" directory.   
   e.g., "./dino/dino_resnet50_pretrain.pth".
4. Download the datasets.
5. Modify the ```main_path``` in the [main.py](https://github.com/The-Shuai/CO3/blob/main/main.py) file on line 22 to match the dataset you intend to validate.      
   e.g., set the ```main_path``` to ```main_path = "./configs/imagenet/config.yaml"```
6. Modify the ```root_path``` on the 2nd line of the ```config.yaml``` file corresponding to your dataset.    
   e.g., within the ```./configs/imagenet/config.yaml``` file, update the ```root_path``` to ```root_path: "./DATA/"```
7. Run
```
CUDA_VISIBLE_DEVICES=0 python main.py
```

## Acknowledgement
This repo benefits from [CaFo](https://github.com/OpenGVLab/CaFo?tab=readme-ov-file), [CLIP](https://github.com/openai/CLIP), [DINO](https://github.com/facebookresearch/dino), and [DALL-E](https://github.com/borisdayma/dalle-mini). Thanks for their wonderful work.

## Citation
```
@inproceedings{shao2024Collaborative,
  title={Collaborative Consortium of Foundation Models for Open-World Few-Shot Learning},
  author={Shao, Shuai and Bai, Yu and Wang, Yan and Liu, Baodi and Liu, Bin},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2024}
}
```
