# Is Attention All You Need For Actigraphy? Foundation Models of Wearable Accelerometer Data for Mental Health Research 🏃‍♀️🏃

This is the GitHub associated with the paper "Is Attention All You Need For Actigraphy? Foundation Models of Wearable Accelerometer Data for Mental Health Research" <br>

Abstract: Wearable accelerometry (actigraphy) has provided valuable data for clinical insights since the 1970s and is increasingly important as wearable devices continue to become widespread. The effectiveness of actigraphy in research and clinical contexts is heavily dependent on the modeling architecture utilized. To address this, we developed the Pretrained Actigraphy Transformer (PAT)—the first pretrained and fully attention-based model designed specifically to handle actigraphy. PAT was pretrained on actigraphy from 29,307 participants in NHANES, enabling it to deliver state-of-the-art performance when fine-tuned across various actigraphy prediction tasks in the mental health domain, even in data-limited scenarios. For example, when trained to predict benzodiazepine usage using actigraphy from only 500 labeled participants, PAT achieved an 8.8 percentage-point AUC improvement over the best baseline. With fewer than 2 million parameters and built-in model explainability, PAT is robust yet easy to deploy in health research settings. <br>

paper: https://doi.org/10.48550/arXiv.2411.15240

## 📔 Tutorial/Demo Notebooks (via Google Colab) 
These notebooks should be self-sufficient so you should be able to run them all the way through. Enjoy exploring!

### ⭐ How to: Fine-tune PAT + Built-in Model Explainability
[Launch Notebook](https://colab.research.google.com/drive/1HemPmkADQYRW214ft8ep8ARkfxPkwEij#scrollTo=eAlhD3TN148g)

This notebook will guide you through:
1. Setting up (importing/connecting to TPU)
2. Loading Demo Data
3. Loading Model
4. Finetuning Model
5. Evaluating Model
6. Model Explainability

> **Note:** You need to connect to TPU **or** GPU, or this notebook will be **very** slow.  
> In Google Colab, go to `runtime -> change runtime type`, then select TPUv2 or GPU. TPU should already be the default when you open the link.

### 🎓 How to: Self-Supervised Pretraining
[Launch Notebook](https://colab.research.google.com/drive/1yLsxmd8fhQzkQLaIl5PB6T6CVyCebA-r)

This notebook will guide you through:
1. Setting Up (importing/connecting to TPU)
2. Choosing Hyperparameters & Settings
3. Loading Demo Data
4. Loading Masked Autoencoder
5. Self-Supervised Training
6. Saving the Model Encoder
7. Inspecting the Autoencoder

> **Note:** You need to connect to TPU, or this notebook will **not** work.  
> In Google Colab, go to `runtime -> change runtime type`, then select TPUv2. This should already be the default when you open the link.<br>
> It is possible to easily adapt this notebook so that you can run with GPUs instead if you choose

---
## 💾 PAT Encoders Weights Download
These are the H5 files that store the pre-trained transformer encoder weights. Don't worry, they are small, just a few megabytes. The demo notebook above, `How to: Fine-tune PAT + Built-in Model Explainability`, shows you where to load encoder weights. 

### Pretrained on 2003-2004, 2005-2006, 2011-2012, 2013-2014 NHANES Actigraphy (N=29,307)
1. [PAT-L](https://www.dropbox.com/scl/fi/dglz917p3hqw5mwbovsv2/PAT-L_21k_weights.h5?rlkey=ppzxvp9i7t9k8j3w9x77fjfil&st=3g3mm845&dl=1)
2. [PAT-M](https://www.dropbox.com/scl/fi/dsd6px97gcipqm80iie17/PAT-M_21k_weights.h5?rlkey=q480rjj5g2id88xt9feie70tj&st=ou924quo&dl=1)
3. [PAT-S](https://www.dropbox.com/scl/fi/ik45lrtqgenm61cgkkgkz/PAT-S_21k_weights.h5?rlkey=n2zv3jhdnvp7w8inir96y1ime&st=xch3lnra&dl=1)

### Pretrained on 2003-2004, 2005-2006, 2011-2012 NHANES Actigraphy (N=21,538)
> ❗ Good if you want to conduct a study with 2013-2014 NHANES actigraphy data
1. [PAT-L](https://www.dropbox.com/scl/fi/exk40hu1nxc1zr1prqrtp/PAT-L_29k_weights.h5?rlkey=t1e5h54oob0e1k4frqzjt1kmz&st=7a20pcox&dl=1)
2. [PAT-M](https://www.dropbox.com/scl/fi/hlfbni5bzsfq0pynarjcn/PAT-M_29k_weights.h5?rlkey=frbkjtbgliy9vq2kvzkquruvg&st=mxc4uet9&dl=1)
3. [PAT-S](https://www.dropbox.com/scl/fi/12ip8owx1psc4o7b2uqff/PAT-S_29k_weights.h5?rlkey=ffaf1z45a74cbxrl7c9i2b32h&st=mfk6f0y5&dl=1)

---

## 🧠 Model Explainability
You may need to know a few details about the model setup to run the model explainability functions. <br>
Here are some parameters for your reference:

```python
## Model Size
if size == "small":

  patch_size = 18
  embed_dim = 96
  # encoder
  encoder_num_heads = 6
  encoder_ff_dim = 256
  encoder_num_layers = 1
  encoder_rate = 0.1


if size == "medium":

  patch_size = 18
  embed_dim = 96
  # encoder
  encoder_num_heads = 12
  encoder_ff_dim = 256
  encoder_num_layers = 2
  encoder_rate = 0.1


if size == "large":

  patch_size = 9
  embed_dim = 96
  # encoder
  encoder_num_heads = 12
  encoder_ff_dim = 256
  encoder_num_layers = 4
  encoder_rate = 0.1

```

## 📜 Attribution
Please cite our work if you use it 

## ☎️ Contacts
**Corresponding Author:** Franklin Ruan | franklin.y.ruan.24@dartmouth.edu <br>


