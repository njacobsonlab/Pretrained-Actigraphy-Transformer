# Is Attention All You Need For Actigraphy? Pre-trained Transformers for Wearable Accelerometer Data 🏃‍♀️🏃

>**All** the information to help you implement ALBERT can be found in this README. If anything is missing/confusing, please let us know (our contacts are below)! 

Abstract: [Missing]

---

## 📔 Tutorial/Demo Notebooks (via Google Colab) 
These notebooks should be self-sufficient so you should be able to run them all the way through. Enjoy exploring!

### ⭐ How to: Fine-tune ALBERT + Built-in Model Explainability
[Launch Notebook](https://colab.research.google.com/drive/13FBOP1rUeAfLmSrVv578XCGHqeEU4FJf)

This notebook will guide you through:
1. Setting up (importing/connecting to TPU)
2. Loading Demo Data
3. Loading Model
4. Finetuning Model
5. Evaluating Model
6. Model Explainability

> **Note:** You need to connect to TPUv2, or this notebook will **NOT** work.  
> In Google Colab, go to `runtime -> change runtime type`, then select TPUv2. This should already be the default when you open the link.

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

> **Note:** You need to connect to TPUv2, or this notebook will **NOT** work.  
> In Google Colab, go to `runtime -> change runtime type`, then select TPUv2. This should already be the default when you open the link.

---
## 💾 Albert Encoders Download
These are the H5 files that store the pre-trained transformer encoder weights. Don't worry, they are small, just a few megabytes. The demo notebook above, `How to: Fine-tune ALBERT + Built-in Model Explainability`, shows you where to load encoders. 

### Pretrained on 2003-2004, 2005-2006, 2011-2012, 2013-2014 NHANES Actigraphy (N=29,307)
1. [ALBERT-L](https://www.dropbox.com/scl/fi/man7n56fmo3m78bbeuic8/encoder_large_90_unsmoothed_mse_all.h5?rlkey=nnovpuo6yf42dqi9od3n9dqpl&st=mfc5f550&dl=0)
2. [ALBERT-M](https://www.dropbox.com/scl/fi/jc1gdzuj1tp6oq0cu9xzk/encoder_medium_90_unsmoothed_mse_all.h5?rlkey=a1609bxbd4pxyvnk5uw2zl5vs&st=drbco21l&dl=0)
3. [ALBERT-S](https://www.dropbox.com/scl/fi/0j03b6wzlav9p00qbg0ok/encoder_small_90_unsmoothed_mse_all.h5?rlkey=yu7s1jar6fbkfv1s71a05nyif&st=1zqo31t2&dl=0)
4. ALBERT-Conv: WIP

### Pretrained on 2003-2004, 2005-2006, 2011-2012 NHANES Actigraphy (N=21,538)
> ❗ Good if you want to conduct a study with 2013-2014 NHANES actigraphy data
1. [ALBERT-L](https://www.dropbox.com/scl/fi/gpa294hjl1cpt2tgf0s7o/encoder_large_90_unsmoothed_mse_all.h5?rlkey=8d7rv9qtt36ammgy14ed769we&st=cvhmwxeo&dl=0)
2. [ALBERT-M](https://www.dropbox.com/scl/fi/7h21c4sv3bbgsy3qdveb0/encoder_medium_90_unsmoothed_mse_all.h5?rlkey=w46b82qx328q0rxk8i1y16cnr&st=0lr3ho5o&dl=0)
3. [ALBERT-S](https://www.dropbox.com/scl/fi/drrs4q7itl83sq6c4ynfy/encoder_small_90_unsmoothed_mse_all.h5?rlkey=of53s0c9ki7mtoq9q8h6p0ybv&st=oufs4zd0&dl=0)
4. [ALBERT-Conv-L](https://www.dropbox.com/scl/fi/p5z1edbtwj4nhmpwz4u4z/conv_encoder_large_90_unsmoothed_mse_all.h5?rlkey=sjguwtxfdt42yzm2e9kw5b940&st=dkan59xd&dl=0)
5. [ALBERT-Conv-M](https://www.dropbox.com/scl/fi/fe5psorrfwuu5kbq0ya10/conv_encoder_medium_90_unsmoothed_mse_all.h5?rlkey=gg4r3irf91n091kkopp2jt490&st=h5uq30cz&dl=0)
6. [ALBERT-Conv-S](https://www.dropbox.com/scl/fi/77sisd63iqzqcm85l87cp/conv_encoder_small_90_unsmoothed_mse_all.h5?rlkey=gyvcbb0hh1x3f3sbyh0z2fjyp&st=5zug12sn&dl=0) 


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

## Contacts
**Corresponding Author:** 
* Franklin Ruan | franklin.y.ruan.24@dartmouth.edu <br>

**Co-author contacts:**


