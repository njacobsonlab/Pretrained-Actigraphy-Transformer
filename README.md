# 🚀 Is Attention All You Need For Actigraphy? Pre-trained Transformers for Wearable Accelerometer Data 🏃‍♀️🏃

**Abstract:** ... blah blah

---

## 📔 Tutorial/Demo Notebooks:
These notebooks should be self-sufficient so you should be able to run them all the way through. Enjoy exploring!

### ⭐ How to: Fine-tune ALBERT + Built-in Model Explainability
[Launch Notebook](https://colab.research.google.com/drive/1sub_5m6fV91GbqEOWT8Sl5RjwN2QnhNh?usp=sharing)

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
[Launch Notebook](https://colab.research.google.com/drive/14VxoXzA374nNqYANI52rXbGuSW2ZAkxZ#scrollTo=_uzonweo6pYs)

This notebook covers:
1. Setting Up (importing/connecting to TPU)
2. Choosing Hyperparameters & Settings
3. Loading Demo Data
4. Loading Masked Autoencoder
5. Self-Supervised Training
6. Saving the Model Encoder
7. Inspecting the Autoencoder

---

## 🧠 Model Explainability
For running the model explainability functions, you may need to know a few details about the model setup.

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
  # decoder
  decoder_num_heads = 6
  decoder_ff_dim = 256
  decoder_num_layers = 1
  decoder_rate = 0.1

if size == "medium":

  patch_size = 18
  embed_dim = 96
  # encoder
  encoder_num_heads = 12
  encoder_ff_dim = 256
  encoder_num_layers = 2
  encoder_rate = 0.1
  # decoder
  decoder_num_heads = 12
  decoder_ff_dim = 256
  decoder_num_layers = 1
  decoder_rate = 0.1

if size == "large":

  patch_size = 9
  embed_dim = 96
  # encoder
  encoder_num_heads = 12
  encoder_ff_dim = 256
  encoder_num_layers = 4
  encoder_rate = 0.1
  # decoder
  decoder_num_heads = 12
  decoder_ff_dim = 256
  decoder_num_layers = 1
  decoder_rate = 0.1
```

## Attribution
Please cite our work if you use it 
