
# Is Attention All You Need For Actigraphy? Pre-trained Transformers for Wearable Accelerometer Data 🏃‍♀️🏃

Abstract... blah blabh

# Tutorial/Demo Notebooks:
These notebooks should be self-sufficient so you should be able to run it all the way through :)

## ⭐ How to: Fine-tune ALBERT + How to use Built-in Model Explainability: https://colab.research.google.com/drive/1sub_5m6fV91GbqEOWT8Sl5RjwN2QnhNh?usp=sharing

This notebook will walk you through: 
* Setting up (importing/connecting to TPU)
* Loading Demo Data
* Loading Model
* Finetuning Model
* Evaluating Model
* Model Explainability 

You have to connect to TPUv2, or this notebook will NOT work <be>  
In Google Colab, go to runtime->change runtime type <br>
Then select TPUv2 <br>
However, this should already be the default setting when you open the link.

## How to: Self-Supervised Pretraining: https://colab.research.google.com/drive/14VxoXzA374nNqYANI52rXbGuSW2ZAkxZ#scrollTo=_uzonweo6pYs

This notebook will walk you through: 
* Setting Up (importing/connecting to TPU)
* Choosing your Hyperparameters & Settings
* Loading Demo Data
* Loading Masked Autoencoder
* Training in a Self-Supervised Manner
* Saving only the model encoder
* Inspecting the autoencoder

You have to connect to TPUv2, or this notebook will NOT work <be>  
In Google Colab, go to runtime->change runtime type <br>
Then select TPUv2 <br>
However, this should already be the default setting when you open the link.

# Model Explainability
You may need some model details to run the model explainability function (i.e, know how many transformer layers or the patch size of each sized model) <br> 
Here are all the parameters:

'''
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
'''

## Attribution
Please cite our work if you use it 
