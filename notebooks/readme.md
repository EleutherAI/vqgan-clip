# vqgan-clip

## VQGAN+CLIP methods

### [VQGAN+CLIP_(codebook_sampling_method).ipynb](https://github.com/EleutherAI/vqgan-clip/blob/main/VQGAN%2BCLIP_(codebook_sampling_method).ipynb) ([on Colab](https://colab.research.google.com/drive/15UwYDsnNeldJFHJ9NdgYBYeo6xPmSelP))

Generates images from text prompts with VQGAN and CLIP (codebook sampling method).

Codebook sampling optimizes a grid of independent categorical distributions over VQGAN codes, parameterized by logits, with gradient descent, for the decoded image's similarity to the CLIP prompt.

### [VQGAN+CLIP_(z+quantize_method).ipynb](https://github.com/EleutherAI/vqgan-clip/blob/main/VQGAN%2BCLIP_(z%2Bquantize_method).ipynb) ([on Colab](https://colab.research.google.com/drive/1L8oL-vLJXVcRzCFbPwOoMkPKJ8-aYdPN))

Generates images from text prompts with VQGAN and CLIP (z+quantize method).

### [VQGAN+CLIP_(MSE regularized z+quantize_method)](https://github.com/EleutherAI/vqgan-clip/blob/main/notebooks/VQGAN%2BCLIP_(MSE%20regularized%20z%2Bquantize_method).ipynb)

Generates images from text prompts with VQGAN and CLIP (z+quantize method) regularized with MSE.

### [Semantic_Style_Transfer_with_CLIP+VQGAN_(Gumbel_VQGAN).ipynb](https://github.com/EleutherAI/vqgan-clip/blob/main/Semantic_Style_Transfer_with_CLIP%2BVQGAN_(Gumbel_VQGAN).ipynb) ([on Colab](https://colab.research.google.com/drive/1kNZYKlGRkkW4SDoawnq1ZoH0jhnX_jlV))

Zero shot semantic style transfer

## Non-VQGAN CLIP methods

### [OpenAI_dVAE+CLIP.ipynb](https://github.com/EleutherAI/vqgan-clip/blob/main/OpenAI_dVAE%2BCLIP.ipynb) ([on Colab](https://colab.research.google.com/drive/10DzGECHlEnL4oeqsN-FWCkIe_sq3wVqt))

Generates images from text prompts with the OpenAI discrete VAE and CLIP.

Codebook sampling optimizes a grid of independent categorical distributions over OpenAI discrete VAE codes, parameterized by logits, with gradient descent, for the decoded image's similarity to the CLIP prompt.

### [CLIP_Guided_Diffusion.ipynb](https://github.com/EleutherAI/vqgan-clip/blob/main/CLIP_Guided_Diffusion.ipynb) ([on Colab](https://colab.research.google.com/drive/1ED6_MYVXTApBHzQObUPaaMolgf9hZOOF))

Generates images from text prompts with CLIP guided diffusion (256x256 output size).

CLIP guided diffusion samples from the diffusion model conditional on the output image being near the target CLIP embedding. In this notebook, the fact that CLIP is not noise level conditioned is dealt with by applying a Gaussian blur with timestep-dependent radius before processing the current timestep's output with CLIP.

### [CLIP_Guided_Diffusion_HQ_256x256.ipynb](https://github.com/EleutherAI/vqgan-clip/blob/main/CLIP_Guided_Diffusion_HQ_256x256.ipynb) ([on Colab](https://colab.research.google.com/drive/12a_Wrfi2_gwwAuN3VvMTwVMz9TfqctNj))

Generates images from text prompts with CLIP guided diffusion (256x256 output size).

CLIP guided diffusion samples from the diffusion model conditional on the output image being near the target CLIP embedding. In this notebook, the fact that CLIP is not noise level conditioned is dealt with by obtaining a denoised prediction of the final timestep and processing that with CLIP.

### [CLIP_Guided_Diffusion_HQ_512x512.ipynb](https://github.com/EleutherAI/vqgan-clip/blob/main/CLIP_Guided_Diffusion_HQ_512x512.ipynb) ([on Colab](https://colab.research.google.com/drive/1V66mUeJbXrTuQITvJunvnWVn96FEbSI3))

Generates images from text prompts with CLIP guided diffusion (512x512 output size).

CLIP guided diffusion samples from the diffusion model conditional on the output image being near the target CLIP embedding. In this notebook, the fact that CLIP is not noise level conditioned is dealt with by obtaining a denoised prediction of the final timestep and processing that with CLIP. It uses a class-conditional diffusion model and this is dealt with by randomizing the input class on each timestep.

### [CLIP_Guided_Diffusion_HQ_512x512_Uncond.ipynb](https://github.com/EleutherAI/vqgan-clip/blob/main/CLIP_Guided_Diffusion_HQ_512x512_Uncond.ipynb) ([on Colab](https://colab.research.google.com/drive/1QBsaDAZv8np29FPbvjffbE1eytoJcsgA))

Generates images from text prompts with CLIP guided diffusion (512x512 output size).

CLIP guided diffusion samples from the diffusion model conditional on the output image being near the target CLIP embedding. In this notebook, the fact that CLIP is not noise level conditioned is dealt with by obtaining a denoised prediction of the final timestep and processing that with CLIP. It uses an unconditional diffusion model that was fine-tuned from the released 512x512 conditional diffusion model using the same training set but with no class labels.

### [CLIP_Semantic_Segmentation.ipynb](https://github.com/EleutherAI/vqgan-clip/blob/main/CLIP_Semantic_Segmentation.ipynb) ([on Colab](https://colab.research.google.com/drive/1BMfl0s0kdgQOTNfeJSF2n6x7Y4D3IyeZ))

Generates a mask from an image using a pixel-wise average over random crops scored by CLIP. In other words, this is a Monte Carlo method. Needs to be calibrated for optimal results. Also used in CLIP Semantic Segmentation.

### [CLIP_Decision_Transformer.ipynb](https://github.com/EleutherAI/vqgan-clip/blob/main/CLIP_Decision_Transformer.ipynb) ([on Colab](https://colab.research.google.com/drive/1dFV3GCR5kasYiAl8Bl4fBlLOCdCfjufI))

Generates images from text prompts with a CLIP conditioned Decision Transformer.

This model outputs logits for the next VQGAN token conditioned on a CLIP embedding, target CLIP score, and sequence of past VQGAN tokens (possibly length 0). It can be used to sample images conditioned on CLIP text prompts.

