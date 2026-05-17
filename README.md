# STAMBRIDGE: Spectral-Temporal Adaptive Mid-Feature Bridge for Stable EEG Visual Decoding

<div align="center">
<img src="imgs/fig_overall.png" alt="STAMBRIDGE Framework" style="max-width: 90%; height: auto;"/>
<p>Overall framework of our proposed STAMBRIDGE model, illustrating the Spectral-Temporal Adaptive Module (STAM) and the Mid-Feature Semantic Bridge (MFSB).</p>
</div>
## 📖 Abstract

Electroencephalography (EEG) visual decoding remains challenging due to the modality gap between low-SNR neural signals and highly structured vision-language spaces. To address this, we propose **STAMBRIDGE**, a versatile two-stage framework that sequentially tackles feature conditioning and cross-modal alignment:
1. **Spectral-Temporal Adaptive Module (STAM):** Replaces hard frequency masking with amplitude-derived soft channel weighting and multi-scale temporal convolutions, explicitly preserving frequency-aware transients while avoiding time-domain ringing artifacts.
2. **Mid-Feature Semantic Bridge (MFSB):** A model-agnostic module that constructs a regularized intermediate space through directed cross-modal interactions, enabling staged distillation and more stable semantic alignment.

STAMBRIDGE achieves state-of-the-art 200-way zero-shot retrieval performance on the THINGS-EEG benchmark, with **34.50% Top-1** and **65.95% Top-5** accuracy. 

## 🧠 Architecture Data Flow

To effectively capture the spatio-temporal dynamics of neural signals and explicitly protect feature integrity, our encoding pipeline strictly follows this logic: 
**Multichannel EEG $\rightarrow$ Frozen Subject Adaptation $\rightarrow$ iTransformer blocks $\rightarrow$ STAM module.**

## 📂 Repository Structure

Based on the current repository, the main components are organized as follows:

* `STAMEncoder.py`: Contains the core EEG encoder integrating the Subject-Specific Linear Layer, iTransformer backbone, and the STAM module.
* `semantic_bridge_plugin.py`: Implementation of the Mid-Feature Semantic Bridge (MFSB) supporting Multi-modal Adaptive Directional Routing (MADR) and staged distillation.
* `subject_layers/`: Directory containing specific subject-level adaptation layers and baseline transformer components (`Transformer_EncDec.py`, `Embed.py`, etc.).
* `datasets.py`: Dataloader designed for the THINGS-EEG benchmark, including memory-safe feature loading.
* `train_stambridge.py`: Main training script for contrastive alignment and staged distillation.
* `get_eegfeatures.py`: Script to extract robust EEG representations after training.
* `diffusion_prior.py` / `custom_pipeline.py` / `generation.py`: Scripts handling the generative prior-based qualitative visual reconstruction via latent diffusion.
* `loss.py`: Implementation of the InfoNCE and CLIP loss functions.
* `util.py` / `debug_util.py`: Utility functions for logging (e.g., Weights & Biases) and debugging.

## 🚀 Getting Started

### 1. Environment Setup

Clone the repository and install the required dependencies:

```bash
git clone [https://github.com/YourUsername/YourRepoName.git](https://github.com/YourUsername/YourRepoName.git)
cd YourRepoName
pip install -r requirements.txt
