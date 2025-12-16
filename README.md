## CapZCIR: Leveraging Balanced Semantics in Caption Guided Approach for Zero-Shot Composed Image Retrieval

[](https://www.google.com/search?q=https://github.com/Giog97/CapZCIR/stargazers)
[](https://www.google.com/search?q=LICENSE)
[](https://www.google.com/search?q=%23%5BCitation%5D)

### Overview

This repository contains the official implementation of the article **“Leveraging Balanced Semantics in Caption Guided Approach for Zero-Shot Composed Image Retrieval”** presented at WACV 2026 and the modifications to its backbone introduced to improve performance.

Zero-Shot Composed Image Retrieval (ZS-CIR) addresses the challenge of retrieving a target image from a gallery using a multimodal query: a reference image combined with a textual modification (e.g., "the [reference image] wearing a red hat").

**CapZCIR** is designed to overcome the limitations of existing training-free ZS-CIR models, which often suffer from poor fine-grained visual detail capture (textual inversion) or caption hallucinations (LLM-based methods). CapZCIR introduces a novel "Balanced Semantics" approach guided by refined captions, providing dynamically adapting feature representations for superior retrieval performance.

---

### Core Methodology

CapZCIR focuses on achieving a **Balanced Semantic** representation by:

1.  **Refined Caption Generation:** Using a caption-guided approach to more accurately describe the reference image and its context, minimizing hallucination errors seen in general LLM-based methods.
2.  **Dynamic Feature Adaptation:** Moving beyond fixed (static) embedding representations to allow the model to dynamically adapt its features during the retrieval process, ensuring better alignment between the modified visual features and the textual description.
3.  **Improved Cross-Modal Alignment:** Enhancing the alignment between the projected visual features and the modification text, crucial for precise, fine-grained modifications.

---

### Installation and Setup

This project requires **Python 3.10** and specific versions of PyTorch and CUDA. We recommend using **Conda** for environment management to ensure all dependencies are met correctly.

#### 1\. Environment Creation

First, create and activate the dedicated Conda environment named `CapZCIR`:

```bash
# Create the Conda environment with Python 3.10
conda create -n CapZCIR python=3.10 -y

# Activate the new environment
conda activate CapZCIR
```

#### 2\. Core Dependencies (PyTorch & CUDA)

Install the required PyTorch version with the matching CUDA Toolkit (version 11.3 in this case).

```bash
# Install PyTorch, TorchVision, and Torchaudio
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -y
```

> **Note:** We recommend ensuring that the installed CUDA toolkit version is compatible with your GPU driver.

#### 3\. Additional Dependencies

Install required libraries, essential scientific libraries, and the **CLIP** model.

```bash
conda install -c conda-forge faiss-gpu=1.7.3 -y
conda install -c conda-forge mkl -y
conda install -c conda-forge scipy scikit-learn -y
conda install -c conda-forge wandb=0.13.10 -y

pip install -r requirements.txt

# Install the official CLIP repository via pip
pip install git+https://github.com/openai/CLIP.git
```

#### 4\. Final Requirements

Finally, install this remaining packages:

```bash
pip install --upgrade transformers #→ needed to use SAM with DAM
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
pip install --upgrade torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
```

#### 5\. Final Observations
Some dependencies required for the DAM captioner to work and obtain captions do not allow CapZCIR to be used, so
- To use DAM:
```bash
pip install transformers transformers==4.27.3
```
- To use CapZCIR:
```bash
pip install transformers==4.21.0
```
---

### Datasets

CapZCIR is evaluated on standard Zero-Shot Composed Image Retrieval benchmarks (CIRR, FashionIQ, CIRCO).

1. Dowload the following Dataset:
    - **Laion_Combined** dataset from **[here](https://drive.google.com/drive/folders/1xXCJJhVgM2nIF8JmgB6iQ50NWtPE-ICt)**. This dataset will be used for training. 
    - **FashionIQ** dataset from **[here](https://github.com/XiaoxiaoGuo/fashion-iq)** 
    - **CIRR** dataset from **[here](https://github.com/Cuberick-Orion/CIRR)** 
    - **CIRCO** dataset from **[here](https://github.com/miccunifi/CIRCO)**
2. Place the datasets in the `./data/datasets` directory.
3. Dowload the following precomputed captions with BLIP-2 and DAM captioner from **[here](https://drive.google.com/drive/folders/1u9BM-cuyHe6N9lZ2eCICdKgMs7_S9VGA?usp=sharing)**. 
    - **NB**: The file names are different from how they are named in the code, so you will need to rearrange the working directories and paths in the code to match the names of the files you have just downloaded. The code files you will need to modify are all located in `./data`.
4. Place the files with captions in the `./data/files` directory.



### Usage and Inference

To train CapZCIR from scratch on a new configuration:
1. Set the config.py file based on what kind of experiment and backbone do you want to run.
2. Change main.py, model.py, utils.py based on what kind of experiment and backbone do you want to run (the code to be taken is in the corresponding 'x_1enocder' or 'x_2encoder' file).
3. Run with the following command:

```bash
# [PLACEHOLDER: Command for training]
python main.py 
```

--
## 💻 Usage and Training

This section details how to configure and run the CapZCIR training pipeline. Due to the experimental nature of the code, different configurations (e.g., different backbones or experimental setups) are managed by manually switching code blocks.

### Training from Scratch

To set up and run a new training experiment:

1.  **Configure Experiment Settings (`config.py`):**
    Open `config.py` and set the core parameters, including name of the parametres file, training hyperparameters, and the specific experiment type you wish to run.

2.  **Select the Model Backbone:**
    CapZCIR supports various encoder backbones (e.g., CLIP variants, etc.). To switch between different model architectures or experimental setups (e.g., `1encoder` vs. `2encoder`):

      * Navigate to the necessary scripts (`main.py`, `model.py`, `utils.py`).
      * **Manually copy** the relevant code blocks from the corresponding file templates (e.g., from `x_1encoder` or `x_2encoder` files) into these main scripts.

3.  **Execute Training:**
    Once the configuration and code have been set for your desired experiment, start the training process using the main script:

    ```bash
    python main.py
    ```

### Inference and Evaluation

To evaluate a trained checkpoint on a benchmark dataset, use the following command structure:
- For CIRR, use the config_test.py to set the path of the pretrained checkpoint and use:
```bash
python cirr_test_submission.py
```

- For CIRCO, use the config_test.py to set the path of the pretrained checkpoint and use:
```bash
python circo_test_submission.py
```
NB: previus command generates the predictions file for uploading on the [CIRR Evaluation Server](https://cirr.cecs.anu.edu.au/) or
the [CIRCO Evaluation Server](https://circo.micc.unifi.it/). The predictions file will be saved in the `new/submission_{dataset}/` folder.

- For FashionIQ, use the config_test_fiq.py to set the path of the pretrained checkpoint and use:
```bash
python main_test_fiq.py
```
---

## Results 

The reported results were computed on the three benchmark datasets for the Zero-Shot Combined Image Retrieval (ZS-CIR) task, namely CIRR, FashionIQ, and CIRCO.
It should be noted that our results were obtained by training on laion_combined and always validating on CIRR, before testing the models on all three datasets. In contrast, in the literature, validation is typically performed on the same dataset used for testing. Consequently, our testing scenario on FashionIQ and CIRCO is more challenging, which is why it is marked with an asterisk (*) in the tables.

### Comparison with Textual Inversion Methods
<img src="analysis_results/Risults_vs_TextInversion.png" alt="Risultati vs Textual Inversion" width="800">

### Comparison with LLM-Based Methods
<img src="analysis_results/Risults_vs_LLM.png" alt="Risultati vs LLM" width="800">


---

## Authors

* **Giovanni Stefanini**\*
* **Pavan K. Rachabathuni**\*
* [**Andrea Ciamarra**](https://scholar.google.com/citations?hl=en&user=LTrUgeEAAAAJ)
* [**Marco Bertini**](https://scholar.google.com/citations?user=SBm9ZpYAAAAJ&hl=en)


**\*** Equal contribution.


## LICENSE
<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a><br />All material is made available under [Creative Commons BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). You can **use, redistribute, and adapt** the material for **non-commercial purposes**, as long as you give appropriate credit by **citing our paper** and **indicate any changes** that you've made.