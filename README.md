# Assignment 1
**Deadline:** 1st September, 11:59pm.

---
## Q1. Adversarial Captioning via CLIP and BLIP [3]
In this question, you will explore how adversarial perturbations can manipulate image representations and captions in Vision-Language Models.

### 1. Feature Extraction with CLIP
* Load a pretrained CLIP model (e.g., ViT-B/32).
* For a given input image, compute its image embedding. Say, an image from CIFAR 10.
* For a chosen target caption, compute the corresponding text embedding. For example, 'a picture of IIITD', 'a picture of stop light', or 'a picture of ADL lecture'.

### 2. Loss Function
* Define the L2 loss between the normalized image embedding and the target caption embedding:
    ```
    L(x, t) = ||ẑ_I(x) - ẑ_T(t)||_2^2
    ```
    where `ẑ_I` and `ẑ_T` denote normalized image and text embeddings, respectively.
* Freeze all CLIP parameters so gradients flow only to the input image.

### 3. Attack
* Compute the gradient of the loss with respect to the input image.
* Perform a single-step FGSM update:
    ```
    x_adv = clip_[0,1](x + ε · sign(∇x L))
    ```
* Generate adversarial images for the aforementioned captions.

### 4. Evaluation with BLIP
* Feed both the clean image and the adversarial image into a pretrained BLIP captioning model.
* Compare the generated captions and comment on how the adversarial perturbation alters the semantics.

### 5. Analysis
* Report the following: cosine similarity between image and target caption embeddings (before vs. after attack).

### Deliverables
* Code implementation (Colab notebook or `.py` file).
* A short report with:
    * Original and adversarial images.
    * Generated captions (clean vs. adversarial).
    * Quantitative results (cosine similarities).
    * Brief discussion of observations.

---
## Q2. Distillation [3]
Use a large pretrained CNN/Transformer model on ImageNet. Use **ResNet50** and **ViT** as teachers. Use the CIFAR-10 dataset and use 100 images per class.

a. Finetune the large teacher model on this set of 1000 samples. Use mixup augmentation.

b. Now consider a student model with two transformer blocks. Sample another set of 1000 samples from CIFAR-10, again 100 per class. But this set should be non-overlapping with the set used for finetuning. Perform distillation and report accuracies for finetuning and distillation. For evaluation, the full test set of 10,000 images should be used. For the student, a patch size of 8x8 can be used. Use 2 and 4 heads, and report accuracies for each case.

c. Analyze which teacher is better.

---
## Q3. SSL [3]
Use the train set and student model from the previous question Q2.

a. Train it using contrastive learning, that is, InfoNCE loss. Report the result on the test set of Q2. Report results for both 2 and 4 heads.

b. Vary the batch size and find the effect of batch size. Choose batch sizes of 4, 8, and 24.

For Q2 and Q3, report all compiled results in the report along with respective settings.
