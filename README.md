# BinaryImageSegmentation

# 1 Introduction
Binary image segmentation aims to separate foreground objects from background regions. In this project, the
challenge arises from the use of sparse scribble annotations as weak supervision. For the training set, both full
ground-truth segmentation masks and scribble annotations are available, which allows us to leverage the ground
truth for supervised learning while exploring how scribbles can guide the model. In contrast, the test set provides
only images and scribbles without ground-truth masks, requiring the model to generalize from weak supervision
and produce full-resolution binary masks at inference time.

# 2 Methodology

2.1 Data Preprocessing and Analysis
For the training set, both full ground-truth masks and sparse scribble annotations were provided. During
training, the ground-truth masks served as the supervision signal for loss computation, while the scribbles
were used to construct auxiliary distance maps that enrich the input representation. Specifically, we com-
puted Euclidean distance transforms for foreground (scribbles = 1) and background (scribbles = 0) pixels using
scipy.ndimage.distance transform. An exponential decay with 𝜎 = 50.0 was then applied to obtain smooth
soft distance maps.
For the test set, only images and scribbles were available (no ground-truth masks). The same preprocessing
pipeline was applied, generating distance maps from scribbles that, together with the RGB channels, form the input
to the trained model. This ensures consistency between training and inference while relying only on weak scribble
supervision at test time.
All images were normalized to the range [0, 1] by dividing pixel values by 255.0. Data augmentation included
random horizontal flips with 50% probability, applied jointly to images and distance maps. Finally, all data
were resized to a fixed resolution of (256 × 344) using skimage.transform.resize with constant mode
and preserve range=True. The final preprocessed input thus consisted of 5 channels: the 3 RGB channels
concatenated with the 2 distance maps.
No in depth exploratory data analysis was conducted, we directly applied preprocessing pipelines provided in the
starter code. We noted that the task involved natural images with a single foreground object per image, annotated
with sparse scribbles. Future work could include visualization of scribble distributions, object sizes or class
balance, which might help explain model behavior.

2.2 Model Architectures
All experiments were implemented in Python 3.11 using PyTorch 2.2 for model building and training, scikit-
image for resizing and distance transforms, scipy.ndimage for distance computations, and NumPy for numerical
operations.
Baseline UNet: The UNet architecture, originally proposed for biomedical image segmentation, employs a sym-
metric encoder-decoder structure that excels at capturing multi-scale features while preserving spatial information.
The encoder consists of successive downsampling stages, each comprising DoubleConv blocks—pairs of 3x3
convolutions followed by batch normalization and ReLU activations—interspersed with max-pooling operations
to reduce spatial dimensions and extract hierarchical features. Skip connections bridge the encoder and decoder,
concatenating feature maps from corresponding levels to mitigate the loss of fine-grained details during down-
sampling. The decoder mirrors the encoder with upsampling via transposed convolutions, followed by additional
DoubleConv blocks to refine the segmentation output. This design is particularly effective for tasks with limited
data, as it promotes feature reuse and gradient flow. However, in weakly supervised settings like scribble annota-
tions, the baseline UNet may struggle with ambiguous regions due to its reliance on global context without explicit
mechanisms to prioritize relevant cues.

Attention UNet: Building upon the baseline UNet, we introduce Attention Gates to enhance the model’s ability
to focus on pertinent spatial features, especially beneficial for sparse supervision where annotations provide only
partial guidance. Attention Gates are integrated at the skip connections, acting as self-gating modules that modulate
the feature maps from the encoder before they are concatenated with the decoder’s upsampled features. Specifically,
each gate computes compatibility scores between the gating signal (from the coarser decoder level) and the input
features (from the encoder), using 1x1 convolutions followed by sigmoid activation to generate soft attention
weights. These weights suppress activations in irrelevant background regions while amplifying those in potential
foreground areas, effectively filtering noise and improving localization. This extension maintains the UNet’s
efficiency but adds interpretability and robustness, allowing the model to better exploit scribble cues for precise
boundary delineation in challenging images with complex backgrounds or varying object scales.

2.3 Training Setup
We train both models from scratch using PyTorch. During training, the full ground-truth segmentation masks were
used as supervision to compute the loss, while the scribbles contributed through the distance maps included in the
input representation. Training details are as follows:
• Loss function: Combined BCE and Dice loss computed against the ground-truth masks:
L = 𝛼 · L𝐵𝐶𝐸 + (1 − 𝛼) · L𝐷𝑖𝑐𝑒, 𝛼 = 0.5 (1)
• Optimizer: Adam with initial learning rate 1 × 10−5.
• Scheduler: CyclicLR with maximum learning rate 3 × 10−4.
• Strategy: 5-fold cross-validation, batch size 8, 110 epochs, mixed precision training.
2.4 Model Selection
We compared models using 5-fold cross-validation. For each fold, mean IoU was calculated across validation
sets. The final model was selected based on the highest average mIoU across folds, ensuring robustness to data
variance and avoiding overfitting to a single split.
2.5 Metrics
The IoU measures the overlap between the predicted segmentation (P) and the ground truth (G). It is calculated
for both the background (bg) and object (obj) classes. The final performance is reported as the mean of these two

values (mIoU):
𝐼𝑜𝑈 = |𝑃 ∩ 𝐺 |
|𝑃 ∪ 𝐺 | , 𝑚𝐼𝑜𝑈 = 1
2 (𝐼𝑜𝑈𝑏𝑔 + 𝐼𝑜𝑈𝑜𝑏 𝑗 ) (2)

# 3 Results
<img width="461" height="130" alt="image" src="https://github.com/user-attachments/assets/957503e7-532e-4537-857f-957be1595342" />



                        





# 4 Conclusion
In this project, we successfully developed and evaluated an Attention U-Net model for weakly supervised binary
image segmentation. By leveraging distance transforms from sparse scribbles and incorporating attention gates,
our model achieved a promising mIoU of 0.773 on the validation data. The results underscore the effectiveness of
attention mechanisms in guiding segmentation models trained on limited annotations.
Theprimarylimitation of this approach is the increased model complexity and sensitivity to hyperparameter tuning.
Future work could extend this project by exploring alternative loss functions, such as focal loss, to better handle
difficult examples, or by investigating more advanced, transformer-based architectures for segmentation. We would
also like to include explainability techniques (e.g., Grad-CAM visualizations), robustness checks against input
perturbations.
