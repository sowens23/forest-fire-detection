# Forest Fire Detection with Deep Learning

Binary image classification of aerial drone imagery as **Fire** or **No Fire**, comparing three convolutional neural network (CNN) approaches in TensorFlow/Keras. Built for CS 480/680 (Deep Learning in Python) at the University of Alaska Fairbanks, Fall 2024.

## Motivation

Wildfire frequency in Alaska has risen sharply over the past two decades. Early detection from aerial imagery could help responders act sooner, so this project explores which CNN architectures and training techniques best separate fire from non-fire images, and why models that look strong in training can fail on new data.

## Dataset

[IEEE FLAME Dataset: Aerial Imagery Pile Burn Detection Using Drones (UAVs)](https://ieee-dataport.org/open-access/flame-dataset-aerial-imagery-pile-burn-detection-using-drones-uavs)

| Split | Fire | No Fire | Total |
| --- | --- | --- | --- |
| Training | 25,018 | 14,357 | 39,375 |
| Test | 5,137 | 3,480 | 8,617 |

RGB images resized to 254x254 (224x224 for ResNet50). The classes are imbalanced: about 60% of test images contain fire.

## Approach

| Model | Architecture | Key techniques |
| --- | --- | --- |
| **Model 1: Baseline CNN** | 2 conv + max-pool blocks (32, 64 filters), dense 128, dropout 0.5 | Rescaling only |
| **Model 2: Regularized CNN** | Fewer filters (16, 32), dense 128, dropout 0.6 | L2 weight decay, heavy data augmentation, class weighting, early stopping, learning-rate scheduling |
| **Model 3: Transfer Learning** | Frozen ResNet50 (ImageNet weights), global average pooling, dense 128, dropout 0.5 | ResNet preprocessing, augmentation, early stopping, learning-rate scheduling |

## Results

| Model | Training accuracy | Test accuracy |
| --- | --- | --- |
| Model 1: Baseline CNN | ~99% | 48.4% |
| Model 2: Regularized CNN | ~94% | 58.6% |
| Model 3: ResNet50 | ~99% | 65 to 68% validation (see notes) |

**Model 1** memorized the training set: near-perfect training accuracy, but below-chance accuracy on unseen images, a clear case of overfitting. **Model 2's** regularization narrowed that gap substantially, though its test accuracy is close to what a model would get by always predicting "Fire" (about 60%), so it had not yet learned a strong general signal. **Model 3**, using pretrained ResNet50 features, showed the best validation accuracy.

## What I Learned

Reviewing the project afterward surfaced several issues that taught me as much as the models did:

1. **Test data was used for validation.** The test set guided early stopping and learning-rate decisions, which biases the reported test accuracy. A separate validation split carved from the training data would give an honest final evaluation.
2. **Half the training epochs were empty.** A `steps_per_epoch` mismatch caused the data generator to run out of data, so every other epoch trained on nothing (visible as near-zero-second epochs in the logs).
3. **Model 3's final evaluation called the wrong model.** The ResNet cell evaluated Model 2 on the test set, so ResNet50's true test accuracy was never measured.
4. **Class imbalance matters for interpretation.** With about 60% fire images, accuracy alone is misleading; precision, recall, and a confusion matrix would show more.

Planned improvements: a proper train/validation/test split, fixing the generator steps, fine-tuning upper ResNet50 layers, and reporting precision, recall, and F1.

## Running the Notebook

```
pip install -r requirements.txt
jupyter notebook firedetection.ipynb
```

Download items 7 (Training) and 8 (Test) from the IEEE FLAME dataset page and arrange them as:

```
data/
├── Training/
│   ├── Fire/
│   └── No_Fire/
└── Test/
    ├── Fire/
    └── No_Fire/
```

Training is slow on a CPU (roughly 25 to 40 minutes per model); a GPU is recommended.

## Repository Contents

```
forest-fire-detection/
├── firedetection.ipynb   # Data pipeline, all three models, and training logs
├── report.pdf            # Final project report
├── requirements.txt
└── README.md
```

A recorded project presentation is available [here](https://youtu.be/51oem7Sxg-c).

## Credits

Co-authored with **Chariety Moler** for CS 480/680 at the University of Alaska Fairbanks. Dataset: IEEE FLAME (Shamsoshoara et al.).