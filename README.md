### This fork lets you use CLIP's vision transformer to generate FFT adversarial images ❓🤖‍💫

- `python train.py --model clipvit`

In `utils/models.py` (at the very end), you can select the CLIP model: 

`model, preprocess = clip.load("ViT-L/14")`

Change that to e.g. `"ViT-B/32"` to use the smallest CLIP model if you want a quick(er) look.

- `python mask_train.py --model clipvit --attack FMN --mask adversarial`
- `python mask_train.py --model clipvit --attack FMN --mask essential`

`--attack: Use FMN, PGD, DF`

Fast Minimum Norm - Projected Gradient Descent - DeepFool

- `python correlation.py --model clipvit --attack FMN`

---------
👇 See the original readme.md for further details 👇
---------

# Relating Implicit Bias and Adversarial Attacks through Intrinsic Dimension

Code for the paper, tested with `python>=3.8, <=3.10`.

Required libraries are reported in `requirements.txt` and they can be installed via `pip`. 

Data and models can be downloaded by running:

```makefile
bash preparation.sh
```

Then, the first step is to obtain trained models, which can by done by typing:

```
python train.py --model
```

The `model` argument can be chosen between `resnet20` (for CIFAR-10) and `resnet18` and `vit` (for Imagenette). This script will create a `trained_models` folder containing the trained parameters for the chosen model, to be used for the following experiments.

Fourier masks (essential frequency masks and adversarial frequency masks) can be trained by running:

```
python mask_train.py --model --attack --mask
```

The first argument follows the same syntax as above, while `attack` determines the kind of adversarial attack to employ and it can be chosen among `FMN`, `PGD` and `DF` (for respectively Fast Minimum Norm , Projected Gradient Descent  and DeepFool). `mask` defines the type of masks to be trained, either `essential` or `adversarial`.

Once both essential frequency masks and adversarial frequency masks have been computed for a given model-attack pair, correlations (based on cosine similarity and on the novel method based on Intrinsic Dimension) can be computed using:

```
python correlation.py --model --attack
```

The experiment on class-specificity of masks, reported in sections 4.7 and A.6, can be reproduced by running:

```
python class_specificity.py
```

And, finally, the class-level masks can be computed and tested running the notebook `class_level_masks.ipynb`.
