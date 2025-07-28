# UNOT: Universal Neural Optimal Transport
This is the official repo for the paper ["Universal Neural Optimal Transport"](https://arxiv.org/abs/2212.00133v5)
(Geuter et al., ICML 2025).
To get started, install the requirements via

```bash
pip install -r requirements.txt
```


## Using the pretrained Model
The pretrained model used for all our experiments is uploaded to the `Models` folder. Make sure to `git lfs pull`
instead of `git pull` to pull the model files as well (if you don't wan't to use the pretrained model, `git pull`
suffices). To use the pretrained FNO (Fourier Neural Operator), simply run

```python
from src.evaluation.import_models import load_fno
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = load_fno("unot_fno", device=device)
mu = ...    # first flattened input measure, shape (batch_size, resolution**2)
nu = ...    # second flattened input measure
g = model(mu, nu)                       # shape (batch_size, resolution**2)
```

To use the FNO trained on variable $\epsilon$, you can load the model as follows:
```python
from src.evaluation.import_models import load_fno_var_epsilon

model = load_fno_var_epsilon("unot_fno_var_eps")
```


## Training
If you want to train your own model, you first need to prepare the test datasets, and can then run a train script as
outlined below.

### Prepare Datasets
To download the test datasets, run
```python
python scripts/make_data.py
```
Then, create test datasets with
```python
python scripts/create_test_set.py
```

### Training a new Model
To train the model, run
```python
python scripts/main_neural_operator.py
```
Various training hyperparameters as well as other (boolean) flags can be passed to this script; 
e.g. to train without wandb logging, run
```python
python scripts/main_neural_operator.py --no-wandb
```
The folder also contains training files to train a model with variable $\epsilon$, or an
MLP instead of an FNO, which only accepts fixed size inputs, but can be trained within
minutes.


## Citation
If you find this repository helpful, please consider citing our paper:

```bibtex
@inproceedings{
geuter2025universal,
title={Universal Neural Optimal Transport},
author={Jonathan Geuter and Gregor Kornhardt and Ingimar Tomasson and Vaios Laschos},
booktitle={Forty-second International Conference on Machine Learning},
year={2025},
url={https://openreview.net/forum?id=t10fde8tQ7}
}
```





