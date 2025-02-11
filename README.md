# UNOT: Universal Neural Optimal Transport
This is the official repo for the paper "Universal Neural Optimal Transport" (Geuter et al., 2025).
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

model = load_fno("Models/unot_fno.pt")
```

To use the FNO trained on variable $\epsilon$, you can load the model as follows:
```python
from src.evaluation.import_models import load_fno_var_epsilon

model = load_fno("Models/unot_fno_var_eps.pt")
```


## Training
If you want to train your own model, you first need to prepare the test datasets, and can then run a train script as
outlined below.

### Prepare Datasets
To download the test datasets, run
```python
scripts/make_data.py
```
Then, create test datasets with
```python
scripts/create_test_set.py
```

### Training a new Model
To train the model, run
```python
scripts/main_neural_operator.py
```
Various training hyperparameters as well as other (boolean) flags can be passed to this script; 
e.g. to train without wandb logging, run
```python
scripts/main_neural_operator.py --no-wandb
```
The folder also contains training files to train a model with variable $\epsilon$, or an
MLP instead of an FNO, which only accepts fixed size inputs, but can be trained within
minutes.


## 


## Citation
If you find this repository helpful, please consider citing our paper:

```bibtex
@article{geuter2025universal,
    title={Universal Neural Optimal Transport},
    author={Geuter, J. and Kornhardt, G. and Tomasson, I. and Laschos, V.},
    year={2025},
    url={coming}
}
```





