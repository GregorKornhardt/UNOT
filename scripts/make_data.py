"""
make_data.py
------------

This script will download and save data for testing.
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.data_functions import (
    get_mnist,
    get_cifar,
    get_lfw,
    get_quickdraw,
    get_quickdraw_multi,
    get_facial_expression,
)

n_samples = 10000
data_path = "./Data"

# mnist
path_mnist = data_path + "/mnist.pt"
get_mnist(n_samples, path_mnist)

# lfw
try:
    path_lfw = data_path + "/lfw.pt"
    get_lfw(n_samples, path_lfw)
except:
    print("LFW dataset not accessible, due to http://vis-www.cs.umass.edu/lfw being down")

# cifar
path_cifar = data_path + "/cifar.pt"
get_cifar(n_samples, path_cifar)

# bear
root_np = data_path + "/quickdraw"
path_bear = "./Data/bear.pt"
class_name = "bear"
get_quickdraw(n_samples, root_np, path_bear, class_name)

# car
root_np = data_path + "/quickdraw"
path_bear = "./Data/car.pt"
class_name = "car"
get_quickdraw(n_samples, root_np, path_bear, class_name)

# quickdraw
root_np = data_path + "/quickdraw"
path_quickdraw = data_path + "/quickdraw.pt"
n_classes = 8
get_quickdraw_multi(n_samples, n_classes, root_np, path_quickdraw)

# facialexpression
root_np = data_path + "/quickdraw"
path_facialexpression = data_path + "/facialexpression.pt"
class_name = "facialexpression"
get_facial_expression(n_samples, path_facialexpression)
