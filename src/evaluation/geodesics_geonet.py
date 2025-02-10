import torch
import ot
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import argparse
from torch.functional import F
from PIL import Image


import src.evaluation.import_models as im
import src.utils.data_functions as df
import src.ot.cost_matrix  as cost

def import_measures(device = 'cpu'):
    a = [[0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.10196078,
        0.19215686, 0.18039216, 0.03137255, 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.00392157, 0.01176471, 0.15686275, 0.65098039,
        0.97647059, 0.91372549, 0.2745098 , 0.02352941, 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.00392157, 0.00784314, 0.14117647, 0.62745098, 0.96862745,
        0.99607843, 0.92941176, 0.47058824, 0.08627451, 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.03921569, 0.32941176, 0.90196078, 0.99215686,
        0.98823529, 0.8       , 0.21568627, 0.01568627, 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.00392157,
        0.        , 0.07058824, 0.49803922, 0.95294118, 1.        ,
        0.90588235, 0.38039216, 0.04313725, 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.00392157,
        0.        , 0.12156863, 0.68627451, 0.97254902, 0.95294118,
        0.58823529, 0.07843137, 0.        , 0.00392157, 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.00392157, 0.00392157, 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.00392157,
        0.        , 0.18823529, 0.80392157, 0.96862745, 0.77647059,
        0.1372549 , 0.        , 0.00392157, 0.00784314, 0.01568627,
        0.00784314, 0.        , 0.        , 0.        , 0.00392157,
        0.01176471, 0.00392157, 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.00392157, 0.        ,
        0.02352941, 0.34509804, 0.89803922, 0.90588235, 0.44313725,
        0.02352941, 0.        , 0.00392157, 0.01568627, 0.03529412,
        0.01568627, 0.        , 0.        , 0.00392157, 0.01176471,
        0.01568627, 0.00392157, 0.        ],
       [0.        , 0.        , 0.00392157, 0.01176471, 0.01176471,
        0.00392157, 0.        , 0.        , 0.00392157, 0.        ,
        0.12156863, 0.69019608, 0.96862745, 0.81960784, 0.15294118,
        0.00392157, 0.        , 0.        , 0.01176471, 0.01568627,
        0.00392157, 0.        , 0.        , 0.00392157, 0.00784314,
        0.00392157, 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.00392157, 0.00784314,
        0.00392157, 0.        , 0.        , 0.00392157, 0.        ,
        0.23137255, 0.92156863, 0.94901961, 0.62352941, 0.04705882,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.00392157, 0.00392157, 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.00392157, 0.        , 0.04313725,
        0.43921569, 0.99215686, 0.85490196, 0.27058824, 0.00784314,
        0.        , 0.00392157, 0.00392157, 0.00392157, 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.00784314, 0.        , 0.17254902,
        0.82352941, 0.97254902, 0.69019608, 0.07058824, 0.00392157,
        0.00392157, 0.        , 0.        , 0.        , 0.00392157,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.00392157, 0.00392157, 0.01960784, 0.35294118,
        0.98431373, 0.92941176, 0.52941176, 0.04705882, 0.        ,
        0.03529412, 0.25490196, 0.40784314, 0.35686275, 0.12941176,
        0.02352941, 0.00392157, 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.00392157, 0.        , 0.10196078, 0.59215686,
        0.99215686, 0.85882353, 0.37647059, 0.16078431, 0.44705882,
        0.73333333, 0.85882353, 0.90980392, 0.87843137, 0.70588235,
        0.32156863, 0.04705882, 0.        , 0.00392157, 0.        ,
        0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.00784314, 0.        , 0.19607843, 0.79215686,
        0.96862745, 0.81960784, 0.45490196, 0.5372549 , 0.88235294,
        1.        , 1.        , 1.        , 1.        , 0.96862745,
        0.7372549 , 0.21568627, 0.00392157, 0.        , 0.        ,
        0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.00392157, 0.        , 0.27843137, 0.90588235,
        0.96078431, 0.84705882, 0.68627451, 0.84313725, 0.97647059,
        0.95294118, 0.83921569, 0.88235294, 0.95294118, 0.96862745,
        0.88627451, 0.31764706, 0.01568627, 0.        , 0.        ,
        0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.01176471, 0.34901961, 0.9372549 ,
        0.97254902, 0.91764706, 0.85490196, 0.90196078, 0.82745098,
        0.43529412, 0.2745098 , 0.40392157, 0.85490196, 0.98039216,
        0.87058824, 0.2745098 , 0.00392157, 0.00392157, 0.        ,
        0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.00392157, 0.01176471, 0.34509804, 0.9372549 ,
        0.98039216, 0.96862745, 0.92941176, 0.84705882, 0.50980392,
        0.05098039, 0.        , 0.36078431, 0.88235294, 0.98039216,
        0.83137255, 0.20392157, 0.        , 0.00392157, 0.        ,
        0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.00392157, 0.        , 0.24705882, 0.81960784,
        0.99215686, 0.98431373, 0.97647059, 0.89803922, 0.57254902,
        0.23529412, 0.37254902, 0.78823529, 0.97254902, 0.96862745,
        0.56470588, 0.09019608, 0.        , 0.00392157, 0.        ,
        0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.00392157, 0.        , 0.10196078, 0.48235294,
        0.87843137, 1.        , 1.        , 0.98039216, 0.88235294,
        0.71372549, 0.82745098, 0.98431373, 0.99215686, 0.79215686,
        0.23529412, 0.00784314, 0.        , 0.00392157, 0.        ,
        0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.00392157, 0.00392157, 0.01960784, 0.14117647,
        0.45490196, 0.77254902, 0.92156863, 0.96862745, 0.96862745,
        0.95294118, 0.94117647, 0.86666667, 0.6745098 , 0.34117647,
        0.05882353, 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.00784314, 0.00784314,
        0.06666667, 0.24705882, 0.38431373, 0.43921569, 0.44313725,
        0.43137255, 0.43137255, 0.34901961, 0.16862745, 0.02352941,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.00392157,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.00392157, 0.00392157, 0.        , 0.        , 0.        ,
        0.00392157, 0.00784314, 0.01176471, 0.01176471, 0.01176471,
        0.01176471, 0.00784314, 0.00784314, 0.00392157, 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.00392157, 0.00392157, 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        ]]

    b = [[0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.00392157, 0.00784314, 0.01176471, 0.01568627,
        0.00784314, 0.00392157, 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.00392157,
        0.00784314, 0.05098039, 0.16862745, 0.28627451, 0.2745098 ,
        0.09803922, 0.01176471, 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.00392157, 0.00392157, 0.        ,
        0.00784314, 0.0745098 , 0.30588235, 0.56470588, 0.55686275,
        0.19215686, 0.01176471, 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.00392157, 0.00392157, 0.        , 0.        ,
        0.00392157, 0.04705882, 0.30196078, 0.69803922, 0.67843137,
        0.16862745, 0.        , 0.        , 0.        , 0.        ,
        0.00392157, 0.00392157, 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.00392157, 0.00392157, 0.        , 0.        ,
        0.        , 0.02745098, 0.27058824, 0.7254902 , 0.69803922,
        0.15294118, 0.        , 0.        , 0.        , 0.00784314,
        0.01568627, 0.01176471, 0.00392157, 0.        , 0.        ,
        0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.00392157, 0.00784314, 0.00392157, 0.        , 0.        ,
        0.        , 0.02352941, 0.2745098 , 0.77647059, 0.78823529,
        0.20784314, 0.00392157, 0.        , 0.        , 0.01568627,
        0.03137255, 0.01568627, 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.00392157, 0.00392157, 0.00392157, 0.        , 0.        ,
        0.        , 0.03529412, 0.34901961, 0.83529412, 0.87058824,
        0.30196078, 0.01960784, 0.        , 0.00784314, 0.01960784,
        0.01960784, 0.00784314, 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.00784314, 0.00392157, 0.        , 0.        ,
        0.        , 0.04313725, 0.34117647, 0.82352941, 0.8745098 ,
        0.34901961, 0.03529412, 0.        , 0.00392157, 0.01176471,
        0.00392157, 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.00784314, 0.00784314, 0.        , 0.        ,
        0.        , 0.02745098, 0.21176471, 0.70980392, 0.83921569,
        0.39607843, 0.05490196, 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.00392157, 0.00784314, 0.        , 0.        ,
        0.00392157, 0.00784314, 0.10588235, 0.56470588, 0.83137255,
        0.51764706, 0.08627451, 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.00392157, 0.00784314, 0.00392157, 0.        ,
        0.00392157, 0.00392157, 0.08627451, 0.61960784, 0.9254902 ,
        0.75294118, 0.16862745, 0.        , 0.        , 0.        ,
        0.00392157, 0.00392157, 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.00392157, 0.00392157, 0.        ,
        0.00392157, 0.        , 0.08235294, 0.7372549 , 0.99607843,
        0.9254902 , 0.27843137, 0.01176471, 0.        , 0.00392157,
        0.01176471, 0.01176471, 0.00392157, 0.        , 0.        ,
        0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.00392157, 0.        , 0.        ,
        0.00392157, 0.        , 0.07843137, 0.77647059, 1.        ,
        0.95294118, 0.30196078, 0.01176471, 0.        , 0.00392157,
        0.01568627, 0.01568627, 0.00392157, 0.        , 0.        ,
        0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.00392157, 0.        , 0.        ,
        0.00392157, 0.        , 0.07843137, 0.74509804, 0.99607843,
        0.9372549 , 0.24313725, 0.00392157, 0.        , 0.        ,
        0.00784314, 0.00784314, 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.00784314, 0.00392157, 0.        ,
        0.        , 0.        , 0.08235294, 0.69019608, 0.98431373,
        0.90980392, 0.2       , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.00392157, 0.01176471, 0.01176471, 0.00392157,
        0.00392157, 0.00392157, 0.08235294, 0.63137255, 0.96078431,
        0.89411765, 0.23529412, 0.00392157, 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.00784314, 0.01960784, 0.03137255, 0.01960784,
        0.00784314, 0.00392157, 0.06666667, 0.53333333, 0.93333333,
        0.91764706, 0.34117647, 0.02352941, 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.00784314, 0.02352941, 0.03137255, 0.01568627,
        0.00392157, 0.        , 0.03921569, 0.44705882, 0.92941176,
        0.96470588, 0.54117647, 0.07058824, 0.        , 0.00392157,
        0.        , 0.        , 0.00392157, 0.00392157, 0.00392157,
        0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.00392157, 0.01176471, 0.01176471, 0.00392157,
        0.        , 0.        , 0.01568627, 0.34117647, 0.90196078,
        0.97647059, 0.7254902 , 0.14901961, 0.00392157, 0.00392157,
        0.00392157, 0.00392157, 0.00392157, 0.00392157, 0.00392157,
        0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.00392157, 0.19607843, 0.81960784,
        0.95686275, 0.77254902, 0.2       , 0.01176471, 0.00392157,
        0.00392157, 0.00392157, 0.00392157, 0.        , 0.        ,
        0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.00392157, 0.00392157, 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.09019608, 0.52941176,
        0.79607843, 0.66666667, 0.16862745, 0.01176471, 0.00392157,
        0.00392157, 0.00392157, 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.00392157, 0.00784314, 0.00392157, 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.02352941, 0.1372549 ,
        0.31764706, 0.2745098 , 0.0627451 , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.00392157, 0.00392157, 0.00392157, 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.00392157,
        0.00392157, 0.00784314, 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        ]]
    a = torch.tensor(a).unsqueeze(0).unsqueeze(0)

    length = 28
    dust_const = 1e-6
    a = F.interpolate(a, size=(length, length), mode='bilinear')
    a[a < 0.1] = 0
    a /= a.sum()
    a += dust_const
    a /= a.sum()
    b = torch.tensor(b)
    b = F.interpolate(b.unsqueeze(0).unsqueeze(0), size=(length, length), mode='bilinear').squeeze()
    b[b < 0.1] = 0

    b /= b.sum()
    b += dust_const
    b /= b.sum()
    mu, nu = a.to(device).flatten(), b.to(device).flatten()
    return mu, nu


def barycenter(predictor, mus, cost_matrix, weights): 
    v0 = torch.ones(mus.shape[1],requires_grad = (True), device = mus.device)

    softmax = torch.nn.Softmax(1)
    opt = torch.optim.Adam([v0], lr=1)
    
    K = torch.exp(-cost_matrix/0.01)
    for k in tqdm(range(200)):
        
        opt.zero_grad()
        v = predictor(softmax(v0.unsqueeze(0)), softmax(v0.unsqueeze(0)))
        v = torch.exp(v)
        for _ in range(1):
            u = softmax(v0.unsqueeze(0)) / (K @ v.squeeze())
            v = softmax(v0.unsqueeze(0)) / (K.T @ u.squeeze())
        f_mu = torch.log(v) * 0.01

        for i in range(mus.shape[0]):
            v = predictor(mus[i].unsqueeze(0), softmax(v0.unsqueeze(0)))
            v = torch.exp(v)
            for _ in range(1):
                u = mus[i] / (K @ v.squeeze())
                v = softmax(v0.unsqueeze(0)) / (K.T @ u.squeeze())
            f = torch.log(u) * 0.01
            g = torch.log(v) * 0.01

            if v0.grad is not None:
                v0.grad += weights[i] * (g.flatten() - f_mu.flatten())
            else:
                v0.grad = weights[i] * (g.flatten() - f_mu.flatten())
            
       
        opt.step()
    return softmax(v0.unsqueeze(0))


def mccann_interpolation_bilinear(P, grid_points, t, N):
    """
    Returns a (N x N) array representing μ_t, the McCann interpolation at time t,
    using bilinear distribution of mass to avoid holes/artifacts.
    """
    mu_t = np.zeros((N, N), dtype=float)

    for i in range(N*N):
        mass_from_i = P[i]
        if not np.any(mass_from_i):
            continue

        x_i, y_i = grid_points[i]

        for j in range(N*N):
            m_ij = mass_from_i[j]
            if m_ij == 0:
                continue

            x_j, y_j = grid_points[j]

            # Continuous interpolation
            x_t = (1 - t)*x_i + t*x_j
            y_t = (1 - t)*y_i + t*y_j

            # Map to [0, N-1] space
            X = x_t * (N - 1)
            Y = y_t * (N - 1)

            # integer coords
            iX = int(np.floor(X))
            iY = int(np.floor(Y))

            # fractional offsets
            alpha_x = X - iX
            alpha_y = Y - iY

            # Scatter to up to 4 pixels
            for dy in [0, 1]:
                for dx in [0, 1]:
                    rr = iY + dy
                    cc = iX + dx
                    if 0 <= rr < N and 0 <= cc < N:
                        weight = ((alpha_x if dx == 1 else (1 - alpha_x)) *
                                  (alpha_y if dy == 1 else (1 - alpha_y)))
                        mu_t[rr, cc] += m_ij * weight

    return mu_t


def geodesic_interpolation(X, t):
    pi_t = (1 - t) * X[:, None, :] + t * X[None, :, :]  # (n, m, d)
    return pi_t.reshape(-1, 2)


def mccann_network(predictor, mu, nu, cost_matrix,  device='cpu'):
    f = predictor(mu, nu)
    v = torch.exp(f).flatten()
    K = torch.exp(-cost_matrix.to(device)/0.01)

    u = mu / (K @ v)
    v = nu / (K.T @ u)
    G = torch.diag(u)@K@torch.diag(v)

    length = 28
    X,_ = torch.tensor(cost.get_point_cloud(length), dtype=torch.float32), torch.tensor(cost.get_point_cloud(length), dtype=torch.float32)
    X = X.to(device)
    
    plan_geodesic = []
    for i in range(0, 5):
        t = i / 4

        # Interpolierte Punkte berechnen
        mu_t = mccann_interpolation_bilinear(G.cpu().detach().numpy(), X.cpu().detach().numpy(), t, 28)
        plan_geodesic.append(mu_t)
    
    return plan_geodesic


def barycenter_geodesic(predictor, mu, nu, cost_matrix, device='cpu'):
    BAR = torch.stack((mu,nu))
    bar_geodesic = []
    weights_list = [[1,0], [0.75, 0.25], [0.5, 0.5], [0.25, 0.75], [0,1]]

    for weights in weights_list:
        bar_geodesic.append( barycenter(predictor, BAR, cost_matrix.to(device), torch.tensor(weights).to(device)))
    
    return bar_geodesic

def true_mccann_geodesic(mu, nu, cost_matrix, device='cpu'):
    length = 28
    X,_ = torch.tensor(cost.get_point_cloud(length), dtype=torch.float32), torch.tensor(cost.get_point_cloud(length), dtype=torch.float32)

    G = ot.bregman.sinkhorn(mu.cpu().detach().numpy(), nu.cpu().detach().numpy(), cost_matrix.cpu().numpy() ,reg = 0.01)
    true_geodesic = []
    
    for i in range(0, 5):
        t = i / 4

        # Interpolierte Punkte berechnen
        mu_t = mccann_interpolation_bilinear(G, X.cpu().detach().numpy(), t, 28)
        true_geodesic.append(mu_t)
    
    return true_geodesic

def geonet_geodesic():
    geonet0 = torch.tensor(np.array(Image.open("Data/geonet/geonet_0.png"))) 
    geonet1 = torch.tensor(np.array(Image.open("Data/geonet/geonet_2.png")))
    geonet2 = torch.tensor(np.array(Image.open("Data/geonet/geonet_3.png")))
    geonet3 = torch.tensor(np.array(Image.open("Data/geonet/geonet_4.png")))
    geonet4 = torch.tensor(np.array(Image.open("Data/geonet/geonet_5.png")))

    from torch.functional import F
    length = 28

    geonet0 = geonet0.sum(-1).unsqueeze(0).unsqueeze(0).float()  # Add batch and channel dimensions
    geonet0 = F.interpolate(geonet0, size=(length, length), mode='bilinear')

    geonet1 = geonet1.sum(-1).unsqueeze(0).unsqueeze(0).float()  # Add batch and channel dimensions
    geonet1 = F.interpolate(geonet1, size=(length, length), mode='bilinear')
    geonet2 = geonet2.sum(-1).unsqueeze(0).unsqueeze(0).float()  # Add batch and channel dimensions
    geonet2 = F.interpolate(geonet2, size=(length, length), mode='bilinear')
    geonet3 = geonet3.sum(-1).unsqueeze(0).unsqueeze(0).float()  # Add batch and channel dimensions
    geonet3 = F.interpolate(geonet3, size=(length, length), mode='bilinear')
    geonet4 = geonet4.sum(-1).unsqueeze(0).unsqueeze(0).float()  # Add batch and channel dimensions
    geonet4 = F.interpolate(geonet4, size=(length, length), mode='bilinear')

    geonet = [geonet0, geonet1, geonet2, geonet3, geonet4]

    return geonet


def plot_geodesics(predictor, device):
    dimension = 28
    length = dimension

    mu, nu = import_measures(device)
    cost_matrix = cost.get_cost_matrix(dimension, device)

    true_geodesic = true_mccann_geodesic(mu, nu, cost_matrix, device)
    plan_geodesic = mccann_network(predictor, mu, nu, cost_matrix, device)
    bar_geodesic = barycenter_geodesic(predictor, mu, nu, cost_matrix, device)
    geonet = geonet_geodesic()

    fig, axs = plt.subplots(4, 6, figsize=(12, 9))

    # Add the left-side labels
    labels = ['True McCann', 'UNOT McCann\n (Ours)', 'Barycenter\nGeodesic (Ours)', 'GeONet']
    for i, label in enumerate(labels):
        axs[i, 0].text(
            -0.2, 0.5, label, va='center', ha='center', rotation=90, fontsize=18, transform=axs[i, 0].transAxes
        )

    # Time labels for the columns
    time_labels = ['$t=0$', 't=0.25', 't=0.5', 't=0.75', 't=1', 'Ground Truth']
    for i, label in enumerate(time_labels):
        axs[3, i].text(15,35,label, fontsize=22, ha='center')  # Set labels below each column

    # Plot True Geodesic
    for i, x in enumerate(true_geodesic):
        axs[0, i].axis('off')  # Hide axes
        axs[0, i].imshow(
            x, 
            cmap='gray', 
        )

    for i in range(4):
        axs[i, -1].axis('off')  # Hide axes
        axs[i, -1].imshow(
            nu.detach().cpu().numpy().reshape(length, length),
            cmap='gray', 
        )
    # Plot Plan Geodesic
    for i, x in enumerate(plan_geodesic):
        axs[1, i].axis('off')  # Hide axes
        axs[1, i].imshow(
            x, 
            cmap='gray', 
        )

    # Plot Barycenter Geodesic
    for i, x in enumerate(bar_geodesic):
        axs[2, i].axis('off')  # Hide axes
        axs[2, i].imshow(x.cpu().detach().numpy().reshape(length, length), cmap='gray')

    # Plot Geonet
    for i, x in enumerate(geonet):
        axs[3, i].axis('off')  # Hide axes
        axs[3, i].imshow(x.cpu().detach().numpy().reshape(length, length), cmap='gray')

    # Adjust the layout
    plt.tight_layout(pad=0.6, w_pad=0.2, h_pad=0.2)

    plt.savefig('Images/geodesic_comparison.pdf')
    plt.show()

def parse_args():
    parser = argparse.ArgumentParser(description='Barycenter shapes')
    parser.add_argument('--model', type=str, default='FNO', help='Model to use')
    parser.add_argument('--dimension', type=int, default=64, help='Dimension of the images')
    return parser.parse_args()


def main():
    if torch.cuda.is_available(): device = torch.device("cuda")
    elif torch.backends.mps.is_available(): device = torch.device("mps")
    else: device = torch.device("cpu")

    args = parse_args()
    predictor = im.load_fno(args.model, device=device)
    plot_geodesics(predictor, device)


if __name__=='__main__':
    # Load the models
    main()