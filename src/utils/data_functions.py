"""
data_functions.py
-------------

Functions to create and process data for training and testing.
"""

import torch
import torch.nn.functional as F
import torchvision
import numpy as np
import os
import urllib
from tqdm import tqdm
from skimage.draw import random_shapes
import src.ot.sinkhorn as sinkhorn
import src.ot.cost_matrix as cost
import src.utils.gaussian_random_field as grf

from tqdm import tqdm
import os
import subprocess
import zipfile
from PIL import Image
import torch
from torchvision import transforms


def test_set_sampler(
        test_set: torch.Tensor,
        n_samples: int
    ) -> torch.Tensor:
    """
    Randomly sample from a data set to create pairs of samples for testing.

    Parameters
    ----------
    test_set : (n_test_set, dim) torch.Tensor
        Test set.
    n_samples : int
        Number of samples.

    Returns
    -------
    test_sample : (n_samples, 2 * dim) torch.Tensor
        Random sample from the test set.
    """

    rand_perm = torch.randperm(test_set.size(0))
    rand_mask_a = rand_perm[:n_samples]
    rand_mask_b = rand_perm[n_samples:2 * n_samples]
    test_sample_a = test_set[rand_mask_a]
    test_sample_b = test_set[rand_mask_b]
    test_sample = torch.cat((test_sample_a, test_sample_b), dim=1)

    return test_sample


def preprocessor(
        dataset: torch.Tensor,
        length: int,
        dust_const: float
    ) -> torch.Tensor:
    """
    Preprocess (resize, normalize, dust) a dataset for training.

    Parameters
    ----------
    dataset : (n, dim) torch.Tensor
        Dataset to be preprocessed.
    length : int
        Length of the side of each image in the dataset.
    dust_const : float
        Constant added to the dataset to avoid zero values (dusting).

    Returns
    -------
    processed_dataset : (n, dim) torch.Tensor
        Preprocessed dataset.
    """

    # reshaping
    processed_dataset = F.interpolate(
        dataset.unsqueeze(1),
        size=(length, length),
        mode='bilinear',
        align_corners=False
    ).squeeze(1)

    # flattening
    processed_dataset = processed_dataset.view(-1, length ** 2)

    # normalizing
    processed_dataset = processed_dataset / processed_dataset.sum(
        dim=1,
        keepdim=True
    )

    # dusting
    processed_dataset = processed_dataset + dust_const

    # normalizing
    processed_dataset = processed_dataset / processed_dataset.sum(
        dim=1,
        keepdim=True
    )

    return processed_dataset


def rand_noise(
        n_samples: int,
        dim: int,
        dust_const: float,
        pairs: bool
    ) -> torch.Tensor:
    """
    Create a batch of random uniform noise distributions with dusting and
    normalization to ensure the samples are positive probability vectors.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    dim : int
        Dimension of the samples.
    dust_const : float
        Constant added to the samples to avoid zero values.
    pairs : bool
        Whether to return a sample of pairs of probability distributions or a
        sample of single probability distributions.

    Returns
    -------
    sample : (n_samples, 2 * dim or dim) torch.Tensor
        Sample of pairs of probability distributions.
    """

    sample_a = torch.rand((n_samples, dim))
    sample_a /= torch.unsqueeze(sample_a.sum(dim=1), 1)
    sample_a = sample_a + dust_const
    sample_a /= torch.unsqueeze(sample_a.sum(dim=1), 1)

    if pairs:
        sample_b = torch.rand((n_samples, dim))
        sample_b /= torch.unsqueeze(sample_b.sum(dim=1), 1)
        sample_b = sample_b + dust_const
        sample_b /= torch.unsqueeze(sample_b.sum(dim=1), 1)
        sample = torch.cat((sample_a, sample_b), dim=1)
        return sample

    else:
        return sample_a


def rand_shapes(
        n_samples: int,
        dim: int,
        dust_const: float,
        pairs: bool
    ) -> torch.Tensor:
    """
    Create a sample of images containing random shapes as pairs of probability
    distributions.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    dim : int
        Dimension of the samples.
    dust_const : float
        Constant added to the samples to avoid zero values.
    pairs : bool
        Whether to return a sample of pairs of probability distributions or a
        sample of single probability distributions.

    Returns
    -------
    sample : (n_samples, 2 * dim or dim) torch.Tensor
        Sample of pairs of probability distributions.
    """

    length = int(dim ** 0.5)
    sample_a = []
    sample_b = []
    for i in range(n_samples):
        image1 = random_shapes(
            (length, length),
            max_shapes=20,
            min_shapes=2,
            min_size=2,
            max_size=12,
            channel_axis=None,
            allow_overlap=True
        )[0]
        image1 = image1.max() - image1
        image1 = image1 / image1.sum()
        image1 = image1 + dust_const
        image1 = image1 / image1.sum()
        sample_a.append(image1.flatten())
        if pairs:
            image2 = random_shapes(
                (length, length),
                max_shapes=8,
                min_shapes=2,
                min_size=2,
                max_size=12,
                channel_axis=None,
                allow_overlap=True
            )[0]
            image2 = image2.max() - image2
            image2 = image2 + dust_const
            image2 = image2 / image2.sum()
            sample_b.append(image2.flatten())

    sample_a = np.array(sample_a)
    sample_a = torch.tensor(sample_a)

    if pairs:
        sample_b = np.array(sample_b)
        sample_b = torch.tensor(sample_b)
        sample = torch.cat((sample_a, sample_b), dim=1)
        return sample
    else:
        return sample_a


def rand_noise_and_shapes(
        n_samples: int,
        dim: int,
        dust_const: float,
        pairs: bool
    ) -> torch.Tensor:
    """
    Generate a data set of pairs of samples of random shapes combined with
    random noise.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    dim : int
        Dimension of the samples.
    dust_const : float
        Constant added to the probability distributions to avoid zero values.
    pairs : bool
        Whether to return a sample of pairs of probability distributions or a
        sample of single probability distributions.

    Returns
    -------
    sample : (n_samples, 2 * dim or dim) torch.Tensor
        Sample of pairs of probability distributions.
    """

    rn = rand_noise(n_samples, dim, dust_const, pairs)
    rs = rand_shapes(n_samples, dim, dust_const, pairs)

    rn_rand_fact = torch.rand((n_samples, 1))
    rs_rand_fact = torch.rand((n_samples, 1))

    sample = rn_rand_fact * rn + rs_rand_fact * rs

    if pairs:
        sample_mu = sample[:, :dim]
        sample_nu = sample[:, dim:]
        sample_mu = sample_mu / torch.unsqueeze(sample_mu.sum(dim=1), 1)
        sample_nu = sample_nu / torch.unsqueeze(sample_nu.sum(dim=1), 1)
        sample = torch.cat((sample_mu, sample_nu), dim=1)
        return sample
    else:
        sample = sample / torch.unsqueeze(sample.sum(dim=1), 1)
        return sample


def get_mnist(
        n_samples: int,
        path: str
    ) -> None:
    """
    Download and save a set of MNIST images as a pytorch tensor in a '.pt' file.

    Parameters
    ----------
    n_samples : int
        Number of samples from the MNIST dataset.
    path : str
        Path to save the dataset.

    Returns
    -------
    None
    """

    dataset = torchvision.datasets.MNIST(
        root="./data",
        download=True,
        transform=torchvision.transforms.ToTensor()
    )
    mnist = torch.zeros((len(dataset), 28, 28))
    for i, datapoint in enumerate(dataset):
        image = datapoint[0]
        mnist[i] = image
    rand_perm = torch.randperm(len(mnist))
    mnist_save = mnist[rand_perm][:n_samples]
    torch.save(mnist_save, path)
    return None


def get_cifar(
        n_samples: int,
        path: str
    ) -> None:
    """
    Download and save a set of CIFAR10 images as a pytorch tensor in a '.pt'
    file.

    Parameters
    ----------
    n_samples : int
        Number of samples from the CIFAR10 dataset.
    path : str
        Path to save the dataset.

    Returns
    -------
    None
    """

    dataset = torchvision.datasets.CIFAR10(
        root="./data",
        download=True,
        transform=torchvision.transforms.Grayscale()
    )
    transformer = torchvision.transforms.ToTensor()
    cifar = torch.zeros((len(dataset), 32, 32))
    for i, datapoint in enumerate(dataset):
        image = transformer(datapoint[0])
        cifar[i] = image
    rand_perm = torch.randperm(len(cifar))
    cifar_save = cifar[rand_perm][:n_samples]
    torch.save(cifar_save, path)

    return None


def get_chinese_mnist(
        n_samples: int,
        path: str
    ) -> None:
    """
    Download and save a set of chinese MNIST images as a pytorch tensor in a '.pt'
    file.

    Parameters
    ----------
    n_samples : int
        Number of samples from the chinese MNIST dataset.
    path : str
        Path to save the dataset.

    Returns
    -------
    None
    """

    dataset = torchvision.datasets.ImageFolder(
        root=path,
        transform=torchvision.transforms.Grayscale()
    )
    transformer = torchvision.transforms.ToTensor()
    c_mnist = torch.zeros((len(dataset), 64, 64))
    for i, datapoint in enumerate(dataset):
        image = transformer(datapoint[0])
        c_mnist[i] = image
    rand_perm = torch.randperm(len(c_mnist))
    c_mnist = c_mnist[rand_perm][:n_samples]
    print(c_mnist.size())
    torch.save(c_mnist, 'Data/chinese_mnist.pt')

    return None


def get_lfw(
        n_samples: int,
        path: str
    ) -> None:
    """
    Download and save a set of LFW images as a pytorch tensor in a '.pt'
    file.

    Parameters
    ----------
    n_samples : int
        Number of samples from the LFW dataset.
    path : str
        Path to save the dataset.

    Returns
    -------
    None
    """

    dataset = torchvision.datasets.LFWPeople(
        root="./data",
        download=True,
        transform=torchvision.transforms.Grayscale()
    )
    transformer = torchvision.transforms.ToTensor()
    lfw = torch.zeros((len(dataset), 250, 250))

    for i, datapoint in enumerate(dataset):
        image = transformer(datapoint[0])
        lfw[i] = image

    rand_perm = torch.randperm(len(lfw))
    lfw_save = lfw[rand_perm][:n_samples]
    torch.save(lfw_save, path)
    return None


def get_quickdraw(
        n_samples: int,
        root_np: str,
        path_torch: str,
        class_name: str
    ) -> None:
    """
    Download and save a set of Quickdraw images of a specified class as a
    pytorch tensor in a '.pt' file using an intermediary numpy array and file.

    Parameters
    ----------
    n_samples : int
        Number of samples from the Quickdraw dataset.
    root_np : str
        Path to folder to save the numpy array.
    path_torch : str
        Path to save the pytorch tensor.
    class_name : str
        Name of the class of images to download.

    Returns
    -------
    None
    """

    # Create directory if it does not exist
    if not os.path.exists(root_np):
        os.makedirs(root_np)

    # Define class-specific URL and filename
    url = f"https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{class_name}.npy"
    filename = os.path.join(root_np, f"{class_name}.npy")

    # Download the dataset file
    urllib.request.urlretrieve(url, filename)

    # Replace spaces in class name with underscores
    class_name = class_name.replace(' ', '_')
    filename = os.path.join(root_np, f"{class_name}.npy")

    # Load numpy array and convert to tensor
    data_array = np.load(filename)
    dataset = torch.from_numpy(data_array).float()

    # Concatenate tensors along the first dimension
    dataset = dataset.reshape(-1, 28, 28)

    rand_perm = torch.randperm(len(dataset))
    dataset = dataset[rand_perm][:n_samples]

    torch.save(dataset, path_torch)

    return None


def get_quickdraw_class_names() -> list:
    """
    Get the list of class names for the Quickdraw dataset.

    Parameters
    ----------
    None

    Returns
    -------
    class_names : list
        List of class names.
    """

    url = "https://raw.githubusercontent.com/googlecreativelab/quickdraw-dataset/master/categories.txt"
    response = urllib.request.urlopen(url)

    class_names = []
    for line in response:
        class_name = line.decode('utf-8').strip()
        class_names.append(class_name)

    return class_names


def get_quickdraw_multi(
        n_samples: int,
        n_classes: int,
        root_np: str,
        path_torch: str
    ) -> None:
    """
    Download and save a set of Quickdraw images from a specified number of
    random classes as a pytorch tensor in a '.pt' file using intermediary numpy
    arrays and files.

    WARNING: SLOWWWW!

    Parameters
    ----------
    n_samples : int
        Number of samples from the Quickdraw dataset.
    n_classes : int
        Number of random classes to use.
    root_np : str
        Path to folder to save the numpy arrays.
    path_torch : str
        Path to save the pytorch tensor.

    Returns
    -------
    None
    """

    datasets = []

    class_names = get_quickdraw_class_names()

    rand_mask = np.random.choice(len(class_names), n_classes, replace=False)

    class_names = np.array(class_names)[rand_mask]

    n_samples_class = n_samples // n_classes

    for class_name in tqdm(class_names):

        # if class_name is two words, replace space with %20
        if ' ' in class_name:
            class_name = class_name.replace(' ', '%20')

        # Create directory if it does not exist
        if not os.path.exists(root_np):
            os.makedirs(root_np)

        # Define class-specific URL and filename
        url = f"https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{class_name}.npy"
        filename = os.path.join(root_np, f"{class_name}.npy")

        # Download the dataset file
        urllib.request.urlretrieve(url, filename)

        # Replace spaces in class name with underscores
        class_name = class_name.replace(' ', '_')
        filename = os.path.join(root_np, f"{class_name}.npy")

        # Load numpy array and convert to tensor
        data_array = np.load(filename)
        dataset = torch.from_numpy(data_array).float()

        # Concatenate tensors along the first dimension
        dataset = dataset.reshape(-1, 28, 28)

        rand_perm = torch.randperm(len(dataset))
        dataset = dataset[rand_perm][:n_samples_class]

        datasets.append(dataset)

    dataset = torch.cat(datasets, dim=0)

    rand_perm = torch.randperm(len(dataset))
    dataset = dataset[rand_perm][:n_samples]

    torch.save(dataset, path_torch)

    return None


def load_and_preprocess(
        measure,
        length,
        dust_const
    ):
    data_paths = {
        'mnist': 'data/mnist.pt',
        'cifar': 'data/cifar.pt',
        'lfw': 'data/lfw.pt',
        'bear': 'data/bear.pt',
        'quickdraw': 'data/quickdraw.pt',
        'facialexpression': 'data/facialexpression.pt',
        'car': 'data/car.pt',
    }
    
    if measure in data_paths:
        data = torch.load(data_paths[measure], weights_only=True)
        processed_data = preprocessor(data, length, dust_const).numpy()
        
        return processed_data
    
    raise ValueError('Measure not found.')


def random_set_measures(
        first_measure: str,
        second_measure: str,
        number_of_samples: int,
        length: int = 28,
        dust_const: int = 1e-4
    ) -> (torch.Tensor, torch.Tensor):
    '''
    Get the measures from the data functions.

    Parameters
    ----------
    first_measure : str
        First probability the measures.
    second_measure : str
        Second probability the measures.
    number_of_samples : int
        Number of samples in the measures.

    Returns
    -------
    set_mu : (dim, number_of_samples) torch.Tensor
        First probability measure.
    set_nu : (dim, number_of_samples) torch.Tensor
        Second probability measure.
    '''
    first_measure = load_and_preprocess(first_measure, length, dust_const)
    second_measure = load_and_preprocess(second_measure, length, dust_const)

    random_indices_first = np.random.choice(
        first_measure.shape[0],
        number_of_samples,
        replace=True
    )
    random_indices_second = np.random.choice(
        second_measure.shape[0],
        number_of_samples,
        replace=True
    )
    
    return torch.tensor(first_measure[random_indices_first]), torch.tensor(second_measure[random_indices_second])


def create_test_set(
        num_elements,
        length,
        dust_const,
        epsilon,
        device
    ):
    with torch.no_grad():
        mnist = torch.load('Data/mnist.pt', weights_only=True)
        lfw = torch.load('Data/lfw.pt', weights_only=True)
        cifar = torch.load('Data/cifar.pt', weights_only=True)
        bear = torch.load('Data/bear.pt', weights_only=True)

    mnist = preprocessor(mnist, length, dust_const)[0:num_elements + 1].to(device)
    lfw = preprocessor(lfw, length, dust_const)[0:num_elements + 1].to(device)
    cifar = preprocessor(cifar, length, dust_const)[0:num_elements + 1].to(device)
    bear = preprocessor(bear, length, dust_const)[0:num_elements + 1].to(device)
    cost_matrix = cost.get_cost_matrix(length, device)

    def distance(
        mu,
        nu
    ):
        test_data_distance = []
        
        for i in tqdm(range(mu.shape[0])):
            _, _, _, dist1000 = sinkhorn.sink(
                mu[i],
                nu[i],
                cost_matrix,
                epsilon,
                torch.ones_like(mu[0]),
                1000
            )
            test_data_distance.append(dist1000)
        
        return torch.stack(test_data_distance, dim=0).to('cpu')
    
    test_set_mnist = distance(
        mnist[0:num_elements],
        mnist[1:num_elements + 1]
    )
    test_set_lfw = distance(
        lfw[0:num_elements],
        lfw[1:num_elements + 1]
    )
    test_set_cifar = distance(
        cifar[0:num_elements],
        cifar[1:num_elements + 1]
    )
    test_set_bear = distance(
        bear[0:num_elements],
        bear[1:num_elements + 1]
    )
    test_set_bear_lfw = distance(
        bear[0:num_elements],
        lfw[1:num_elements + 1]
    )
    test_set_mnist_cifar = distance(
        mnist[0:num_elements],
        cifar[1:num_elements + 1]
    )

    dist_mnist = {
        'mu': mnist[0:num_elements].to('cpu'),
        'nu': mnist[1:num_elements + 1].to('cpu'),
        'dist': test_set_mnist
    }
    dist_lfw = {
        'mu': lfw[0:num_elements].to('cpu'),
        'nu': lfw[1:num_elements + 1].to('cpu'),
        'dist': test_set_lfw
    }
    dist_cifar = {
        'mu': cifar[0:num_elements].to('cpu'),
        'nu': cifar[1:num_elements + 1].to('cpu'),
        'dist': test_set_cifar
    }
    dist_bear = {
        'mu': bear[0:num_elements].to('cpu'),
        'nu': bear[1:num_elements + 1].to('cpu'),
        'dist': test_set_bear
    }
    dist_bear_lfw = {
        'mu': bear[0:num_elements].to('cpu'),
        'nu': lfw[1:num_elements + 1].to('cpu'),
        'dist': test_set_bear_lfw
    }
    dist_mnist_cifar = {
        'mu': mnist[0:num_elements].to('cpu'),
        'nu': cifar[1:num_elements + 1].to('cpu'),
        'dist': test_set_mnist_cifar
    }

    test_set = {
        'mnist': dist_mnist,
        'lfw': dist_lfw,
        'cifar': dist_cifar,
        'bear': dist_bear,
        'bear_lfw': dist_bear_lfw,
        'mnist_cifar': dist_mnist_cifar
    }
    
    torch.save(test_set, f'Data/test_set__dim_{length}__eps_{epsilon}.pt')


def create_data_set(
        num_elements,
        length,
        dust_const,
        epsilon,
        number_of_samples,
        device
    ):
    with torch.no_grad():
        mnist = torch.load('Data/mnist.pt', weights_only=True)
        lfw = torch.load('Data/lfw.pt', weights_only=True)
        cifar = torch.load('Data/cifar.pt', weights_only=True)
        bear = torch.load('Data/bear.pt', weights_only=True)

    mnist = preprocessor(mnist, length, dust_const)[1000: num_elements + 1000].to(device)
    lfw = preprocessor(lfw, length, dust_const)[1000: num_elements + 1000].to(device)
    cifar = preprocessor(cifar, length, dust_const)[1000: num_elements + 1000].to(device)
    bear = preprocessor(bear, length, dust_const)[1000: num_elements + 1000].to(device)
    cost_matrix = cost.get_cost_matrix(length, device)

    def get_g(
        mu,
        nu,
        n
    ):
        test_data_distance = []
        rand_indices1 = torch.randint(0, mu.shape[0], (n,))
        rand_indices2 = torch.randint(0, mu.shape[0], (n,))
        _, v = sinkhorn.sink_vec(
            mu[rand_indices1],
            nu[rand_indices2],
            cost_matrix,
            epsilon,
            torch.ones_like(mu[rand_indices1]),
            50
        )
        
        data = torch.stack(
            (mu[rand_indices1], nu[rand_indices2], torch.log(v)),
            dim=1
        ).to('cpu')
        # Remove samples with NaN values
        valid_samples = ~torch.isnan(data).any(dim=(1, 2))
        data = data[valid_samples].reshape(-1, 3, length ** 2)
        return data
    
    print('MNIST')
    test_set_mnist = get_g(
        mnist[0:num_elements],
        mnist[1:num_elements + 1],
        number_of_samples
    )
    print('LFW')
    test_set_lfw = get_g(
        lfw[0:num_elements],
        lfw[1:num_elements + 1],
        number_of_samples
    )
    print('CIFAR')
    test_set_cifar = get_g(
        cifar[0:num_elements],
        cifar[1:num_elements + 1],
        number_of_samples
    )
    print('Bear')
    test_set_bear = get_g(
        bear[0:num_elements],
        bear[1:num_elements + 1],
        number_of_samples
    )
    print('Bear LFW')
    test_set_bear_lfw = get_g(
        bear[0:num_elements],
        lfw[1:num_elements + 1],
        number_of_samples
    )
    print('MNIST CIFAR')
    test_set_mnist_cifar = get_g(
        mnist[0:num_elements],
        cifar[1:num_elements + 1],
        number_of_samples
    )
    print(test_set_mnist.shape)
    test_set = torch.cat(
        (
            test_set_mnist,
            test_set_lfw,
            test_set_cifar,
            test_set_bear,
            test_set_bear_lfw,
            test_set_mnist_cifar
        ),
        dim=0
    )
    print(test_set.shape)
    torch.save(test_set, f'Data/data_set__dim_{length}__eps_{epsilon}.pt')


def create_data_set_grf(
        num_samples,
        num_elements,
        length,
        dust_const,
        epsilon,
        sinkhorn_iter,
        incl_random_shapes: bool = False,
        sampling=False,
        device: str = 'cpu'
    ):
    cost_matrix = cost.get_cost_matrix(length, device)

    def get_g_rand(
        mu,
        nu,
        n
    ):
        rand_indices1 = torch.randint(0, mu.shape[0], (n,))
        rand_indices2 = torch.randint(0, mu.shape[0], (n,))
        _, v = sinkhorn.sink_vec(
            mu[rand_indices1],
            nu[rand_indices2],
            cost_matrix,
            epsilon,
            torch.ones_like(mu[rand_indices1]),
            sinkhorn_iter
        )
        
        data = torch.stack(
            (mu[rand_indices1], nu[rand_indices2], torch.log(v)),
            dim=1
        ).to('cpu')
        # Remove samples with NaN values
        valid_samples = ~torch.isnan(data).any(dim=(1, 2)) & ~torch.isinf(data).any(dim=1).any(dim=1)
        data = data[valid_samples].reshape(-1, 3, length ** 2)
        return data
    
    
    def get_g(
        mu,
        nu
    ):
        _, v = sinkhorn.sink_vec(
            mu,
            nu,
            cost_matrix,
            epsilon,
            torch.ones_like(mu),
            sinkhorn_iter
        )
        
        data = torch.stack(
            (mu, nu, torch.log(v)),
            dim=1
        ).to('cpu')
        # Remove samples with NaN values
        valid_samples = ~torch.isnan(data).any(dim=(1, 2)) & ~torch.isinf(data).any(dim=1).any(dim=1)
        data = data[valid_samples].reshape(-1, 3, length ** 2)
        return data
    
    Mu = []
    Nu = []
    for i in range(num_samples // 2):
        mu = grf.gaussian_random_field(
            alpha=5,
            size=length,
            flag_normalize=False
        ).to(device)
        nu = grf.gaussian_random_field(
            alpha=5,
            size=length,
            flag_normalize=False
        ).to(device)
        Mu.append(mu)
        Mu.append(mu ** 2)
        Nu.append(nu)
        Nu.append(nu ** 2)
    for i in range(num_samples // 2):
        mu = grf.gaussian_random_field(
            alpha=3,
            size=length,
            flag_normalize=False
        ).to(device)
        nu = grf.gaussian_random_field(
            alpha=3,
            size=length,
            flag_normalize=False
        ).to(device)
        Mu.append(mu)
        Mu.append(mu ** 2)
        Nu.append(nu)
        Nu.append(nu ** 2)
    for i in range(num_samples // 2):
        mu = grf.gaussian_random_field(
            alpha=10,
            size=length,
            flag_normalize=False
        ).to(device)
        nu = grf.gaussian_random_field(
            alpha=10,
            size=length,
            flag_normalize=False
        ).to(device)
        Mu.append(mu)
        Mu.append(mu ** 2)
        Nu.append(nu)
        Nu.append(nu ** 2)

    if incl_random_shapes:
        mu = rand_shapes(
            num_samples // 2,
            length ** 2,
            dust_const,
            pairs=False
        ).float().reshape(-1, length, length).to(device)
        nu = rand_shapes(
            num_samples // 2,
            length ** 2,
            dust_const,
            pairs=False
        ).float().reshape(-1, length, length).to(device)
        for x in mu:
            mu_grf = grf.gaussian_random_field(
                alpha=5,
                size=length,
                flag_normalize=False
            ).to(device)
            Mu.append(x)
            Mu.append(x * mu_grf)
        for x in nu:
            nu_grf = grf.gaussian_random_field(
                alpha=5,
                size=length,
                flag_normalize=False
            ).to(device)
            Nu.append(x)
            Nu.append(x * nu_grf)

    Mu = torch.stack(Mu).reshape(-1, length ** 2)
    Nu = torch.stack(Nu).reshape(-1, length ** 2)
    Mu -= Mu.min(1)[0].unsqueeze(1)
    Nu -= Nu.min(1)[0].unsqueeze(1)
    Mu /= Mu.sum(dim=1, keepdim=True)
    Nu /= Nu.sum(dim=1, keepdim=True)
    Mu += dust_const
    Nu += dust_const

    Mu /= Mu.sum(dim=1, keepdim=True)
    Nu /= Nu.sum(dim=1, keepdim=True)

    test_set_rand = get_g_rand(Mu, Nu, num_elements)
    test_set = get_g(Mu, Nu)
    
    test_set = torch.cat((test_set_rand, test_set), dim=0)

    if sampling:
        return test_set
    print('Number of samples:', test_set.shape[0])
    torch.save(test_set, f'Data/data_set_grf_dim_{length}_eps_{epsilon}_rand_shapes_{incl_random_shapes}.pt')

def get_facial_expression(n_samples: int, out_path: str) -> None:

        data_dir = 'data'
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        zip_path = os.path.join(data_dir, 'facial_expression.zip')
        # Download the dataset using Kaggle API
        subprocess.run([
            'kaggle', 'datasets', 'download', '-d', 'mhantor/facial-expression', '-p', data_dir
        ], check=True)

        # Unzip the file into a subfolder
        extract_path = os.path.join(data_dir, 'facial_expression')
        if not os.path.exists(extract_path):
            os.makedirs(extract_path)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_path)

        # Collect image files from the extracted folder
        image_files = []
        for root, _, files in os.walk(extract_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_files.append(os.path.join(root, file))

        image_files = sorted(image_files)
        if n_samples > len(image_files):
            n_samples = len(image_files)

        # Define transformation: convert to grayscale, resize, then to tensor
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((48, 48)),
            transforms.ToTensor()
        ])

        images = []
        for file in image_files[:n_samples]:
            img = Image.open(file)
            img = transform(img)
            images.append(img)
        dataset = torch.stack(images, dim=0)

        torch.save(dataset, out_path)