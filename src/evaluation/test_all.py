"""
test_all.py
-----------

This module contains the function to compare barycenters computed with pot and neural networks.
Run with python -m src.evaluation.test_all

"""

import torch
import math
import time
import argparse
import os
import numpy as np
from tqdm import tqdm

import jax.numpy as jnp
from ott.geometry.geometry import Geometry
from ott.problems.linear import linear_problem
from ott.initializers.linear import initializers
from ott.geometry import pointcloud

import src.utils.data_functions as df
import src.evaluation.plot as pa
import src.ot.sinkhorn as sink
import src.ot.cost_matrix as cost
from .import_models import load_fno, load_fno_var_epsilon, load_mlp


def set_evaluation(
    set_mu: torch.Tensor,
    set_nu: torch.Tensor,
    cost_matrix: torch.tensor,
    sinkhorn: callable,
    predictor,
    error_bound: float = 0.01,
    epsilon: float = 1e-2,
    max_time_s: float = 0.05,
    length: int = 28,
):
    dict_relative_error = {"predicted": [], "ones": [], "gauss": []}
    dict_iteration = {"predicted": [], "ones": [], "gauss": []}
    dict_time = {"predicted": [], "ones": [], "gauss": []}
    set_violation = {"predicted": [], "ones": [], "gauss": []}
    dict_relative_error_iter = {"predicted": [], "ones": [], "gauss": []}
    dict_violation_iter = {"predicted": [], "ones": [], "gauss": []}

    number_samples = set_mu.shape[0]
    ones = torch.ones_like(set_mu)

    geom = get_geom(length, epsilon)
    mu_jnp = jnp.array(set_mu.cpu().numpy())
    nu_jnp = jnp.array(set_nu.cpu().numpy())
    u0_gauss = [
        get_gauss_init(geom, mu_jnp[i], nu_jnp[i]) for i in range(number_samples)
    ]
    u0_gauss = torch.stack(u0_gauss).to(set_mu.device)

    v_0 = torch.exp(predictor(set_mu, set_nu)).reshape(-1, length * length)

    with torch.no_grad():
        relative_error_dim = None  # relative_error_over_dimension(set_mu, set_nu, predictor, epsilon, 10, 64)

        _, _, distance_1000_step = sink.sink_vec_dist(
            set_mu, set_nu, cost_matrix, epsilon, ones, 1000
        )
        _, _, distance_ones = sink.sink_vec_dist(
            set_mu, set_nu, cost_matrix, epsilon, ones, 1
        )
        _, _, distance_predicted = sink.sink_vec_dist(
            set_mu, set_nu, cost_matrix, epsilon, v_0, 1
        )
        _, _, distance_gauss = sink_vec_dist_gauss(
            set_mu, set_nu, cost_matrix, epsilon, u0_gauss, 1
        )

        time_predicted = sinkhorn_computation_predictor(
            set_mu,
            set_nu,
            cost_matrix,
            epsilon,
            distance_1000_step,
            predictor,
            max_time_s=max_time_s,
        )
        time_ones = sinkhorn_computation_time(
            set_mu,
            set_nu,
            cost_matrix,
            epsilon,
            distance_1000_step,
            max_time_s=max_time_s,
        )
        time_gauss = sinkhorn_computation_time_gauss(
            set_mu,
            set_nu,
            cost_matrix,
            epsilon,
            distance_1000_step,
            u0_gauss,
            max_time_s=max_time_s,
        )

        k_ones = sinkhorn_iteration_until_error(
            set_mu,
            set_nu,
            cost_matrix,
            0.01,
            ones,
            1000,
            distance_1000_step,
            error_bound,
        )
        k_predicted = sinkhorn_iteration_until_error(
            set_mu,
            set_nu,
            cost_matrix,
            0.01,
            v_0,
            1000,
            distance_1000_step,
            error_bound,
        )
        k_gauss = sinkhorn_iteration_until_error(
            set_mu,
            set_nu,
            cost_matrix,
            0.01,
            u0_gauss,
            1000,
            distance_1000_step,
            error_bound,
        )

        violation_ones = sinkhorn_constraint_violation(
            set_mu, set_nu, cost_matrix, epsilon, ones
        )
        violation_predicted = sinkhorn_constraint_violation(
            set_mu, set_nu, cost_matrix, epsilon, v_0
        )
        violation_gauss = sinkhorn_constraint_violation_gauss(
            set_mu, set_nu, cost_matrix, epsilon, u0_gauss
        )

        relativ_error_iter_ones = sinkhorn_error_iteration(
            set_mu, set_nu, cost_matrix, epsilon, ones, 50, distance_1000_step
        )
        relativ_error_iter_predicted = sinkhorn_error_iteration(
            set_mu, set_nu, cost_matrix, epsilon, v_0, 50, distance_1000_step
        )
        relativ_error_iter_gauss = sinkhorn_error_iteration_gauss(
            set_mu, set_nu, cost_matrix, epsilon, u0_gauss, 50, distance_1000_step
        )

        # calulate relative error and append to dictionary
        dict_relative_error["predicted"] = (
            torch.abs(distance_predicted - distance_1000_step) / distance_1000_step
        )
        dict_relative_error["ones"] = (
            torch.abs(distance_ones - distance_1000_step) / distance_1000_step
        )
        dict_relative_error["gauss"] = (
            torch.abs(distance_gauss - distance_1000_step) / distance_1000_step
        )

        dict_time["predicted"] = time_predicted
        dict_time["ones"] = time_ones
        dict_time["gauss"] = time_gauss

        dict_iteration["predicted"] = k_predicted
        dict_iteration["ones"] = k_ones
        dict_iteration["gauss"] = k_gauss

        set_violation["predicted"] = violation_predicted
        set_violation["ones"] = violation_ones
        set_violation["gauss"] = violation_gauss

        dict_relative_error_iter["predicted"] = relativ_error_iter_predicted
        dict_relative_error_iter["ones"] = relativ_error_iter_ones
        dict_relative_error_iter["gauss"] = relativ_error_iter_gauss

        dict_violation_iter["predicted"] = violation_predicted
        dict_violation_iter["ones"] = violation_ones
        dict_violation_iter["gauss"] = violation_gauss

        del (
            v_0,
            distance_1000_step,
            distance_ones,
            distance_predicted,
            time_predicted,
            time_ones,
            k_ones,
            k_predicted,
            u0_gauss,
            distance_gauss,
            time_gauss,
            k_gauss,
            violation_ones,
            violation_predicted,
            violation_gauss,
        )

    return (
        dict_relative_error,
        dict_time,
        dict_iteration,
        set_violation,
        dict_relative_error_iter,
        dict_violation_iter,
        relative_error_dim,
    )


def relative_error_over_dimension(
    set_mu: torch.Tensor,
    set_nu: torch.Tensor,
    predictor,
    epsilon: float = 1e-2,
    min_len: int = 10,
    max_len: int = 64,
):
    dict_relative_error = {"predicted": [], "ones": [], "gauss": []}
    original_set_mu = set_mu
    original_set_nu = set_nu
    original_len = int(math.isqrt(set_mu.shape[-1]))
    number_samples = set_mu.shape[0]

    for length in tqdm(range(min_len, max_len, 2)):

        set_mu = df.preprocessor(
            original_set_mu.reshape(-1, original_len, original_len), length, 1e-6
        ).to(original_set_mu.device)
        set_nu = df.preprocessor(
            original_set_nu.reshape(-1, original_len, original_len), length, 1e-6
        ).to(original_set_nu.device)
        cost_matrix = cost.fast_get_cost_matrix(length, device=original_set_mu.device)
        ones = torch.ones_like(set_mu)
        geom = get_geom(length, epsilon)

        v_0 = torch.exp(predictor(set_mu, set_nu)).reshape(-1, length * length)
        mu_jnp = jnp.array(set_mu.cpu().numpy())
        nu_jnp = jnp.array(set_nu.cpu().numpy())
        u0_gauss = [
            get_gauss_init(geom, mu_jnp[i], nu_jnp[i]) for i in range(number_samples)
        ]
        u0_gauss = torch.stack(u0_gauss).to(set_mu.device)

        with torch.no_grad():

            _, _, distance_1000_step = sink.sink_vec_dist(
                set_mu, set_nu, cost_matrix, epsilon, ones, 1000
            )

            _, _, distance_ones = sink.sink_vec_dist(
                set_mu, set_nu, cost_matrix, epsilon, ones, 1
            )
            _, _, distance_predicted = sink.sink_vec_dist(
                set_mu, set_nu, cost_matrix, epsilon, v_0, 1
            )
            _, _, distance_gauss = sink_vec_dist_gauss(
                set_mu, set_nu, cost_matrix, epsilon, u0_gauss, 1
            )

            dict_relative_error["predicted"].append(
                (
                    torch.abs(distance_predicted - distance_1000_step)
                    / distance_1000_step
                ).cpu()
            )
            dict_relative_error["ones"].append(
                (
                    torch.abs(distance_ones - distance_1000_step) / distance_1000_step
                ).cpu()
            )
            dict_relative_error["gauss"].append(
                (
                    torch.abs(distance_gauss - distance_1000_step) / distance_1000_step
                ).cpu()
            )

    del (
        v_0,
        distance_1000_step,
        distance_ones,
        distance_predicted,
        distance_gauss,
        u0_gauss,
        ones,
    )
    return dict_relative_error


def relative_error_over_dimension_eps(
    set_mu: torch.Tensor,
    set_nu: torch.Tensor,
    predictor,
    epsilon_min: float = 1e-2,
    epsilon_max: float = 1,
    num_epsilons: int = 10,
    min_len: int = 10,
    max_len: int = 64,
):
    original_set_mu = set_mu
    original_set_nu = set_nu
    original_len = int(math.isqrt(set_mu.shape[-1]))
    number_samples = set_mu.shape[0]
    epsilon = torch.logspace(
        math.log10(epsilon_min), math.log10(epsilon_max), num_epsilons
    )

    average = torch.zeros(max_len - min_len, num_epsilons)

    for i, len in tqdm(enumerate(range(min_len, max_len, 1))):
        set_mu = df.preprocessor(
            original_set_mu.reshape(-1, original_len, original_len), len, 1e-6
        ).to(original_set_mu.device)
        set_nu = df.preprocessor(
            original_set_nu.reshape(-1, original_len, original_len), len, 1e-6
        ).to(original_set_nu.device)
        cost_matrix = cost.fast_get_cost_matrix(len, device=original_set_mu.device)
        ones = torch.ones_like(set_mu)
        for j, eps in enumerate(epsilon):
            mu_0 = torch.exp(predictor(set_mu, set_nu, eps)).reshape(-1, len * len)

            with torch.no_grad():
                _, _, distance_1000_step = sink.sink_vec_dist(
                    set_mu, set_nu, cost_matrix, eps, ones, 1000
                )
                _, _, distance_predicted = sink.sink_vec_dist(
                    set_mu, set_nu, cost_matrix, eps, mu_0, 1
                )  # sink_vec_dist_half(set_mu, set_nu, cost_matrix, eps, mu_0,1)

            relat_er = torch.nanmean(
                torch.abs(distance_predicted - distance_1000_step) / distance_1000_step
            )
            if not torch.isnan(relat_er):
                average[i, j] = relat_er
            else:
                print("Error in relative error calculation!!!!")

    return average


def set_ot_distance(
    set_mu: torch.Tensor,
    set_nu: torch.Tensor,
    cost_matrix: torch.tensor,
    sinkhorn: callable,
    predictor,
    epsilon: float = 1e-2,
):
    dict_relative_error = {"predicted": [], "ones": []}
    number_samples = set_mu.shape[0]
    ones = torch.ones_like(set_mu[0])

    for i in tqdm(range(number_samples)):
        with torch.no_grad():
            v_0 = torch.exp(predictor(set_mu[i : i + 1], set_nu[i : i + 1])).flatten()

            _, _, distance_1000_step = sinkhorn(
                set_mu, set_nu, cost_matrix, epsilon, ones, 1000
            )
            _, _, _, distance_ones = sinkhorn(
                set_mu, set_nu, cost_matrix, epsilon, ones, 1
            )
            _, _, _, distance_predicted = sinkhorn(
                set_mu, set_nu, cost_matrix, epsilon, v_0, 1
            )

            # calulate relative error and append to dictionary
            dict_relative_error["predicted"].append(
                torch.abs(distance_predicted - distance_1000_step) / distance_1000_step
            )
            dict_relative_error["ones"].append(
                torch.abs(distance_ones - distance_1000_step) / distance_1000_step
            )
            del v_0, distance_1000_step, distance_ones, distance_predicted

    return dict_relative_error


def sinkhorn_iteration_until_error(
    set_mu: torch.Tensor,
    set_nu: torch.Tensor,
    cost_matrix: torch.Tensor,
    epsilon: float,
    v0: torch.Tensor,
    maxiter: int,
    dist_target: float,
    error_bound: float,
) -> int:
    K = torch.exp(-cost_matrix / epsilon)
    iter = []
    for i in range(set_mu.shape[0]):
        mu = set_mu[i]
        nu = set_nu[i]
        v = v0[i]

        for k in range(maxiter):
            u = mu / (K @ v)
            v = nu / (K.T @ u)

            G = torch.diag(u) @ K @ torch.diag(v)
            dist = torch.trace(cost_matrix.T @ G)
            if torch.abs(dist - dist_target[i]) / dist_target[i] < error_bound:
                iter.append(k)
                break

    return iter


def sinkhorn_error_iteration(
    set_mu: torch.Tensor,
    set_nu: torch.Tensor,
    cost_matrix: torch.Tensor,
    epsilon: float,
    v0: torch.Tensor,
    maxiter: int,
    dist_target: float,
):
    K = torch.exp(-cost_matrix / epsilon)
    V = v0
    relative_error = []
    marginal_constraint_violation = []

    for _ in range(maxiter):
        mcv = torch.zeros(set_mu.shape[0])

        U = set_mu / (K @ V.T).T
        V = set_nu / (K.T @ U.T).T

        for i in range(set_mu.shape[0]):
            u = U[i]
            v = V[i]
            G = torch.diag(u) @ K @ torch.diag(v)
            mcv[i] = (
                torch.norm(torch.sum(G, axis=0) - set_mu[i], p=1)
                + torch.norm(torch.sum(G, axis=1) - set_nu[i], p=1) / 2
            )

        D = K * cost_matrix
        dist = torch.einsum("si,ij,sj->s", U, D, V)

        relative_error.append((torch.abs(dist - dist_target) / dist_target))
        marginal_constraint_violation.append(mcv)
    return relative_error, marginal_constraint_violation


def sinkhorn_error_iteration_gauss(
    set_mu: torch.Tensor,
    set_nu: torch.Tensor,
    cost_matrix: torch.Tensor,
    epsilon: float,
    u0: torch.Tensor,
    maxiter: int,
    dist_target: float,
):
    K = torch.exp(-cost_matrix / epsilon)
    U = u0
    relative_error = []
    marginal_constraint_violation = []

    for _ in range(maxiter):
        mcv = torch.zeros(set_mu.shape[0])

        V = set_nu / (K.T @ U.T).T
        U = set_mu / (K @ V.T).T

        for i in range(set_mu.shape[0]):
            u = U[i]
            v = V[i]
            G = torch.diag(u) @ K @ torch.diag(v)
            mcv[i] = (
                torch.norm(torch.sum(G, axis=0) - set_mu[i], p=1)
                + torch.norm(torch.sum(G, axis=1) - set_nu[i], p=1) / 2
            )
        D = K * cost_matrix
        dist = torch.einsum("si,ij,sj->s", U, D, V)

        relative_error.append((torch.abs(dist - dist_target) / dist_target))
        marginal_constraint_violation.append(mcv)
    return relative_error, marginal_constraint_violation


def set_iteration_until_given_error(
    set_mu: torch.Tensor,
    set_nu: torch.Tensor,
    cost_matrix: torch.tensor,
    sinkhorn: callable,
    predictor,
    epsilon: float = 1e-2,
    error_bound: float = 0.01,
):
    dict_relative_error = {"predicted": [], "ones": []}
    number_samples = set_mu.shape[0]
    ones = torch.ones_like(set_mu[0])

    for i in tqdm(range(number_samples)):
        with torch.no_grad():
            v_0 = torch.exp(predictor(set_mu[i : i + 1], set_nu[i : i + 1])).flatten()
            v_0 = v_0.flatten()

            _, _, _, distance_1000_step = sink.sink(
                set_mu, set_nu, cost_matrix, 0.01, ones, 1000
            )

            k_ones = sinkhorn_iteration_until_error(
                set_mu,
                set_nu,
                cost_matrix,
                0.01,
                ones,
                1000,
                distance_1000_step,
                error_bound,
            )
            k_predicted = sinkhorn_iteration_until_error(
                set_mu,
                set_nu,
                cost_matrix,
                0.01,
                v_0,
                1000,
                distance_1000_step,
                error_bound,
            )

            # calculate relative error and append to dictionary
            dict_relative_error["predicted"].append(k_predicted)
            dict_relative_error["ones"].append(k_ones)

    return dict_relative_error


def set_marginal_constraint_violation(
    set_mu: torch.Tensor,
    set_nu: torch.Tensor,
    cost_matrix: torch.tensor,
    predictor,
    epsilon: float = 1e-2,
):
    dict_violation = {"predicted": [], "ones": []}
    number_samples = set_mu.shape[0]
    ones = torch.ones_like(set_mu[0])
    for i in tqdm(range(number_samples)):
        with torch.no_grad():
            v_0 = torch.exp(predictor(set_mu[i : i + 1], set_nu[i : i + 1])).flatten()
            violation_ones = sinkhorn_constraint_violation(
                set_mu, set_nu, cost_matrix, epsilon, ones
            )
            violation_predicted = sinkhorn_constraint_violation(
                set_mu, set_nu, cost_matrix, epsilon, v_0
            )

            # append to dictionary
            dict_violation["predicted"].append(violation_predicted)
            dict_violation["ones"].append(violation_ones)

    return dict_violation


def sinkhorn_constraint_violation(
    set_mu: torch.Tensor,
    set_nu: torch.Tensor,
    cost_matrix: torch.Tensor,
    epsilon: float,
    v0: torch.Tensor,
) -> float:
    K = torch.exp(-cost_matrix / epsilon)
    constr_violation = []
    for i in range(set_mu.shape[0]):
        mu = set_mu[i]
        nu = set_nu[i]

        v = v0[i]

        u = mu / (K @ v)
        v = nu / (K.T @ u)
        G = torch.diag(u) @ K @ torch.diag(v)
        costraint_violation = torch.norm(torch.sum(G, axis=1) - mu, p=1) + torch.norm(
            torch.sum(G, axis=0) - nu, p=1
        )
        costraint_violation /= 2
        constr_violation.append(costraint_violation.item())

    return constr_violation


def sinkhorn_constraint_violation_gauss(
    set_mu: torch.Tensor,
    set_nu: torch.Tensor,
    cost_matrix: torch.Tensor,
    epsilon: float,
    u0: torch.Tensor,
) -> float:
    K = torch.exp(-cost_matrix / epsilon)
    constr_violation = []
    for i in range(set_mu.shape[0]):
        mu = set_mu[i]
        nu = set_nu[i]

        u = u0[i]
        v = nu / (K.T @ u)
        u = mu / (K @ v)
        G = torch.diag(u) @ K @ torch.diag(v)
        costraint_violation = torch.norm(torch.sum(G, axis=1) - mu, p=1) + torch.norm(
            torch.sum(G, axis=0) - nu, p=1
        )
        costraint_violation /= 2
        constr_violation.append(costraint_violation.item())

    return constr_violation


def set_computation_time(
    set_mu: torch.Tensor,
    set_nu: torch.Tensor,
    cost_matrix: torch.tensor,
    predictor,
    epsilon: float = 1e-2,
    max_time_s: float = 0.01,
):
    dict_time = {"predicted": [], "ones": [], "gauss": []}
    number_samples = set_mu.shape[0]
    ones = torch.ones_like(set_mu[0])
    geom = get_geom(int(torch.sqrt(set_mu.shape[0])), epsilon)

    for i in tqdm(range(number_samples)):
        with torch.no_grad():
            _, _, _, distance_1000_step = sink.sink(
                set_mu, set_nu, cost_matrix, epsilon, ones, 1000
            )

            v0_pred = torch.exp(
                predictor(set_mu[i : i + 1], set_nu[i : i + 1])
            ).flatten()

            mu_jnp = jnp.array(set_mu[i].cpu().numpy())
            nu_jnp = jnp.array(set_nu[i].cpu().numpy())
            u0_gauss = get_gauss_init(geom, mu_jnp, nu_jnp)

            time_predicted = sinkhorn_computation_time(
                set_mu,
                set_nu,
                cost_matrix,
                epsilon,
                distance_1000_step,
                v0_pred,
                max_time_s,
            )
            time_ones = sinkhorn_computation_time(
                set_mu,
                set_nu,
                cost_matrix,
                epsilon,
                distance_1000_step,
                ones,
                max_time_s,
            )
            time_gauss = sinkhorn_computation_time(
                set_mu,
                set_nu,
                cost_matrix,
                epsilon,
                distance_1000_step,
                u0_gauss,
                max_time_s,
            )

            # append to dictionary
            dict_time["predicted"].append(time_predicted)
            dict_time["ones"].append(time_ones)
            dict_time["gauss"].append(time_gauss)

    return dict_time


def sinkhorn_computation_time(
    set_mu: torch.Tensor,
    set_nu: torch.Tensor,
    cost_matrix: torch.Tensor,
    epsilon: float,
    distance_true: float,
    v0=None,
    max_time_s: float = 0.01,
):
    K = torch.exp(-cost_matrix / epsilon)
    u_list = []
    v_list = []
    time_list = []
    relativ_error = []

    start_time = time.time()
    if v0 is not None:
        V = v0
    else:
        V = torch.ones_like(set_mu)

    for _ in range(100):
        U = set_mu / (K @ V.T).T
        V = set_nu / (K.T @ U.T).T
        time_list.append(time.time() - start_time)
        u_list.append(U.clone())
        v_list.append(V.clone())
        start_time = time.time()

        if time_list[-1] > max_time_s:
            break

    D = K * cost_matrix
    for i in range(len(u_list)):
        # Größe: [i, j]
        dist = torch.einsum("si,ij,sj->s", u_list[i], D, v_list[i])
        relativ_error.append(torch.abs(dist - distance_true) / distance_true)

    return (time_list, relativ_error)


def sinkhorn_computation_predictor(
    set_mu: torch.Tensor,
    set_nu: torch.Tensor,
    cost_matrix: torch.Tensor,
    epsilon: float,
    distance_true: float,
    predictor=None,
    max_time_s: float = 0.01,
):
    K = torch.exp(-cost_matrix / epsilon)
    u_list = []
    v_list = []
    time_list = []
    relativ_error = []

    length = int(math.isqrt(set_mu.shape[-1]))

    with torch.no_grad():
        # predictor = torch.compile(predictor)
        V = predictor(set_mu, set_nu)

        start_time = time.time()

        V = torch.exp(predictor(set_mu, set_nu).reshape(-1, length * length)).reshape(
            -1, length * length
        )
    for _ in range(100):
        U = set_mu / (K @ V.T).T
        V = set_nu / (K.T @ U.T).T
        time_list.append(time.time() - start_time)
        u_list.append(U.clone())
        v_list.append(V.clone())
        start_time = time.time()

        if time_list[-1] > max_time_s:
            break

    D = K * cost_matrix
    for i in range(len(u_list)):
        # Größe: [i, j]
        dist = torch.einsum("si,ij,sj->s", u_list[i], D, v_list[i])
        relativ_error.append(torch.abs(dist - distance_true) / distance_true)

    return (time_list, relativ_error)


def sinkhorn_computation_time_gauss(
    set_mu: torch.Tensor,
    set_nu: torch.Tensor,
    cost_matrix: torch.Tensor,
    epsilon: float,
    distance_true: float,
    u0=None,
    max_time_s: float = 0.01,
):
    K = torch.exp(-cost_matrix / epsilon)
    u_list = []
    v_list = []
    time_list = []
    relativ_error = []

    start_time = time.time()
    U = u0

    for _ in range(100):
        V = set_nu / (K.T @ U.T).T
        U = set_mu / (K @ V.T).T
        time_list.append(time.time() - start_time)
        u_list.append(U.clone())
        v_list.append(V.clone())
        start_time = time.time()

        if time_list[-1] > max_time_s:
            break

    D = K * cost_matrix
    for i in range(len(u_list)):
        # Größe: [i, j]
        dist = torch.einsum("si,ij,sj->s", u_list[i], D, v_list[i])
        relativ_error.append(torch.abs(dist - distance_true) / distance_true)

    return (time_list, relativ_error)


def get_geom(n: int, eps: float) -> Geometry:
    """
    Get a geometry object for the optimal transport problem on a 2D grid.

    Parameters
    ----------
    n : int
        Number of points per dimension.
    eps : float
        Regularisation parameter.

    Returns
    -------
    geom : Geometry
        Geometry object for the optimal transport problem on a 2D grid.
    """

    # Generate a 2D grid of n points per dimension
    cloud = cost.get_point_cloud(n)

    geom = pointcloud.PointCloud(cloud, cloud, epsilon=eps)

    return geom


def get_gauss_init(geom: Geometry, mu: jnp.ndarray, nu: jnp.ndarray) -> jnp.ndarray:
    """
    Get a Gaussian initialisation for the dual vector v.

    Parameters
    ----------
    geom : Geometry
        Geometry of the problem.
    mu : jnp.ndarray
        Source distribution.
    nu : jnp.ndarray
        Target distribution.
    eps : float
        Regularisation parameter.

    Returns
    -------
    v : jnp.ndarray
        Gaussian initialisation for the dual vector v.
    """

    init = initializers.GaussianInitializer()
    prob = linear_problem.LinearProblem(geom, mu, nu)
    u = init.init_dual_a(prob, False)

    return torch.tensor(np.array(u))


def sink_vec_dist_gauss(
    MU: torch.Tensor,
    NU: torch.Tensor,
    C: torch.Tensor,
    eps: float,
    U0: torch.Tensor,
    n_iters: int,
) -> torch.Tensor:
    """
    A vectorized version of the Sinkhorn algorithm to create scaling factors
    to be used for generating targets.

    Parameters
    ----------
    MU : (n_samples, dim) torch.Tensor
        First probability distributions.
    NU : (n_samples, dim) torch.Tensor
        Second probability distributions.
    C : (dim, dim) torch.Tensor
        Cost matrix.
    eps : float
        Regularization parameter.
    V0 : (n_samples, dim) torch.Tensor
        Initial guess for scaling factors V.
    n_iters : int
        Maximum number of iterations.

    Returns
    -------
    U : (n_samples, dim) torch.Tensor
        1st Scaling factor.
    V : (n_samples, dim) torch.Tensor
        2nd Scaling factor.
    dim: (n_samples) torch.Tensor
        Distance.
    """
    K = torch.exp(-C / eps)
    U = U0

    V = NU / (K.T @ U.T).T
    U = MU / (K @ V.T).T

    D = K * C  # Größe: [i, j]
    dist = torch.einsum("si,ij,sj->s", U, D, V)  # Größe: [s]
    return U, V, dist


def test_neural_operator(args, device):
    predictor = load_fno(
        args.model, args.modes, args.width, args.activation, args.grid, device=device
    )

    sinkhorn = sink.sink_vec_dist
    approximation = {}
    time = {}
    mcv = {}
    error_iter = {}
    iter = {}
    set_error_dim = {}

    def evaluate_dataset_pair(
        data1, data2, dim, number_of_samples, device, predictor, sinkhorn, cost_matrix
    ):  
        try:
            set_mu, set_nu = df.random_set_measures(data1, data2, number_of_samples, dim)
        except:
            print('Dataset not found')
        
        results = set_evaluation(
            set_mu.to(device),
            set_nu.to(device),
            cost_matrix,
            sinkhorn,
            predictor,
            0.01,
            length=dim,
        )
        return results

    # Process 28x28 datasets
    datasets_28 = [("mnist", "mnist"), ("cifar", "cifar"), ("mnist", "cifar")]

    cost_matrix_28 = cost.get_cost_matrix(28, device=device)

    for data1, data2 in datasets_28:
        key = data1 if data1 == data2 else f"{data1}-{data2}"
        results = evaluate_dataset_pair(
            data1,
            data2,
            28,
            args.number_of_samples,
            device,
            predictor,
            sinkhorn,
            cost_matrix_28,
        )

        approximation[key] = results[0]
        time[key] = results[1]
        iter[key] = results[2]
        mcv[key] = results[3]
        error_iter[key] = results[4]
        print(f"Finished {key}")

    # Process 64x64 datasets
    datasets_64 = [("lfw", "lfw"), ("bear", "bear"), ("lfw", "bear")]

    cost_matrix_64 = cost.get_cost_matrix(64, device=device)

    for data1, data2 in datasets_64:
        key = data1 if data1 == data2 else f"{data1}-{data2}"
        results = evaluate_dataset_pair(
            data1,
            data2,
            64,
            args.number_of_samples,
            device,
            predictor,
            sinkhorn,
            cost_matrix_64,
        )

        approximation[key] = results[0]
        time[key] = results[1]
        iter[key] = results[2]
        mcv[key] = results[3]
        error_iter[key] = results[4]
        print(f"Finished {key}")

        data_set = ["mnist", "cifar", "bear", "lfw", "facialexpression", "car"]

    for i, data1 in enumerate(data_set):
        set_error_dim[data1] = {}

        for data2 in data_set[: i + 1]:
            set_mu, set_nu = df.random_set_measures(
                data1, data2, args.number_of_samples, 64
            )
            error = relative_error_over_dimension(
                set_mu.to(device), set_nu.to(device), predictor, 0.01, 10, 64
            )
            set_error_dim[data1][data2] = error

    pa.plot_error_dim_matrix(data_set, set_error_dim, args.model)

    compelte_dict = {
        "approximation": approximation,
        "marginal_constraint": mcv,
        "time": time,
        "error_iter": error_iter,
        "iteration": iter,
        "dim": (data_set, set_error_dim),
    }

    os.makedirs("experiments", exist_ok=True)
    torch.save(compelte_dict, f"experiments/approximation_NO_{args.model}.pt")

    # print(set_time)
    pa.plot_approximation(approximation, args.model)
    pa.plot_marginal_constraint(mcv, args.model)
    pa.plot_time(time, args.model)
    pa.plot_error_over_iter(error_iter, args.model)

    print("Saved the results in the experiments folder.")


def test_neural_operator_var_eps(args, device):
    predictor = load_fno_var_epsilon(
        args.model, args.modes, args.width, args.activation, args.grid, device
    )

    data_set = ["mnist", "cifar", "bear", "lfw", "facialexpression", "car"]
    set_error_dim = {}
    for i, data1 in enumerate(tqdm(data_set, desc="Outer loop")):
        set_error_dim[data1] = {}

        for data2 in tqdm(data_set[: i + 1], desc=f"Inner loop {data1}"):
            try:
                set_mu, set_nu = df.random_set_measures(
                    data1, data2, args.number_of_samples, 64
                )
            except:
                print('Dataset not found')
            error = relative_error_over_dimension_eps(
                set_mu.to(device),
                set_nu.to(device),
                predictor,
                min_len=10,
                max_len=70,
                num_epsilons=40,
            )
            set_error_dim[data1][data2] = error
            print(f"Error between {data1} and {data2} is finished.")

    torch.save(
        set_error_dim,
        f"experiments/approximation_NO_var_epsilon_{args.model}_{args.number_of_samples}.pt",
    )
    pa.plot_error_dim_eps_matrix(
        data_set,
        set_error_dim,
    )


def test_mlp(args, device):
    predictor = load_fno(
        args.model, args.modes, args.width, args.activation, args.grid, device=device
    )

    sinkhorn = sink.sink_vec_dist
    approximation = {}
    time = {}
    mcv = {}
    error_iter = {}
    iter = {}
    set_error_dim = {}

    def evaluate_dataset_pair(
        data1, data2, dim, number_of_samples, device, predictor, sinkhorn, cost_matrix
    ):
        set_mu, set_nu = df.random_set_measures(data1, data2, number_of_samples, dim)
        results = set_evaluation(
            set_mu.to(device),
            set_nu.to(device),
            cost_matrix,
            sinkhorn,
            predictor,
            0.01,
            length=dim,
        )
        return results

    # Process 28x28 datasets
    datasets = [("mnist", "mnist"), ("cifar", "cifar"), ("mnist", "cifar")]

    cost_matrix = cost.get_cost_matrix(args.dimension, device=device)

    for data1, data2 in datasets:
        key = data1 if data1 == data2 else f"{data1}-{data2}"
        results = evaluate_dataset_pair(
            data1,
            data2,
            args.dimension,
            args.number_of_samples,
            device,
            predictor,
            sinkhorn,
            cost_matrix,
        )

        approximation[key] = results[0]
        time[key] = results[1]
        iter[key] = results[2]
        mcv[key] = results[3]
        error_iter[key] = results[4]
        print(f"Finished {key}")

    # Process 64x64 datasets
    datasets = [("lfw", "lfw"), ("bear", "bear"), ("lfw", "bear")]

    for data1, data2 in datasets:
        key = data1 if data1 == data2 else f"{data1}-{data2}"
        results = evaluate_dataset_pair(
            data1,
            data2,
            args.dimension,
            args.number_of_samples,
            device,
            predictor,
            sinkhorn,
            cost_matrix,
        )

        approximation[key] = results[0]
        time[key] = results[1]
        iter[key] = results[2]
        mcv[key] = results[3]
        error_iter[key] = results[4]
        print(f"Finished {key}")

        data_set = ["mnist", "cifar", "bear", "lfw", "facialexpression", "car"]

    compelte_dict = {
        "approximation": approximation,
        "marginal_constraint": mcv,
        "time": time,
        "error_iter": error_iter,
        "iteration": iter,
        "dim": (data_set, set_error_dim),
    }

    os.makedirs("experiments", exist_ok=True)
    torch.save(compelte_dict, f"experiments/approximation_MLP_{args.model}.pt")

    # print(set_time)
    pa.plot_approximation(approximation, args.model)
    pa.plot_marginal_constraint(mcv, args.model)
    pa.plot_time(time, args.model)
    pa.plot_error_over_iter(error_iter, args.model)

    print("Saved the results in the experiments folder.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the evaluation of the UNOT/MLP model."
    )
    parser.add_argument(
        "--number-of-samples",
        type=int,
        default=500,
        help="Number of samples in the measures.",
    )
    parser.add_argument(
        "--model", type=str, default="predictor_28_v2", help="Name of the model to use."
    )
    parser.add_argument(
        "--network", type=str, default="NO", help="Name of the network to use."
    )

    # Neural Operator arguments
    parser.add_argument(
        "--in-channels", type=int, default=2, help="Number of input channels"
    )
    parser.add_argument(
        "--modes", type=int, nargs=2, default=[10, 10], help="Modes for Fourier layers"
    )
    parser.add_argument("--width", type=int, default=64, help="Width of the network")
    parser.add_argument(
        "--activation", type=str, default="gelu", help="Activation function"
    )
    parser.add_argument("--grid", action="store_true", help="Whether to use grid")

    # MLP arguments
    parser.add_argument("--dimension", type=int, default=28, help="Dimension of input")
    parser.add_argument(
        "--width-predictor", type=int, default=4, help="Width of MLP predictor"
    )
    parser.add_argument(
        "--num-layers", type=int, default=3, help="Number of layers in MLP"
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    os.makedirs("Images", exist_ok=True)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if args.network == "NO":
        test_neural_operator(args, device)
    elif args.network == "NO_var_eps":
        test_neural_operator_var_eps(args, device)
    elif args.network == "MLP":
        test_mlp(args, device)


if __name__ == "__main__":
    main()
