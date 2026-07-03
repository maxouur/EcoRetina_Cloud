import numpy as np
from numpy.typing import NDArray
import math
from scipy.stats import t
from itertools import combinations, product
from typing import List, Tuple
from numba import njit, prange
from concurrent.futures import ThreadPoolExecutor
import hashlib

class LinearRegression:
    """
    Linear regression model using the normal equation.
    Supports multi-output regression (Y can have multiple columns).
    """

    def __init__(self):
        self.params: NDArray | None = None
        self.rsquared: NDArray | None = None
        self.ss_residual: NDArray | None = None
        self.n: int | None = None
        self.k: int | None = None
        self.aic: NDArray | None = None
        self.bic: NDArray | None = None  

    def fit(self, Y: NDArray, X: NDArray) -> None:
        """
        Fit the linear regression model using the normal equation.
        """
        self.n, self.k = X.shape
        self.params = np.linalg.solve(X.T @ X, X.T @ Y)
        Y_pred = X @ self.params
        self.ss_residual = np.sum((Y - Y_pred) ** 2, axis=0)
        ss_total = np.sum((Y - np.mean(Y, axis=0)) ** 2, axis=0)
        self.rsquared = np.where(
            np.isclose(ss_total, 0),
            np.nan,
            1 - (self.ss_residual / ss_total)
        )
        
        self.aic = None
        self.bic = None

    def compute_aic(self) -> NDArray:
        """
        Compute the Akaike Information Criterion (AIC) for the fitted model.
        """
        if self.aic is not None:
            return self.aic 

        if self.ss_residual is None or self.n is None or self.k is None:
            raise ValueError("Model must be fitted before computing AIC.")

        sigma2 = self.ss_residual / self.n

        log_likelihood = np.where(
            np.isclose(sigma2, 0),
            np.nan,
            -0.5 * self.n * (np.log(2 * np.pi * sigma2) + 1)
        )

        self.aic = np.where(
            np.isnan(log_likelihood),
            np.nan,
            -2 * log_likelihood + 2 * self.k
        )

        return self.aic

    def compute_bic(self) -> NDArray:
        """
        Compute the Bayesian Information Criterion (BIC) for the fitted model.
        """
        if self.bic is not None:
            return self.bic  

        if self.ss_residual is None or self.n is None or self.k is None:
            raise ValueError("Model must be fitted before computing BIC.")

        sigma2 = self.ss_residual / self.n

        log_likelihood = np.where(
            np.isclose(sigma2, 0),
            np.nan,
            -0.5 * self.n * (np.log(2 * np.pi * sigma2) + 1)
        )

        self.bic = np.where(
            np.isnan(log_likelihood),
            np.nan,
            -2 * log_likelihood + self.k * np.log(self.n)
        )

        return self.bic

    def predict(self, X: NDArray) -> NDArray:
        """
        Predict using the fitted linear regression model.
        """
        if self.params is None:
            raise ValueError("Model must be fitted before prediction.")
        return X @ self.params
    


def find_no_cross_params(params: List[float]) -> List[Tuple[float, float]]:
    """
    Generate unique, non-crossing parameter combinations for feature interactions.

    A non-crossing combination means no opposite pairs (e.g., 1 and -1).
    Ensures that combinations avoid negating effects and include min/max self-pairs.

    Args:
        params: List of power values (e.g., [-1.0, 0.0, 1.0]).

    Returns:
        List of unique (param1, param2) tuples for no-cross combinations.
    """
    min_val, max_val = min(params), max(params)
    filtered_combinations = [
        (x, y) for x, y in combinations(params, 2) 
        if x != -y and (x * y >= 0 or x == 0 or y == 0)
    ]
    filtered_combinations.extend([(min_val, min_val), (max_val, max_val)])
    unique_combinations = {}
    for x, y in filtered_combinations:
        tuple_sum = x + y
        if tuple_sum not in unique_combinations:
            unique_combinations[tuple_sum] = (x, y)
    return list(unique_combinations.values())

def find_combinations(
    con_cols_indices: List[int],
    dummy_cols_indices: List[int],
    params: List[float],
    cross_dummy: bool
) -> List[Tuple[int, int, float, float]]:
    """
    Generate feature interaction tuples using specified powers and column indices.

    Creates no-cross and cross-product combinations of continuous and dummy features,
    returning encoded interaction terms (index1, index2, power1, power2).

    Args:
        con_cols_indices: Indices of continuous variables.
        dummy_cols_indices: Indices of dummy/categorical variables.
        params: List of powers to apply to each feature.
        cross_dummy: Whether to include dummy-dummy and dummy-continuous interactions.

    Returns:
        List of 4-tuples representing feature interaction specifications.
    """
    
    no_cross_con_cols = [(x, x) for x in con_cols_indices]
    cross_con_cols = [(a, b) for a, b in combinations(con_cols_indices, 2)]
   
    no_cross_params = find_no_cross_params(params)
    cross_params = [(a, b) for a, b in product(params, repeat=2) if a != 0 and b != 0]

    no_cross_con_tuples = [
        (np.int32(a), np.int32(b), np.float64(c), np.float64(d))
        for (a, b), (c, d) in product(no_cross_con_cols, no_cross_params)
    ]
    cross_con_tuples = [
        (np.int32(a), np.int32(b), np.float64(c), np.float64(d))
        for (a, b), (c, d) in product(cross_con_cols, cross_params)
    ]
    
    no_cross_dummy_cols = [(x, x) for x in dummy_cols_indices]
    cross_con_dummy_cols = [(x, y) for x in dummy_cols_indices for y in con_cols_indices]
    cross_dummy_dummy_cols = list(combinations(dummy_cols_indices, 2))

    no_cross_dummy_tuples = [
        (np.int32(a), np.int32(b), np.float64(1.0), np.float64(0.0))
        for (a, b) in no_cross_dummy_cols
    ]
    cross_con_dummy_tuples = [
        (np.int32(a), np.int32(b), np.float64(1.0), np.float64(1.0))
        for (a, b) in cross_con_dummy_cols
    ]
    cross_dummy_dummy_tuples = [
        (np.int32(a), np.int32(b), np.float64(1.0), np.float64(1.0))
        for (a, b) in cross_dummy_dummy_cols
    ]

    cross_dummy_tuples = cross_con_dummy_tuples + cross_dummy_dummy_tuples if cross_dummy else cross_con_dummy_tuples
    
    all_tuples = [(np.int32(0), np.int32(0), np.float64(0.0), np.float64(0.0))] + \
                 no_cross_con_tuples + cross_con_tuples + \
                 no_cross_dummy_tuples + cross_dummy_tuples

    return all_tuples


@njit(parallel=True, fastmath=True)
def precompute_powers(data: NDArray, unique_powers: List[float]) -> NDArray:
    """
    Precompute powers of each feature for faster interaction generation.

    Args:
        data: Original input data of shape (n_samples, n_features).
        unique_powers: Powers to raise each feature to.

    Returns:
        3D array with shape (n_features, n_powers, n_samples) containing precomputed values.
    """
     
    n_samples, n_features = data.shape
    precomputed = np.empty((n_features, len(unique_powers), n_samples), dtype=np.float32)

    for feature_idx in prange(n_features):
        for power_idx, power in enumerate(unique_powers):
            precomputed[feature_idx, power_idx, :] = data[:, feature_idx] ** np.float32(power)

    return precomputed

@njit(parallel=True, fastmath=True)
def generate_features(
    data: NDArray, 
    combinations: List[Tuple[int, int, float, float]], 
    precomputed: NDArray, 
    unique_powers: List[float], 
    chunk_size: int = 1000  
) -> np.ndarray:
    """
    Generate transformed interaction features from specified combinations.

    Args:
        data: Original input data.
        combinations: List of (i, j, a, b) specifying interactions.
        precomputed: Precomputed power values from `precompute_powers`.
        unique_powers: List of all power values.
        chunk_size: Number of combinations to process per chunk.

    Returns:
        Transformed feature matrix of shape (n_samples, n_combinations).
    """
    
    n_samples = data.shape[0]
    n_combinations = len(combinations)
    
    result = np.empty((n_samples, n_combinations), dtype=np.float32)  

    for chunk_start in prange(n_combinations):
        if chunk_start % chunk_size == 0:
            chunk_end = min(chunk_start + chunk_size, n_combinations)

            for comb_idx in range(chunk_start, chunk_end):
                i, j, a, b = combinations[comb_idx]
                i, j = int(i), int(j)

                a_idx = unique_powers.index(a)
                b_idx = unique_powers.index(b)

                pow_a = precomputed[i, a_idx, :n_samples]
                pow_b = precomputed[j, b_idx, :n_samples]

                result[:, comb_idx] = pow_a * pow_b

    return result


def get_feature_hashes(X: np.ndarray) -> list[str]:
    """
    Compute SHA-256 hashes of columns in a matrix for unique identification.

    Args:
        X: 2D array of features.

    Returns:
        List of hexadecimal hash strings, one per column.
    """
    return [hashlib.sha256(X[:, i].tobytes()).hexdigest() for i in range(X.shape[1])]

@njit(fastmath=True, parallel=True)
def compute_abs_correlations(X: NDArray, y: NDArray) -> NDArray:
    """
    Compute absolute Pearson correlations between each feature and target.

    Args:
        X: Input features.
        y: Target vector.

    Returns:
        Array of absolute correlation values for each feature.
    """
    n_samples, n_features = X.shape
    y_mean = np.sum(y) / n_samples
    X_mean = np.zeros(n_features, dtype=np.float32)

    for i in prange(n_features):
        X_mean[i] = np.sum(X[:, i]) / n_samples

    num = np.zeros(n_features, dtype=np.float32)
    denom = np.zeros(n_features, dtype=np.float32)

    for i in prange(n_features):
        num[i] = np.sum((X[:, i] - X_mean[i]) * (y - y_mean))
        denom[i] = np.sqrt(np.sum((X[:, i] - X_mean[i]) ** 2) * np.sum((y - y_mean) ** 2))

    correlations = num / denom
    return np.abs(correlations)


def get_reg_sets(tuple: Tuple[NDArray, NDArray], max_r2: float, grid: float, max_iterations: int) -> List[NDArray]:
    """
    Iteratively build subsets of features based on increasing R² values.

    Args:
        tuple: Tuple of (X, correlation index) to sort features.
        max_r2: Maximum allowed R² before stopping.
        grid: Step size for R² grid search.
        max_iterations: Maximum number of iterations allowed.

    Returns:
        List of feature index subsets (NDArray), each representing a model.
    """
    reg = LinearRegression()
    X=tuple[0]
    corr_index=tuple[1]
    r2 = 0
    reg_sets = []
    reg_indices = [X.shape[1] - 1, 0]
    exc_indices = []
    r2_grid_dict={}
    reg_sets.append(corr_index[reg_indices])
    mask = np.ones((X.shape[1],), dtype=bool)
    iteration = 0

    while r2 <= max_r2 and iteration < max_iterations:
        iteration += 1
        mask[reg_indices] = False
        mask[exc_indices] = False
        reg.fit(Y=X[:, mask], X=X[:, reg_indices])
        r2_array = np.full(mask.shape, np.nan)  
        r2_array[mask] = reg.rsquared  
        valid_rsquared = reg.rsquared[(reg.rsquared <= max_r2) & (reg.rsquared > r2)]
        if len(valid_rsquared) == 0:
            break
        r2 = np.min(valid_rsquared)
        r2_grid = min(math.ceil(r2 / grid) * grid, max_r2 + grid) + (grid if math.ceil(r2) == 0 else 0) - (0 if math.ceil(r2 / grid) * grid <= max_r2 else grid)
        r2_grid = max(r2, r2_grid)
        new_cols_candidates = list(np.where(r2_array <= r2_grid)[0])
        first_element=new_cols_candidates.pop(0)
        reg_indices.append(first_element)
        exc_indices = list(np.where(r2_array > max_r2)[0])
        r2_grid_dict[r2_grid]=corr_index[reg_indices]

    return list(r2_grid_dict.values())

def filter_reg_sets(pair: Tuple[int, int], indices_list: List[NDArray], arr_list: List[NDArray], max_r2: float) -> List[int]:
    """
    Filter feature indices that lead to high multicollinearity (R² above threshold).

    Args:
        pair: Pair of indices (source model, test array).
        indices_list: List of feature index arrays for each model.
        arr_list: List of feature arrays.
        max_r2: Maximum acceptable R² to avoid exclusion.

    Returns:
        List of feature indices to exclude.
    """
    indices=indices_list[pair[0]]
    arr=arr_list[pair[1]]
    X=arr[:, indices]
    reg = LinearRegression()
    reg_indices = [0,1]
    exc_indices=[]
    for i in range(2, X.shape[1]):
        reg.fit(Y=X[:, i], X=X[:, reg_indices])
        if reg.rsquared < max_r2:
            reg_indices.append(i)
        else:
            exc_indices.append(i)
    return exc_indices

def remove_elements_from_arrays(arr_list: List[NDArray], arr: NDArray) -> List[NDArray]:
    """
    Remove specified indices from each array in a list.

    Args:
        arr_list: List of index arrays.
        arr: Array of indices to remove.

    Returns:
        New list of arrays with specified elements removed.
    """
    filtered_list = [x[~np.isin(x, arr)] for x in arr_list]
    return filtered_list



def regression_task(y_train:NDArray, X_train: NDArray, y_test:NDArray, X_test:NDArray, loss:str, subset_indices:NDArray)-> float:
    """
    Evaluate a regression model on a subset of features and compute a loss metric.

    Args:
        y_train: Training targets.
        X_train: Full training feature set.
        y_test: Test targets.
        X_test: Full test feature set.
        loss: Loss metric to compute ('mse', 'mae', 'mape', 'aic', 'bic').
        subset_indices: Indices of features to use.

    Returns:
        Computed loss value.
    """
    X_train_subset = np.take(X_train, subset_indices, axis=1)
    model = LinearRegression()
    model.fit(y_train, X_train_subset)
    if loss== 'mse' or loss== 'mape' or loss== 'mae':
        X_test_subset = np.take(X_test, subset_indices, axis=1)
        y_pred = model.predict(X_test_subset)

        if loss== 'mse':
            result = np.mean((y_test - y_pred) ** 2)
        
        elif loss== 'mape':
            result=np.mean(np.abs((y_test - y_pred) / y_test))

        elif loss== 'mae':
            result=np.mean(np.abs(y_test - y_pred))

    elif loss =='aic':
        result=model.compute_aic()

    elif loss == 'bic':
        result=model.compute_bic()
    
    else:
            raise ValueError(f"Unknown loss '{loss}'. Expected 'mse', 'mape', 'mae', 'aic' or 'bic'.")

    return result



def concurrent_regressions(y_train: NDArray, X_train: NDArray, y_test: NDArray, X_test: NDArray, loss:str, reg_sets: List[NDArray]) -> List[float]:
    """
    Run multiple regression evaluations in parallel.

    Args:
        y_train: Training targets.
        X_train: Full training features.
        y_test: Test targets.
        X_test: Full test features.
        loss: Loss function to use.
        reg_sets: List of feature subsets to evaluate.

    Returns:
        List of loss values, one per subset.
    """
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda args: regression_task(*args), [(y_train, X_train, y_test, X_test, loss, subset) for subset in reg_sets]))
    return results

def sort_array(array_to_sort: NDArray, reference_array: NDArray) -> NDArray:
    """
    Sort an array to match the order of a reference array.

    Args:
        array_to_sort: Array to be reordered.
        reference_array: Reference array defining correct order.

    Returns:
        Reordered array matching the reference order.
    """
    ref_index_map = {value: index for index, value in enumerate(reference_array)}
    sorted_array = sorted(array_to_sort, key=lambda x: ref_index_map[x])
    return sorted_array


def get_intervals(y_train:NDArray, X_train:NDArray, beta:NDArray, X_new:NDArray, confidence:float):
    """
    Compute prediction and confidence intervals for linear regression predictions.

    Args:
        y_train: Training targets.
        X_train: Training features used for fitting.
        beta: Model coefficients.
        X_new: New data points for prediction.
        confidence: Confidence level (e.g., 0.95).

    Returns:
        Tuple of arrays: (pi_lower, pi_upper, ci_lower, ci_upper).
    """
    n, k= X_train.shape
    y_pred_train= X_train @ beta
    residuals= y_train - y_pred_train 
    sigma2 = np.sum(residuals**2) / (n - k)
    alpha = 1 - confidence
    t_crit = t.ppf(1 - alpha / 2, df=n - k)

    pi_lower = np.empty(X_new.shape[0], dtype=np.float64)
    pi_upper = np.empty(X_new.shape[0], dtype=np.float64)
    ci_lower = np.empty(X_new.shape[0], dtype=np.float64)
    ci_upper = np.empty(X_new.shape[0], dtype=np.float64)

    XTX_inv = np.linalg.inv(X_train.T @ X_train)

    for i in range(X_new.shape[0]):
        
        x_star=X_new[i, :]
        y_star = (x_star @ beta)
     
        pred_var_pi = sigma2 * (1 + x_star @ XTX_inv @ x_star.T)
        pred_var_ci = sigma2 * (x_star @ XTX_inv @ x_star.T)

        margin_pi = (t_crit * np.sqrt(pred_var_pi))
        margin_ci = (t_crit * np.sqrt(pred_var_ci))

        pi_lower[i] = y_star - margin_pi
        pi_upper[i] = y_star + margin_pi
        ci_lower[i] = y_star - margin_ci
        ci_upper[i] = y_star + margin_ci

    return pi_lower, pi_upper, ci_lower, ci_upper
    