import pandas as pd
import numpy as np 
import psutil
import concurrent.futures
import statsmodels.api as sm
import warnings

from typing import List, Optional, Tuple, Dict, Union
from numpy.typing import NDArray
from itertools import permutations

# Bypass CodeCarbon to avoid warnings on Mac M1/M2 chips


warnings.filterwarnings("ignore")

# Imports of utility functions from utils.py
from utils import (
    find_combinations, precompute_powers, generate_features, 
    get_feature_hashes, compute_abs_correlations, get_reg_sets, 
    filter_reg_sets, remove_elements_from_arrays, concurrent_regressions, 
    sort_array, get_intervals
)

class EcoRETINA:
    """
    EcoRETINA: An innovative, eco-friendly algorithm specifically designed for out-of-sample
    prediction. It functions as a regression-based flexible approximator, linear in parameters but
    nonlinear in inputs, utilizing a selective model search to optimize performance.

    This model builds engineered polynomial interaction features, selects subsets
    based on R² grid search, and fits a final statistical model using `statsmodels`.
    """

    def __init__(self):

    
        self.model_indices: Optional[NDArray] = None
        self.params: List[float] = []
        self.chunk_size: int = 500
        self.sm_model = None
        self.best_score: float = float('inf')
        self.combinations: List = []
        self.X_total: Optional[NDArray] = None
        self.X: Optional[NDArray] = None
        self.y: Optional[NDArray] = None
        
        
        self.add_relu: bool = False
        self.handle_zeros: str = 'prevent_division'
        self.epsilon: Union[float, str] = 'auto'
        self.add_log: bool = False
        self.con_cols_indices: List[int] = []
        self.cols_with_zeros_: List[int] = []
        self.translations_: Dict[int, float] = {}

    def fit(self, y: NDArray, X: NDArray, con_cols_indices: List[int], dummy_cols_indices: List[int], 
            col_names: Optional[List[str]] = None, params: List[float] = [-1.0, 0.0, 1.0], 
            cross_dummy: bool = False, max_r2: float = 0.9, grid: float = 0.005, 
            reg_type: str = 'linear', loss: str = 'mse', max_instances: int = 100000, 
            max_reg: int = 100, model_step: int = 1, chunk_size: int = 500, seed: int = 8, 
            cov_type: str = 'nonrobust', handle_zeros: str = 'prevent_division', 
            epsilon: Union[float, str] = 'auto', add_log: bool = False, add_relu: bool = False) -> None:
        
        self.params = params
        self.chunk_size = chunk_size
        self.handle_zeros = handle_zeros
        self.epsilon = epsilon
        self.con_cols_indices = con_cols_indices
        self.add_log = add_log
        self.add_relu = add_relu # CORRECTION : L'algorithme garde maintenant ReLU en mémoire !

        X, y = self._prepare_data(X, y, max_instances, seed)
        X = self._handle_zeros_in_data(X)

        combinations_list = find_combinations(con_cols_indices=self.con_cols_indices, dummy_cols_indices=dummy_cols_indices, params=self.params, cross_dummy=cross_dummy)
       
        if self.handle_zeros == 'prevent_division' and self.cols_with_zeros_:
            filtered_combos = []
            for combo in combinations_list:
                if len(combo) == 4:
                    a, b, c, d = combo
                    if (a in self.cols_with_zeros_ and c < 0) or (b in self.cols_with_zeros_ and d < 0):
                        continue
                filtered_combos.append(combo)
            combinations_list = filtered_combos

        self.combinations_all_ = combinations_list

        with np.errstate(divide='ignore', invalid='ignore'):
            precomputed_powers = precompute_powers(X, self.params)
            
        X_total = generate_features(X, combinations_list, precomputed_powers, self.params, self.chunk_size)
        hashes = get_feature_hashes(X_total)

        # --- GESTION DES LOGS AVEC SAUVEGARDE DU SHIFT ---
        log_names = []
        self.log_shifts_ = {} # On sauvegarde les décalages pour le predict()
        if self.add_log and len(self.con_cols_indices) > 0:
            X_log_list = []
            for col in self.con_cols_indices:
                col_data = X[:, col]
                if (col_data <= 0).any():
                    shift = np.abs(col_data.min()) + 1.0
                    self.log_shifts_[col] = shift
                    log_col = np.log(col_data + shift)
                else:
                    self.log_shifts_[col] = 0.0
                    log_col = np.log(col_data)
                X_log_list.append(log_col.reshape(-1, 1))
                
                base_name = col_names[col] if col_names is not None else f"x_{col}"
                log_names.append(f"ln({base_name})")
            
            if X_log_list:
                X_total = np.hstack([X_total, np.hstack(X_log_list)]) 

        # --- GESTION DU RELU AVEC SAUVEGARDE DE LA MEDIANE ---
        relu_names = []
        self.relu_thresholds_ = {} # On sauvegarde les médianes pour le predict()
        if self.add_relu and len(self.con_cols_indices) > 0:
            X_relu_list = []
            for col in self.con_cols_indices:
                col_data = X[:, col]
                threshold = np.median(col_data)
                self.relu_thresholds_[col] = threshold
                relu_col = np.maximum(0, col_data - threshold)
                X_relu_list.append(relu_col.reshape(-1, 1))
                
                base_name = col_names[col] if col_names is not None else f"x_{col}"
                relu_names.append(f"ReLU({base_name})")
            
            if X_relu_list:
                X_total = np.hstack([X_total, np.hstack(X_relu_list)])

        hashes = get_feature_hashes(X_total)

        # On envoie toutes les nouvelles colonnes pour les nommer proprement sans crasher
        all_extra_names = log_names + relu_names
        variables_df = self._generate_feature_names(col_names, combinations_list, hashes, X_total.shape[1], log_names=all_extra_names)

        X_chunks, y_chunks, corr_indices_dic, X_chunks_sorted_dic = self._split_and_correlate(X_total, y, hashes)
        reg_set_list_filt, indices_list = self._extract_and_filter_reg_sets(X_chunks, X_chunks_sorted_dic, corr_indices_dic, max_r2, grid, max_reg)
        model_indices = self._evaluate_subsamples(X_chunks, y_chunks, reg_set_list_filt, corr_indices_dic, loss, model_step)
        self._fit_final_model(X_total, y, variables_df, model_indices, reg_type, cov_type)

    def _generate_feature_names(self, col_names: Optional[List[str]], combinations_list: List, hashes: NDArray, total_cols: int, log_names: List[str] = []) -> pd.DataFrame:
        if col_names is not None:
            transf_variables = [(f"{col_names[a]}{'^' + str(c) if c != 1 else ''}" if c != 0 else "") + (f" * {col_names[b]}{'^' + str(d) if d != 1 else ''}" if c != 0 and d != 0 else f"{col_names[b]}{'^' + str(d) if d != 1 else ''}" if c == 0 else "") for (a, b, c, d) in combinations_list[1:]]
        else:
            transf_variables = [(f"x_{a}{'^' + str(c) if c != 1 else ''}" if c != 0 else "") + (f" * x_{b}{'^' + str(d) if d != 1 else ''}" if c != 0 and d != 0 else f"x_{b}{'^' + str(d) if d != 1 else ''}" if c == 0 else "") for (a, b, c, d) in combinations_list[1:]]

        all_var_names = ['constant'] + transf_variables + log_names

        extended_combinations = list(combinations_list)
        if log_names:
            # CORRECTION : Force l'alignement parfait du nombre de colonnes pour Pandas
            for i in range(len(log_names)):
                extended_combinations.append(('extra_feat', i))

        variables_df = pd.DataFrame({'variable': all_var_names, 'combination': extended_combinations, 'hash': hashes})
        indices = np.arange(0, total_cols)
        return variables_df.loc[indices]

    def transform(self, X: NDArray) -> NDArray:
        if X.ndim == 1: X = X.reshape(1, -1)
        X_safe = X.copy()

        if self.handle_zeros == 'translate':
            for col, shift in self.translations_.items():
                X_safe[:, col] += shift

        for col in self.con_cols_indices:
            if (X_safe[:, col] == 0).any():
                if self.handle_zeros == 'translate' and col not in self.translations_:
                    shift = np.abs(X_safe[:, col].min()) + 1.0 if self.epsilon == 'auto' else float(self.epsilon)
                    X_safe[:, col] += shift
                elif self.handle_zeros == 'prevent_division' and col not in self.cols_with_zeros_:
                    fallback_val = 1e-5 if self.epsilon == 'auto' else float(self.epsilon)
                    X_safe[:, col] = np.where(X_safe[:, col] == 0, fallback_val, X_safe[:, col])
                    
        with np.errstate(divide='ignore', invalid='ignore'):
            from utils import precompute_powers, generate_features
            precomputed_powers = precompute_powers(X_safe, self.params)
            
        # L'algorithme utilise proprement combinations_all_ (sans le texte), ce qui évite le crash de Numba !
        X_features = generate_features(X_safe, self.combinations_all_, precomputed_powers, self.params, self.chunk_size) if hasattr(self, 'combinations_all_') else generate_features(X_safe, self.combinations, precomputed_powers, self.params, self.chunk_size)
        
        if self.add_log and len(self.con_cols_indices) > 0:
            X_log_list = []
            for col in self.con_cols_indices:
                col_data = X_safe[:, col]
                shift = getattr(self, 'log_shifts_', {}).get(col, 0.0)
                val = np.maximum(1e-5, col_data + shift)
                log_col = np.log(val)
                X_log_list.append(log_col.reshape(-1, 1))
            if X_log_list:
                X_features = np.hstack([X_features, np.hstack(X_log_list)])

        if self.add_relu and len(self.con_cols_indices) > 0:
            X_relu_list = []
            for col in self.con_cols_indices:
                col_data = X_safe[:, col]
                threshold = getattr(self, 'relu_thresholds_', {}).get(col, 0.0)
                relu_col = np.maximum(0, col_data - threshold)
                X_relu_list.append(relu_col.reshape(-1, 1))
            if X_relu_list:
                X_features = np.hstack([X_features, np.hstack(X_relu_list)])

        if self.model_indices is not None:
            X_transformed = X_features[:, self.model_indices]
        else:
            X_transformed = X_features

        return X_transformed

    def predict(self, X: NDArray, confidence: float = 0.95) -> NDArray:
        X_transformed = self.transform(X)
        y_pred = self.sm_model.predict(X_transformed)
        from utils import get_intervals
        self.pi_lower, self.pi_upper, self.ci_lower, self.ci_upper = get_intervals(y_train=self.y, X_train=self.X, beta=self.sm_model.params.values, X_new=X_transformed, confidence=confidence)
        return y_pred

    def _prepare_data(self, X: NDArray, y: NDArray, max_instances: int, seed: int) -> Tuple[NDArray, NDArray]:
        """
        Randomly shuffle and subsample the dataset to a maximum of `max_instances` rows.
        Drop rows with zeros in continuous columns if `handle_zeros` is set to 'drop_rows'.
        """
        
        n_rows = X.shape[0]
        rng = np.random.default_rng(seed)
        indices = rng.permutation(n_rows)

        y_sub = np.take(y, indices, axis=0)[:max_instances]
        X_sub = np.take(X, indices, axis=0)[:max_instances]

        if self.handle_zeros == 'drop_rows':
            mask = (X_sub[:, self.con_cols_indices] != 0).all(axis=1)
            return X_sub[mask], y_sub[mask]
        
        return X_sub, y_sub

    def _handle_zeros_in_data(self, X: NDArray) -> NDArray:
        """
        Detect and handle zeros in continuous columns based on the specified strategy
        """
        self.cols_with_zeros_ = []
        self.translations_ = {}
        
        for col in self.con_cols_indices:
            if (X[:, col] == 0).any():
                if self.handle_zeros == 'translate':
                    if self.epsilon == 'auto':
                        shift = np.abs(X[:, col].min()) + 1.0
                    else:
                        shift = float(self.epsilon)
                        while (X[:, col] + shift == 0).any():
                            shift += float(self.epsilon)
                    
                    X[:, col] += shift
                    self.translations_[col] = shift
                    
                elif self.handle_zeros == 'prevent_division':
                    self.cols_with_zeros_.append(col)
                    
        return X

    

    def _split_and_correlate(self, X_total: NDArray, y: NDArray, hashes: NDArray):
        """
        Split the dataset into three chunks and compute absolute correlations for each chunk.
        Returns the chunks, their corresponding target values, and sorted indices based on correlations."""
        X_0, X_1, X_2 = np.array_split(X_total, 3)
        y_0, y_1, y_2 = np.array_split(y, 3)

        X_chunks = [X_0, X_1, X_2]
        y_chunks = [y_0, y_1, y_2]
        corr_indices_dic = {}
        X_chunks_sorted_dic = {}

        for idx, (X_chunk, y_chunk) in enumerate(zip(X_chunks, y_chunks)):
            correlations = compute_abs_correlations(X_chunk, y_chunk)
            abs_corr = -correlations 
            corr_index = np.lexsort((hashes, abs_corr))
            corr_index = corr_index[corr_index != 0]
            corr_index = np.append(corr_index, 0)
            corr_indices_dic[idx] = corr_index
            X_chunks_sorted_dic[idx] = np.take(X_chunk, corr_index, axis=1)

        return X_chunks, y_chunks, corr_indices_dic, X_chunks_sorted_dic

    def _extract_and_filter_reg_sets(self, X_chunks, X_chunks_sorted_dic, corr_indices_dic, max_r2, grid, max_reg):
        """
        Extract regression sets from each chunk and filter them based on correlations and R² thresholds.
        Returns the filtered regression sets and their corresponding indices."""
        inputs = [(X_chunks_sorted_dic[0], corr_indices_dic[0]), (X_chunks_sorted_dic[1], corr_indices_dic[1]), (X_chunks_sorted_dic[2], corr_indices_dic[2])]

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(3, psutil.cpu_count(logical=False))) as executor:
            reg_set_list = list(executor.map(lambda args: get_reg_sets(*args), [(tuple, max_r2, grid, max_reg-2) for tuple in inputs]))

        groups = list(permutations([0,1,2], 2))
        indices_list = [reg_set_list[0][-1], reg_set_list[1][-1], reg_set_list[2][-1]]

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(3, psutil.cpu_count(logical=False))) as executor:
            exc_cols_list = list(executor.map(lambda args: filter_reg_sets(*args), [(tuple, indices_list, X_chunks, max_r2) for tuple in groups]))

        exc_indices_0 = np.array(sorted(set(exc_cols_list[0] + exc_cols_list[1])))
        exc_indices_1 = np.array(sorted(set(exc_cols_list[2] + exc_cols_list[3])))
        exc_indices_2 = np.array(sorted(set(exc_cols_list[4] + exc_cols_list[5])))

        exc_indices_0 = indices_list[0][exc_indices_0] if len(exc_indices_0) != 0 else np.array([])
        exc_indices_1 = indices_list[1][exc_indices_1] if len(exc_indices_1) != 0 else np.array([])
        exc_indices_2 = indices_list[2][exc_indices_2] if len(exc_indices_2) != 0 else np.array([])
        exc_indices_list = [exc_indices_0, exc_indices_1, exc_indices_2]

        reg_set_list_filt = [remove_elements_from_arrays(reg_set_list[i], exc_indices_list[i]) for i in range(len(reg_set_list))]
        
        return reg_set_list_filt, indices_list

    def _evaluate_subsamples(self, X_chunks, y_chunks, reg_set_list_filt, corr_indices_dic, loss, model_step):
        """
        Evaluate all permutations of the three data chunks to find the best model indices based on the specified
        loss metric. Returns the best model indices.
        """
        subsample_indices = list(permutations([0,1,2]))
        score_dict = {}
        models_dict = {}

        for i, j in enumerate(subsample_indices):
            X_0, X_1, X_2 = X_chunks[j[0]], X_chunks[j[1]], X_chunks[j[2]]
            y_0, y_1, y_2 = y_chunks[j[0]], y_chunks[j[1]], y_chunks[j[2]]
            
            reg_set = reg_set_list_filt[j[0]]
            score_list = concurrent_regressions(y_train=y_0, X_train=X_0, y_test=y_1, X_test=X_1, loss=loss, reg_sets=reg_set)
            
            candidate_model = reg_set[score_list.index(min(score_list))]
            candidate_model_sorted = sort_array(array_to_sort=candidate_model, reference_array=corr_indices_dic[j[1]])
            candidate_model_sorted = np.hstack((0, candidate_model_sorted[0:-1]))
            
            candidate_model_list_1 = [candidate_model[:k] for k in range(1, len(candidate_model) + 1)][::model_step]
            candidate_model_list_2 = [candidate_model_sorted[:k] for k in range(1, len(candidate_model_sorted) + 1)][::model_step]
            
            score_list_1 = concurrent_regressions(y_train=y_1, X_train=X_1, y_test=y_2, X_test=X_2, loss=loss, reg_sets=candidate_model_list_1)
            score_list_2 = concurrent_regressions(y_train=y_1, X_train=X_1, y_test=y_2, X_test=X_2, loss=loss, reg_sets=candidate_model_list_2)
            
            best_score_1 = min(score_list_1)
            best_score_2 = min(score_list_2)
            
            best_score, score_list, candidate_model_list = (best_score_1, score_list_1, candidate_model_list_1) if best_score_1 <= best_score_2 else (best_score_2, score_list_2, candidate_model_list_2)
            model_indices = candidate_model_list[score_list.index(best_score)]
            
            score_dict[i] = best_score
            models_dict[i] = model_indices

        self.best_score = min(score_dict.values())
        key = next((k for k, v in score_dict.items() if v == self.best_score))
        return models_dict[key]

    def _fit_final_model(self, X_total, y, variables_df, model_indices, reg_type, cov_type):
        """
        Fit the final statistical model using the best model indices and store the fitted model.
        """

        self.model_indices = model_indices

        self.combinations = list(variables_df['combination'].iloc[model_indices].values)
        model_variables = variables_df.loc[model_indices, 'variable'] 
       
        X = X_total[:, model_indices]
        self.X_total = X_total
        self.X = X
        self.y = y
        X_df = pd.DataFrame(X, columns=model_variables)
       
        if reg_type == 'logit':
            self.sm_model = sm.Logit(y, X_df).fit(cov_type=cov_type)
        elif reg_type == 'probit':
            self.sm_model = sm.Probit(y, X_df).fit(cov_type=cov_type)
        elif reg_type == 'linear':
            self.sm_model = sm.OLS(y, X_df).fit(cov_type=cov_type)
        else:
            raise ValueError(f"Unknown reg_type '{reg_type}'. Expected 'linear', 'logit', or 'probit'.")

    
