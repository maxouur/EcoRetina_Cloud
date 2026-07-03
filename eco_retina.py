import pandas as pd
import numpy as np 
import psutil
from numpy.typing import NDArray
from itertools import permutations
from typing import List
import concurrent.futures
import statsmodels.api as sm
from itertools import permutations 
from codecarbon import  EmissionsTracker
import warnings
warnings.filterwarnings("ignore")

from utils import (find_combinations, precompute_powers, generate_features, get_feature_hashes, compute_abs_correlations, get_reg_sets, 
filter_reg_sets, remove_elements_from_arrays, concurrent_regressions, sort_array, get_intervals)

class EcoRETINA:
    """
    EcoRETINA: An innovative, eco-friendly algorithm specifically designed for out-of-sample
    prediction. It functions as a regression-based flexible approximator, linear in parameters but
    nonlinear in inputs, utilizing a selective model search to optimize performance.

    This model builds engineered polynomial interaction features, selects subsets
    based on R² grid search, and fits a final statistical model using `statsmodels`.
    """

    def __init__(self):
        """
        Initialize EcoRETINA with placeholders for model attributes.
        """
        self.model_indices: NDArray | None = None

    

    def fit(self, y: NDArray, X: NDArray, con_cols_indices: List[int], dummy_cols_indices: List[int], col_names: List[str] | None, params: List[float]=[-1.0, 0.0, 1.0], cross_dummy: bool=False, 
            max_r2:float=0.9, grid:float=0.005, reg_type:str='linear', loss:str='mse',max_instances:int=100000, max_reg:int=100, model_step:int=1, chunk_size:int=500, 
            seed:int=8, cov_type:str='nonrobust' ) -> None:
        """
        Fit the EcoRETINA model on a dataset using a grid-based subset selection strategy.

        Args:
            y: Target variable.
            X: Input feature matrix.
            con_cols_indices: Indices of continuous variables.
            dummy_cols_indices: Indices of categorical (dummy) variables.
            col_names: Optional feature names for labeling interactions.
            params: List of exponents to apply to features.
            cross_dummy: Whether to allow interactions between dummy variables.
            max_r2: Maximum R² threshold for feature subset inclusion.
            grid: R² grid step for incremental feature search.
            reg_type: Regression model type: 'linear', 'logit', or 'probit'.
            loss: Loss metric to use during model selection.
            max_instances: Maximum number of training rows to consider.
            max_reg: Maximum number of regression features allowed.
            model_step: Step size for evaluating nested models.
            chunk_size: Chunk size for feature generation.
            seed: Random seed for reproducibility.
            cov_type: Type of covariance estimation for statsmodels.
        """
        
        tracker = EmissionsTracker(tracking_mode='process', output_file='eco_retina_emissions.csv', project_name='Eco-RETINA')
        tracker.start()
        

        self.params=params
        self.chunk_size=chunk_size
        
        n_rows=X.shape[0]
        rng = np.random.default_rng(seed)
        indices = rng.permutation(n_rows)

        y = np.take(y, indices, axis=0)[:max_instances]
        X = np.take(X, indices, axis=0)[:max_instances]

        mask = (X[:, con_cols_indices] != 0).all(axis=1) 
        y = y[mask]
        X = X[mask]

        combinations_list=find_combinations(con_cols_indices=con_cols_indices, dummy_cols_indices=dummy_cols_indices, params=self.params, cross_dummy=cross_dummy)

        precomputed_powers = precompute_powers(X, self.params)

        X_total = generate_features(X, combinations_list, precomputed_powers, self.params, self.chunk_size)

        hashes= get_feature_hashes(X_total)

        if col_names is not None:
            transf_variables = [
                (f"{col_names[a]}{'^' + str(c) if c != 1 else ''}" if c != 0 else "") +
                (f" * {col_names[b]}{'^' + str(d) if d != 1 else ''}" if c != 0 and d != 0 else f"{col_names[b]}{'^' + str(d) if d != 1 else ''}" if c == 0 else "")
                for (a, b, c, d) in combinations_list[1:]
            ]
        else:
            transf_variables = [
            (f"x_{a}{'^' + str(c) if c != 1 else ''}" if c != 0 else "") +
            (f" * x_{b}{'^' + str(d) if d != 1 else ''}" if c != 0 and d != 0 else f"x_{b}{'^' + str(d) if d != 1 else ''}" if c == 0 else "")
            for (a, b, c, d) in combinations_list[1:]
        ]


        variables_df=pd.DataFrame({'variable': ['constant'] + transf_variables, 'combination':  combinations_list, 'hash': hashes})
        indices = np.arange(0, X_total.shape[1])
        variables_df=variables_df.loc[indices]

        X_0, X_1, X_2 = np.array_split(X_total, 3)
        y_0, y_1, y_2 = np.array_split(y, 3)


        X_chunks = [X_0, X_1, X_2]
        y_chunks = [y_0, y_1, y_2]
        corr_indices_dic = {}
        X_chunks_sorted_dic={}

        for idx, (X_chunk, y_chunk) in enumerate(zip(X_chunks, y_chunks), start=0):
            correlations = compute_abs_correlations(X_chunk, y_chunk)
            abs_corr = -correlations 
            corr_index = np.lexsort((hashes, abs_corr))
            corr_index=corr_index[corr_index !=0]
            corr_index=np.append(corr_index, 0)
            corr_indices_dic[idx] = corr_index
            X_chunks_sorted_dic[idx] = np.take(X_chunk, corr_index, axis=1)
            

        inputs = [(X_chunks_sorted_dic[0],corr_indices_dic[0]), (X_chunks_sorted_dic[1], corr_indices_dic[1]), (X_chunks_sorted_dic[2], corr_indices_dic[2])]

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(3, psutil.cpu_count(logical=False))) as executor:
            reg_set_list = list(executor.map(lambda args: get_reg_sets(*args), [(tuple, max_r2, grid, max_reg-2) for tuple in inputs]))

        groups = list(permutations([0,1,2], 2))
        indices_list=[reg_set_list[0][-1], reg_set_list[1][-1], reg_set_list[2][-1]]
        arr_list=[X_0, X_1, X_2]

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(3, psutil.cpu_count(logical=False))) as executor:
            exc_cols_list = list(executor.map(lambda args: filter_reg_sets(*args), [(tuple, indices_list, arr_list, max_r2) for tuple in groups]))

        exc_indices_0=np.array(sorted(set(exc_cols_list[0]+ exc_cols_list[1])))
        exc_indices_1=np.array(sorted(set(exc_cols_list[2]+ exc_cols_list[3])))
        exc_indices_2=np.array(sorted(set(exc_cols_list[4]+ exc_cols_list[5])))

        exc_indices_0=indices_list[0][exc_indices_0] if len(exc_indices_0) != 0 else np.array([])
        exc_indices_1=indices_list[1][exc_indices_1] if len(exc_indices_1) != 0 else np.array([])
        exc_indices_2=indices_list[2][exc_indices_2] if len(exc_indices_2) != 0 else np.array([])
        exc_indices_list=[exc_indices_0, exc_indices_1, exc_indices_2]

        reg_set_list_filt=[remove_elements_from_arrays(reg_set_list[i], exc_indices_list[i]) for i in range(len(reg_set_list))]

        subsample_indices=list(permutations([0,1,2]))
        score_dict={}
        models_dict={}
        for i,j in enumerate(subsample_indices):
            X_0=X_chunks[j[0]]
            X_1=X_chunks[j[1]]
            X_2=X_chunks[j[2]]
            y_0=y_chunks[j[0]]
            y_1=y_chunks[j[1]]
            y_2=y_chunks[j[2]]
            reg_set=reg_set_list_filt[j[0]]
            score_list=concurrent_regressions(y_train=y_0, X_train=X_0, y_test=y_1, X_test=X_1, loss=loss, reg_sets=reg_set)
            candidate_model=reg_set[score_list.index(min(score_list))]
            candidate_model_sorted=sort_array(array_to_sort=candidate_model, reference_array=corr_indices_dic[j[1]])
            candidate_model_sorted=np.hstack((0, candidate_model_sorted[0:-1]))
            candidate_model_list_1= [candidate_model[:i] for i in range(1, len(candidate_model) + 1)][::model_step]
            candidate_model_list_2= [candidate_model_sorted[:i] for i in range(1, len(candidate_model_sorted) + 1)][::model_step]
            score_list_1=concurrent_regressions(y_train=y_1, X_train=X_1, y_test=y_2, X_test=X_2, loss=loss, reg_sets=candidate_model_list_1)
            score_list_2=concurrent_regressions(y_train=y_1, X_train=X_1, y_test=y_2, X_test=X_2, loss=loss, reg_sets=candidate_model_list_2)
            best_score_1=min(score_list_1)
            best_score_2=min(score_list_2)
            best_score, score_list, candidate_model_list =(best_score_1, score_list_1, candidate_model_list_1) if best_score_1 <= best_score_2 else (best_score_2, score_list_2, candidate_model_list_2)
            model_indices=candidate_model_list[score_list.index(best_score)]
            score_dict[i]=best_score
            models_dict[i]=model_indices
            
        self.best_score=min(score_dict.values())
        key = next((k for k, v in score_dict.items() if v == self.best_score))
        model_indices=models_dict[key]
        self.combinations=list(variables_df['combination'].iloc[model_indices].values)

        model_variables=variables_df.loc[model_indices, 'variable'] 
       
        X=X_total[:, model_indices]
        self.X_total=X_total
        self.X=X
        self.y=y
        X_df = pd.DataFrame(X, columns=model_variables)
       
        if reg_type== 'logit':
            self.sm_model=sm.Logit(y, X_df).fit(cov_type=cov_type)

        elif reg_type== 'probit':
            self.sm_model=sm.Probit(y, X_df).fit(cov_type=cov_type)

        elif reg_type== 'linear':
            self.sm_model = sm.OLS(y, X_df).fit(cov_type=cov_type)
        
        else:
            raise ValueError(f"Unknown reg_type '{reg_type}'. Expected 'linear', 'logit', or 'probit'.")
        
        tracker.stop()
        
    def load_emissions_report(self) -> pd.DataFrame:
        """
        Load the emissions report logged by CodeCarbon during the fit process.

        Returns:
            A pandas DataFrame with energy consumption and emissions data.
        """
        
        return pd.read_csv('eco_retina_emissions.csv')
    
    def predict(self, X: NDArray, confidence:float=0.95 ) -> NDArray:
        """
        Predict using the trained EcoRETINA model and compute prediction intervals.

        Args:
            X: Input data to predict (single sample or batch).
            confidence: Confidence level for the prediction and confidence intervals.

        Returns:
            Predicted values for the input data.
        """
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        precomputed_powers = precompute_powers(X, self.params)
        X = generate_features(X, self.combinations, precomputed_powers, self.params, self.chunk_size)
        y_pred=self.sm_model.predict(X)
        self.pi_lower, self.pi_upper, self.ci_lower, self.ci_upper=get_intervals(y_train=self.y, X_train=self.X, beta=self.sm_model.params.values, X_new=X, confidence=confidence )

        return y_pred






    
       