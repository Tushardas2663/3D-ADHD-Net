import numpy as np
from scipy.stats import spearmanr
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
from sklearn.feature_selection import mutual_info_regression
from tensorflow.keras.models import Model
import os
import zipfile
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf

from tensorflow.keras.layers import (
    Input, Conv3D, BatchNormalization, MaxPooling3D, 
    Reshape, LSTM, Dense, Dropout, Layer, Bidirectional,
    MultiHeadAttention, LayerNormalization, Flatten
)
from tensorflow.keras import backend as K

class DeepTrace:
    def __init__(self, model, layer_map):
        """
        DeepTrace Interpretability Framework.
        
        Args:
            model: Trained Keras model instance.
            layer_map: Dictionary mapping logical names to internal layer names.
                       Order matters. Must be [Deepest Layer -> ... -> Shallowest Layer].
                       e.g. {'Transformer': 'trans_output', 'LSTM': 'lstm_out', 'CNN': 'conv_out'}
        """
        self.model = model
        self.layer_map = layer_map

    def select_balanced_top_k(self,correlations, k):
        indices = np.arange(len(correlations))
        pos_mask = correlations > 0
        neg_mask = correlations < 0
        pos_indices = indices[pos_mask]
        neg_indices = indices[neg_mask]
        
        pos_indices = pos_indices[np.argsort(np.abs(correlations[pos_indices]))[::-1]]
        neg_indices = neg_indices[np.argsort(np.abs(correlations[neg_indices]))[::-1]]
        
        k_pos_target = k // 2
        k_neg_target = k - k_pos_target 
        
        selected_pos = pos_indices[:k_pos_target]
        selected_neg = neg_indices[:k_neg_target]
        final_selection = list(selected_pos) + list(selected_neg)
        
        missing_slots = k - len(final_selection)
        if missing_slots > 0:
            remaining_pos = pos_indices[k_pos_target:]
            remaining_neg = neg_indices[k_neg_target:]
            remaining_pool = np.concatenate([remaining_pos, remaining_neg])
            if len(remaining_pool) > 0:
                remaining_pool = remaining_pool[np.argsort(np.abs(correlations[remaining_pool]))[::-1]]
                final_selection.extend(remaining_pool[:missing_slots])
                
        return np.array(final_selection)

    

    def compute_heatmap(self,X_analysis,K_COHORT=15):
        """
        Generates the DeepTrace diagnostic heatmap.
        
        Args:
            X_analysis: Input data array (Batch, H, W, Time).
            K_COHORT: Number of features to trace at each layer.
        
        Returns:
            heatmap: 2D numpy array representing directional importance.
        """


        outputs = [self.model.get_layer(name).output for name in self.layer_map.values()]
        outputs.append(self.model.output) 
        trace_model = Model(inputs=self.model.input, outputs=outputs)
        results = trace_model.predict(X_analysis, verbose=1)

        activations_raw = results[:-1]
        y_probs = results[-1].flatten()

        def pool_activations(tensor):
            if tensor.ndim == 3: return np.mean(tensor, axis=1) 
            elif tensor.ndim == 5: return np.mean(tensor, axis=(1, 2, 3)) 
            return tensor

        A_pooled = {name: pool_activations(act) for name, act in zip(self.layer_map.keys(), activations_raw)}




        first_layer=list(self.layer_map.keys())[0]
        trans_feats = A_pooled[first_layer]
        n_feats = trans_feats.shape[1]


        anchor_scores = []
        for f in range(n_feats):
            rho, _ = spearmanr(trans_feats[:, f], y_probs)
            anchor_scores.append(0 if np.isnan(rho) else rho)
        anchor_scores = np.array(anchor_scores)


        current_indices = self.select_balanced_top_k(anchor_scores, K_COHORT)

    
        current_cohort = []
        for idx in current_indices:
            direction = np.sign(anchor_scores[idx])
            current_cohort.append({
                'signal': trans_feats[:, idx]*direction, 
                'id': f"Trans({idx})"
            })

    

      
        layer_sequence  = list(self.layer_map.keys())
        prev_layer_name = "Transformer Cohort"

        

        consistency_scores = {}

        for i, layer_name in enumerate(layer_sequence[1:]):
            current_feats = A_pooled[layer_name]
            n_filters = current_feats.shape[1]
            
          
            layer_best_corrs = [] 
            layer_best_mis = [] 
            
            for f in range(n_filters):
                feat_signal = current_feats[:, f]
                max_r = 0
                best_target_signal = None
                
             
                for target in current_cohort:
                    r, _ = spearmanr(target['signal'], feat_signal)
                    if np.isnan(r): r = 0
                    if abs(r) > abs(max_r):
                        max_r = r
                        best_target_signal = target['signal']
                
                layer_best_corrs.append(max_r)
                
              
                if best_target_signal is not None:
              
                    mi_score = mutual_info_regression(feat_signal.reshape(-1, 1), best_target_signal, random_state=42)[0]
                    layer_best_mis.append(mi_score)
                else:
                    layer_best_mis.append(0)
            
            layer_best_corrs = np.array(layer_best_corrs)
            layer_best_mis = np.array(layer_best_mis)
            
        
          
            consistency_r, _ = pearsonr(np.abs(layer_best_corrs), layer_best_mis)
            consistency_scores[layer_name] = consistency_r
            print(f"[{layer_name}] Spearman-MI Consistency Score: {consistency_r:.4f}")
            
         
            top_k_indices = self.select_balanced_top_k(layer_best_corrs, K_COHORT)
            
            
            next_cohort = []
            for idx in top_k_indices:
                r_val = layer_best_corrs[idx]
                direction = np.sign(r_val)
                next_cohort.append({
                    'signal': current_feats[:, idx]*direction, 
                    'id': f"{layer_name}({idx})"
                })
            current_cohort = next_cohort
            
            

        
       
        X_spatial = np.mean(X_analysis, axis=3).squeeze()
        h, w = X_spatial.shape[1], X_spatial.shape[2]

       
        spearman_heatmap = np.zeros((h, w))
       

        

        for item in current_cohort:
            target_vec = item['signal']
            path_map_sp = np.zeros((h, w))
           
            
            for r in range(h):
                for c in range(w):
                    pixel_vec = X_spatial[:, r, c]
                    if np.std(pixel_vec) > 1e-9:
                      
                        rho, _ = spearmanr(target_vec, pixel_vec)
                        path_map_sp[r, c] = rho if not np.isnan(rho) else 0
                        
                       
            spearman_heatmap += path_map_sp
           

        spearman_heatmap /= K_COHORT
     
        return spearman_heatmap
