import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv3D, BatchNormalization, MaxPooling3D, 
    Reshape, LSTM, Dense, Dropout, Layer, Bidirectional,
    MultiHeadAttention, LayerNormalization, Flatten
)
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, accuracy_score
)



class PositionalEmbedding(Layer):
    def __init__(self, sequence_length, output_dim, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)
        self.sequence_length = sequence_length; self.output_dim = output_dim
        self.position_embeddings = tf.keras.layers.Embedding(input_dim=sequence_length, output_dim=output_dim)
        self.supports_masking = True
    def call(self, inputs):
        positions = tf.range(start=0, limit=self.sequence_length, delta=1)
        embedded_positions = self.position_embeddings(positions); return inputs + embedded_positions
    def get_config(self):
        config = super(PositionalEmbedding, self).get_config()
        config.update({"sequence_length": self.sequence_length,"output_dim": self.output_dim}); return config

class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim; self.num_heads = num_heads; self.ff_dim = ff_dim; self.rate = rate
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([Dense(ff_dim, activation="relu"), Dense(embed_dim),])
        self.layernorm1 = LayerNormalization(epsilon=1e-6); self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate); self.dropout2 = Dropout(rate)
    def call(self, inputs, training=None):
        attn_output = self.att(inputs, inputs); attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output); ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training); return self.layernorm2(out1 + ffn_output)
    def get_config(self):
        config = super(TransformerBlock, self).get_config()
        config.update({"embed_dim": self.embed_dim, "num_heads": self.num_heads, "ff_dim": self.ff_dim, "rate": self.rate}); return config


def cnn_lstm_transformer_block(input_tensor, name_prefix, transformer_embed_dim=64, 
                               transformer_num_heads=4, transformer_ff_dim=128, 
                               transformer_blocks=1, dropout_rate=0.3):
    

    x = Conv3D(16, (3, 3, 3), padding='same', activation='relu', name=f'{name_prefix}_conv3d_1')(input_tensor)
    x = BatchNormalization(name=f'{name_prefix}_bn_1')(x)
    x = MaxPooling3D((2, 2, 2), name=f'{name_prefix}_pool_1')(x)
    
    x = Conv3D(32, (3, 3, 3), padding='same', activation='relu', name=f'{name_prefix}_conv3d_2')(x)
    x = BatchNormalization(name=f'{name_prefix}_bn_2')(x)
    x = MaxPooling3D((2, 2, 2), name=f'{name_prefix}_pool_2')(x)
    
    x = Conv3D(64, (3, 3, 3), padding='same', activation='relu', name=f'{name_prefix}_conv3d_3')(x)
    x = BatchNormalization(name=f'{name_prefix}_bn_3')(x)
    x = MaxPooling3D((2, 2, 2), name=f'{name_prefix}_pool_3')(x)
    

    shape_after_conv = K.int_shape(x)
    t_reduced = shape_after_conv[3] if shape_after_conv[3] is not None else -1
    if t_reduced == 0: t_reduced = 1 
    features_for_lstm = shape_after_conv[1] * shape_after_conv[2] * shape_after_conv[4]
    x = Reshape(target_shape=(t_reduced, features_for_lstm), name=f'{name_prefix}_reshape')(x)
    


    
    if USE_TRANSFORMER:
        x_lstm_output = Bidirectional(LSTM(units=transformer_embed_dim // 2, return_sequences=True), name=f'{name_prefix}_bilstm_1')(x)
        

        if x_lstm_output.shape[-1] != transformer_embed_dim:
             x_lstm_output = Dense(transformer_embed_dim, activation='relu', name=f'{name_prefix}_lstm_output_dense')(x_lstm_output)
        

        x_trans = PositionalEmbedding(t_reduced, transformer_embed_dim, name=f'{name_prefix}_pos_emb')(x_lstm_output)
        for i in range(transformer_blocks):
            x_trans = TransformerBlock(transformer_embed_dim, transformer_num_heads, transformer_ff_dim, rate=dropout_rate, name=f'{name_prefix}_trans_{i}')(x_trans)
        
        return Flatten(name=f'{name_prefix}_flatten')(x_trans)
    
    else:
      
        x_lstm_output = Bidirectional(LSTM(units=transformer_embed_dim // 2, return_sequences=False), name=f'{name_prefix}_bilstm_final')(x)
        return x_lstm_output

def create_single_branch_model(input_shape, num_classes=1, dropout_rate=0.3):
    input_eeg = Input(shape=input_shape, name='raw_eeg_input')
    

    features = cnn_lstm_transformer_block(input_eeg, name_prefix='eeg_branch', dropout_rate=dropout_rate)

    x = Dense(units=32, activation='relu', name='dense_head_1')(features)
    x = Dropout(dropout_rate)(x) 
    x = Dense(units=16, activation='relu', name='dense_head_2')(x)
    x = Dropout(dropout_rate)(x)
    
    if num_classes == 1:
        output_layer = Dense(units=1, activation='sigmoid', name='output')(x)
    else:
        output_layer = Dense(units=num_classes, activation='softmax', name='output')(x)
        
    model_name = "EEG_Transformer_Model" if USE_TRANSFORMER else "EEG_CNNLSTM_Model"
    return Model(inputs=input_eeg, outputs=output_layer, name=model_name)





"""

Training guidelines: Subject-Independent 5-fold cross-validation
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
all_unique_subject_ids = np.unique(subject_ids)
all_subjects_y = np.array(all_subjects_y)

subject_to_label = {sid: all_subjects_y[list(subject_ids).index(sid)] for sid in all_unique_subject_ids}

for fold_idx, (train_val_idx, test_idx) in enumerate(kf.split(all_unique_subject_ids)):
    

    test_subject_ids = all_unique_subject_ids[test_idx]
    train_val_subject_ids = all_unique_subject_ids[train_val_idx]
    

    train_subject_ids, val_subject_ids = train_test_split(train_val_subject_ids, test_size=0.1, random_state=42)
    
    print(f"\nFold {fold_idx+1}: Train Subj={len(train_subject_ids)}, Val Subj={len(val_subject_ids)}, Test Subj={len(test_subject_ids)}")

    
 
    X_train, y_train, _ = get_data_for_subjects(all_subjects_X_eeg, all_subjects_y, subject_ids, train_subject_ids)
    X_val, y_val, _ = get_data_for_subjects(all_subjects_X_eeg, all_subjects_y, subject_ids, val_subject_ids)
    X_test, y_test, test_subj_ids_epochs = get_data_for_subjects(all_subjects_X_eeg, all_subjects_y, subject_ids, test_subject_ids)

    Model uses:  epochs=30, batch_size=32

  """
    
  
   
    
   
  

