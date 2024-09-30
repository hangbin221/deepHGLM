import os, random, time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
from scipy.special import polygamma, loggamma
from lifelines.utils import concordance_index

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.random.set_seed(seed)
    
class config:
    seed = 42
    device = "cuda:0" 

def brier_score(y, d, mu):

    time_range = np.arange(0, max(y), 0.1)

    temp_df = pd.DataFrame()
    temp_df['y'], temp_df['d'], temp_df['mu'] = y, d, mu
    temp_df.sort_values(by=['y'])
    y, d, mu = np.array(temp_df['y']), np.array(temp_df['d']), np.array(temp_df['mu'])
    
    # y, d

    if 0 in d: y_unique_pos, tie_count_pos = np.delete(np.unique(y*d, return_counts=True), 0, 1)
    else: y_unique_pos, tie_count_pos = np.unique(y*d, return_counts=True)
    Mi_pos = np.array([yi >= y_unique_pos for yi in y])

    # y, 1-d

    if 1 in d: y_unique_neg, tie_count_neg = np.delete(np.unique(y*(1-d), return_counts=True), 0, 1)
    else: y_unique_neg, tie_count_neg = np.unique(y*(1-d), return_counts=True)
    Mi_neg = np.array([yi >= y_unique_neg for yi in y])

    Ak_pos = tie_count_pos / np.sum(Mi_pos, axis=0)
    Ak_neg = tie_count_neg / np.sum(Mi_neg, axis=0)
    Gx_pos = np.exp(Mi_pos@(np.log(1-Ak_pos+1e-7)))
    Gx_neg = np.exp(Mi_neg@(np.log(1-Ak_neg+1e-7)))  
    Gx_haz = Mi_pos@ (tie_count_pos/(mu@Mi_pos))

    Gt0 = np.ones(np.shape(time_range))
    for yi in y_unique_neg:
        Gt0[time_range>=yi] = Gx_neg[y==yi][-1]
    St0  = np.ones(np.shape(time_range))
    Lt0 = np.zeros(np.shape(time_range))
    for yi in y_unique_pos:
        St0[time_range>=yi] = Gx_pos[y==yi][-1]
        Lt0[time_range>=yi] = Gx_haz[y==yi][-1]

    BS0, BSc = [], []
    for t in range(len(time_range)):
        ot = np.where(y > time_range[t], 1, 0)
        wt = (1-ot)*d/Gx_neg + ot/Gt0[t]
        Stc = np.exp(-Lt0[t])        
        BS0.append(np.mean(((ot-St0[t])**2)*wt))
        BSc.append(np.mean(((ot-Stc**mu)**2)*wt))
        
    ibs0,ibsc0 = [], []        
    for idx in range(len(time_range)-1):
        ibs0.append( np.diff(time_range)[idx]*((BS0[idx]+BS0[idx+1])/2))
        ibsc0.append(np.diff(time_range)[idx]*((BSc[idx]+BSc[idx+1])/2))
     
    ibs  = sum(ibs0) /(max(time_range)-min(time_range))  
    ibsc = sum(ibsc0)/(max(time_range)-min(time_range))
    
    result = {'t0' : time_range, 'Reference' : BS0, 'Brier Score' : BSc, 'Reference_ibs' : ibs, 'IBS' : ibsc}        
    return result

def cumul_hazard(y, d, xb, zv):
    if 0 in d: y_unique, tie_count = np.delete(np.unique(y*d, return_counts=True), 0, 1)
    else: y_unique, tie_count = np.unique(y*d, return_counts=True)
    Mi = np.array([yi >= y_unique for yi in y])
    res = Mi@(tie_count/(np.exp(xb+zv).T@Mi)[0])
    return res

def d1_lambda(lam, y_sum, mu_sum):
    out = (
        polygamma(0, (y_sum+(1/lam))) - polygamma(0, (1/lam)) + np.log((1/lam)) + 1 
        - np.log(mu_sum+(1/lam)) - (y_sum+(1/lam))/(mu_sum+(1/lam))
    )
    out = - (1/lam**2) * out
    return np.sum(out)

def d2_lambda(lam, y_sum, mu_sum):
    out = (
        polygamma(1, (y_sum+(1/lam))) - polygamma(1, (1/lam)) + lam - 1/(mu_sum+(1/lam)) 
        - (mu_sum-y_sum)/((mu_sum+(1/lam))**2)
    )
    out = (1/lam**4) * out    
    return np.sum(out)

def gf_hlik_loss(mean_model, X, Z, y, d, d_sum, cluster_size, lam, disp):
    
    # cluster size is a vector of cluster sizes
    # canonicalizer is a vector of ci(lam; d)
    # X, Z, y, d are from the mini-batch
    # d_sum and cluster_size are from the whole train set
    
    xb, zv = mean_model([X, Z])
    mu = K.flatten(K.exp(xb+zv))

    sort_index = K.reverse(tf.nn.top_k(y, k=np.shape(y)[0], sorted=True).indices, axes=0)
    y_sort = K.gather(reference=y,  indices=sort_index)
    d_sort = K.gather(reference=d,  indices=sort_index)
    y_pred = K.gather(reference=mu, indices=sort_index)
    
    tie_count = np.unique(y_sort[d_sort==1], return_counts=True)[1]
    tie_count = tf.convert_to_tensor(tie_count.reshape(1, -1), dtype=tf.float32)    
    # if 0 in d_sort: tie_count = K.expand_dims(tf.unique_with_counts(y_sort*d_sort).count[1:],0)
    # else: tie_count = K.expand_dims(tf.unique_with_counts(y_sort*d_sort).count,0)

    ind_mat = tf.cast(K.expand_dims(y_sort,0) == K.expand_dims(np.unique(y_sort), 1), tf.float32)
    cum_haz = tf.linalg.band_part(K.ones((K.shape(ind_mat)[0], K.shape(ind_mat)[0])), 0, -1)@ind_mat@K.expand_dims(y_pred)
    hlik= (K.expand_dims(d,0)@(xb+zv) - tie_count@K.expand_dims(K.log(cum_haz[ind_mat@K.expand_dims(y_pred*d_sort)!=0]))) / np.shape(y)[0]
        
    # f(v)    
    hlik += K.sum((zv/lam - K.exp(zv)/lam)/(Z@cluster_size)) / np.shape(y)[0]
    
    # canonicalizer        
    if disp:
        hlik += K.sum((
            - K.log(lam)/lam - tf.math.lgamma(1/lam)
            + Z@(-(d_sum+1/lam)*K.log(d_sum+1/lam) + d_sum+1/lam + tf.math.lgamma(d_sum+1/lam))) 
            / (Z@cluster_size)
        ) / np.shape(y)[0]
    
    loss = - hlik
    return loss

def gf_update_params(mean_model, X, Z, y, d, d_sum, cluster_size, lam, optimizer, disp):  
    with tf.GradientTape() as tape:
        loss = gf_hlik_loss(mean_model, X, Z, y, d, d_sum, cluster_size, lam, disp)
    if disp:
        gradients = tape.gradient(loss, mean_model.trainable_weights+[lam])
        optimizer.apply_gradients(zip(gradients, mean_model.trainable_weights+[lam])) 
    else:
        gradients = tape.gradient(loss, mean_model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, mean_model.trainable_weights))
    return loss

def gf_train_one_epoch(mean_model, train_batch, d_sum, cluster_size, lam, optimizer, disp):    
    losses = [] 
    for step, (X_batch, Z_batch, y_batch, d_batch) in enumerate(train_batch):
        loss = gf_update_params(mean_model, X_batch, Z_batch, y_batch, d_batch, d_sum, cluster_size, lam, optimizer, disp)
        losses.append(loss)
    return losses

def find_rand_var(mean_model, X, Z, y, d, lam_init=0.5, maxiter=1000, threshold=1e-5):
    lam = lam_init
    xb, zv = mean_model([X, Z])
    mu_sum = Z.T@(cumul_hazard(y, d, xb, zv)*np.exp(xb).T[0])
    y_sum = (Z.T@y).reshape(-1,1)
    for _ in range(maxiter):
        update = d1_lambda(lam, y_sum, mu_sum)/d2_lambda(lam, y_sum, mu_sum)
        lam -= update
        if lam < 0:
            lam.assign(1e-18)
        if np.abs(update) < threshold:
            break
    return np.float32(lam)

def gf_train_model(
    mean_model, train_batch, train_data, validation_data, optimizer, lam_init, 
    pretrain=20, patience=20, max_epochs=1000, disp_method='Newton', disp_update = 1, 
    monitor = 'loss', pretrain_optimizer = None, wts_init=None, seed=42):

    # disp_method is one of 'Newton' and 'SGD'
    # monitor is one of 'loss', 'bscore', 'cindex' (for early stopping criteria)
    # If pretrain_optimizer is None, it is same with the optimizer    

    X_train, Z_train, y_train, d_train = train_data
    X_valid, Z_valid, y_valid, d_valid = validation_data
    N_train, N_valid = np.shape(y_train)[0], np.shape(y_valid)[0]
    
    cluster_size_train = np.diag(Z_train.T@Z_train).reshape(-1,1)
    cluster_size_valid = np.diag(Z_valid.T@Z_valid).reshape(-1,1)
    y_sum_train, y_sum_valid = (Z_train.T@y_train).reshape(-1,1), (Z_valid.T@y_valid).reshape(-1,1)
    d_sum_train, d_sum_valid = (Z_train.T@d_train).reshape(-1,1), (Z_valid.T@d_valid).reshape(-1,1)    
       
    if wts_init is not None:
        mean_model.set_weights(wts_init)
    lam = tf.Variable(lam_init, name='lam', trainable=True, constraint=lambda x: tf.clip_by_value(x, 1e-18, np.infty))

    train_losses, valid_losses, train_monitor, valid_monitor = [np.zeros(max_epochs) for _ in range(4)]    
    lam_history, compute_time = [np.zeros(max_epochs) for _ in range(2)]

    if pretrain_optimizer is None: pretrain_optimizer = optimizer
    for epoch in range(pretrain):
        train_loss = gf_train_one_epoch(mean_model, train_batch, d_sum_train, cluster_size_train, lam, pretrain_optimizer, disp=False)
    # min_valid_loss = gf_hlik_loss(mean_model, X_valid, Z_valid, y_valid, d_valid, d_sum_valid, cluster_size_valid, lam, disp)
    
    min_valid_measure = np.infty
    patience_count = 0
    
    temp_start = time.time()
    for epoch in range(max_epochs):
        if epoch%disp_update == 0:
            if disp_method == 'Newton':
                train_loss = gf_train_one_epoch(mean_model, train_batch, d_sum_train, cluster_size_train, lam, optimizer, disp=False)                
                train_losses[epoch] = np.mean(train_loss)
                lam.assign(find_rand_var(mean_model, X_train, Z_train, y_train, d_train, lam_init=float(lam), maxiter=1000, threshold=1e-5))
            if disp_method == 'SGD':
                # Note that this method returns the loss with disp=True. 
                # Thus, if disp_update is not 1, it should be avoided to compare this loss with losses from other epochs.
                train_loss = gf_train_one_epoch(mean_model, train_batch, d_sum_train, cluster_size_train, lam, optimizer, disp=True)
                train_losses[epoch] = np.mean(train_loss)
        else:
            train_loss = gf_train_one_epoch(mean_model, train_batch, d_sum_train, cluster_size_train, lam, optimizer, disp=False)
            train_losses[epoch] = np.mean(train_loss)

        wts = mean_model.get_weights()
        valid_loss = gf_hlik_loss(mean_model, X_valid, Z_valid, y_valid, d_valid, d_sum_valid, cluster_size_valid, lam, disp=True)
        valid_losses[epoch] = np.float32(valid_loss)[0][0]
        
        compute_time[epoch] = time.time() - temp_start
        lam_history[epoch] = np.float32(lam)

        if monitor is 'loss':
            train_monitor[epoch] = train_losses[epoch]
            valid_monitor[epoch] = valid_losses[epoch]
            valid_measure = valid_monitor[epoch]
        else:
            xb_train, zv_train = mean_model([X_train, Z_train])
            xb_valid, zv_valid = mean_model([X_valid, Z_valid])
            mu_train = np.exp(xb_train+zv_train)
            mu_valid = np.exp(xb_valid+zv_valid)            
            if monitor == 'bscore':
                train_monitor[epoch] = brier_score(y_train, d_train, mu_train)['IBS']
                valid_monitor[epoch] = brier_score(y_valid, d_valid, mu_valid)['IBS']
                valid_measure = valid_monitor[epoch]
            if monitor == 'cindex':
                train_monitor[epoch] = concordance_index(y_train, -mu_train, event_observed = d_train)
                valid_monitor[epoch] = concordance_index(y_valid, -mu_valid, event_observed = d_valid)
                valid_measure = -valid_monitor[epoch]

        if valid_measure > min_valid_measure:
            patience_count += 1
            if patience_count == patience: break
        else: 
            min_valid_measure = valid_measure
            patience_count = 0

    res = {
        "wts" : wts,
        "lam" : np.float32(lam),
        "train_loss"  : train_losses[:(epoch+1)],
        "valid_loss"  : valid_losses[:(epoch+1)],
        "train_monitor"  : train_monitor[:(epoch+1)],
        "valid_monitor"  : valid_monitor[:(epoch+1)],
        "lam_history" : lam_history[:(epoch+1)],
        "compute_time": compute_time[:(epoch+1)]
    }
    
    return res

def coxph_loss(mean_model, X, y, d):

    mu = K.exp(K.transpose(mean_model([X])))

    sort_index = K.reverse(tf.nn.top_k(y, k=np.shape(y)[0], sorted=True).indices, axes=0)
    y_sort = K.gather(reference=y,  indices=sort_index)
    d_sort = K.gather(reference=d,  indices=sort_index)
    y_pred = K.gather(reference=mu, indices=sort_index)

    yd = y_sort*d_sort
    tie_count = tf.cast(tf.unique_with_counts(
            tf.boolean_mask(yd, tf.greater(yd, 0))
        ).count, dtype = tf.float32)


    ind_matrix = K.expand_dims(y_sort, 0) - K.expand_dims(y_sort, 1)
    ind_matrix = K.equal(x = ind_matrix, y = K.zeros_like(ind_matrix))
    ind_matrix = K.cast(x = ind_matrix, dtype = tf.float32)
    
    time_count = K.cumsum(tf.unique_with_counts(y_sort).count)
    time_count = K.cast(time_count - K.ones_like(time_count), dtype = tf.int32)

    ind_matrix = K.gather(ind_matrix, time_count)
    
    tie_haz = y_pred * d_sort
    tie_haz = K.dot(ind_matrix, K.expand_dims(tie_haz)) 
    event_index = tf.math.not_equal(tie_haz,0) 
    tie_haz = tf.boolean_mask(tie_haz, event_index)
    
    tie_risk = K.log(y_pred) * d_sort
    tie_risk = K.dot(ind_matrix, K.expand_dims(tie_risk))
    tie_risk = tf.boolean_mask(tie_risk, event_index)
    
    cum_haz = K.dot(ind_matrix, K.expand_dims(y_pred))
    cum_haz = K.reverse(tf.cumsum(K.reverse(cum_haz, axes = 0)), axes = 0)
    cum_haz = tf.boolean_mask(cum_haz, event_index)

    plik = tf.math.reduce_sum(tie_risk)-tf.math.reduce_sum(tie_count*tf.math.log(cum_haz))    
                   
    return -plik