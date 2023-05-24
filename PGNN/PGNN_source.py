import os, random, time
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from scipy.special import polygamma, loggamma

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.random.set_seed(seed)    
class config:
    seed = 42
    device = "cuda:0" 


def pg_hlik_loss(mean_model, X, Z, y, lam, disp, batch_ratio):
    
    # f(y|v)
    log_mu = K.transpose(K.sum(mean_model([X, Z]), axis=0))
    hlik = K.sum(y*log_mu-K.exp(log_mu))/np.shape(y)[0] 
    
    # f(v)
    v = mean_model.weights[-1]
    hlik += K.sum((v - K.exp(v) - K.log(lam))/lam - tf.math.lgamma(1/lam)) * batch_ratio/np.shape(y)[0] 
    
    # c(lam; y)
    if disp:
        y_sum = K.transpose(Z)@K.reshape(y, (np.shape(y)[0],1)) + 1/lam
        hlik += K.sum(- y_sum*K.log(y_sum) + y_sum + tf.math.lgamma(y_sum)) * batch_ratio/np.shape(y)[0] 
    
    loss = - hlik
    return loss

def update_params(mean_model, X, Z, y, loss_ftn, optimizer, lam, disp, batch_ratio):  
    with tf.GradientTape() as tape:
        loss = loss_ftn(mean_model, X, Z, y, lam, disp, batch_ratio)
    if disp:
        gradients = tape.gradient(loss, mean_model.trainable_weights+[lam])
        optimizer.apply_gradients(zip(gradients, mean_model.trainable_weights+[lam])) 
    else:
        gradients = tape.gradient(loss, mean_model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, mean_model.trainable_weights))
    return loss

def train_one_epoch(mean_model, train_batch, loss_ftn, optimizer, lam, disp, batch_ratio):    
    losses = [] 
    for step, (X_batch, Z_batch, y_batch) in enumerate(train_batch):
        loss = update_params(mean_model, X_batch, Z_batch, y_batch, loss_ftn, optimizer, lam, disp, batch_ratio)
        losses.append(loss)
    return losses

def adjust_wts(wts, lam):    
    v_adjs = - np.log(np.mean(np.exp(wts[-1])))
    wts[-2] = wts[-2] - v_adjs
    wts[-1] = wts[-1] + v_adjs        
    return wts

def lambda_mme(mu_sum, v_pred):
    # mu_sum = Z_train.T@np.exp(mean_model([X_train, Z_train])[0])
    lam = np.mean((np.exp(v_pred)-1)**2)*(0.5+np.sqrt(0.25+np.shape(v_pred)[0]*np.sum((np.exp(v_pred)-1)**2/mu_sum)/(np.sum((np.exp(v_pred)-1)**2)**2)))
    if np.isnan(lam): lam = np.var(v_pred)
    return lam

def train_model(
    mean_model, train_batch, train_data, validation_data, loss, optimizer, lam_init, 
    batch_ratio=1., patience=0, pretrain=0, max_epochs=1000, moments_epochs=0, 
    adjust=True, wts_init=None, seed=42):

    X_train, Z_train, y_train = train_data
    X_valid, Z_valid, y_valid = validation_data
    N_train, N_valid = np.shape(y_train)[0], np.shape(y_valid)[0]
    
    K.clear_session()
    seed_everything(seed)
    
    if wts_init is not None:
        mean_model.set_weights(wts_init)
    lam = tf.Variable(lam_init, name='lam', trainable=True, constraint=lambda x: tf.clip_by_value(x, 1e-18, np.infty))
    disp = True

    train_losses, valid_losses = [], []
    train_mse, valid_mse, lam_history, compute_time = [np.zeros(max_epochs) for _ in range(4)]

    min_valid_loss = np.infty
    for epoch in range(pretrain):
        train_loss = train_one_epoch(mean_model, train_batch, loss, optimizer, lam, disp=False, batch_ratio=batch_ratio)
        valid_loss = pg_hlik_loss(mean_model, X_valid, Z_valid, y_valid, lam, disp=True, batch_ratio=batch_ratio)
        min_valid_loss = valid_loss
    
    patience_count = 0
    temp_start = time.time()
    for epoch in range(max_epochs):

        if epoch < moments_epochs:
            train_loss = train_one_epoch(mean_model, train_batch, loss, optimizer, lam, disp=False, batch_ratio=batch_ratio)
            out_train = mean_model([X_train, Z_train])
            wts = mean_model.get_weights()
            mu_sum = Z_train.T@np.exp(out_train[0])
            lam.assign(lambda_mme(mu_sum, wts[-1]))         
        else:
            train_loss = train_one_epoch(mean_model, train_batch, loss, optimizer, lam, disp, batch_ratio=batch_ratio)
            if np.isnan(lam): disp = False
            if not disp: lam.assign(lambda_mme(mu_sum, wts[-1]))
            out_train = mean_model([X_train, Z_train])
            wts = mean_model.get_weights()
            if adjust:
                wts = adjust_wts(wts, lam)
                mean_model.set_weights(wts)
        valid_loss = pg_hlik_loss(mean_model, X_valid, Z_valid, y_valid, lam, disp=True, batch_ratio=batch_ratio)
        out_valid = mean_model([X_valid, Z_valid])

        mu_train = np.exp(np.sum(out_train, axis=0).T)
        mu_valid = np.exp(np.sum(out_valid, axis=0).T)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        train_mse[epoch] = np.mean((y_train-mu_train)**2)
        valid_mse[epoch] = np.mean((y_valid-mu_valid)**2)
        lam_history[epoch] = lam
        compute_time[epoch] = (time.time() - temp_start)

        if valid_loss > min_valid_loss:
            patience_count += 1
            if patience_count == patience: break
        else:
            min_valid_loss = valid_loss

    res = {
        "wts" : wts,
        "lam" : np.float32(lam),
        "train_loss": train_losses[:(epoch+1)],
        "valid_loss": valid_losses[:(epoch+1)],
        "train_mse" : train_mse[:(epoch+1)],
        "valid_mse" : valid_mse[:(epoch+1)],
        "lam_history"  : lam_history[:(epoch+1)],
        "compute_time" : compute_time[:(epoch+1)]
    }
    
    return res
