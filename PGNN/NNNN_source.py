import os, random, time
import numpy as np
import tensorflow as tf
from tensorflow.linalg import slogdet, det, inv, solve, cholesky, cholesky_solve
from tensorflow.keras import backend as K
from scipy.special import polygamma, loggamma
from scipy.stats import norm

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.random.set_seed(seed)
    
class config:
    seed = 42
    device = "cuda:0" 

def nn_hlik_loss(mean_model, X, Z, y, ni, phi, lam, disp=True, batch_ratio=1.):
    
    # ni = np.diag(ZTZ) is the vector of n1, n2, ...
    
    # f(y|v)
    mu = K.transpose(K.sum(mean_model([X, Z]), axis=0))
    hlik = - K.sum(K.square(y - mu)) / phi / np.shape(y)[0]    
    
    # f(v)
    v = mean_model.weights[-1]
    hlik += - K.sum(K.square(v)) / lam * batch_ratio / np.shape(y)[0]
    
    # c(lam; y)
    if disp: 
        hlik += - (K.sum(K.log(lam*ni+phi)) + (np.shape(Z)[0]-np.shape(Z)[1])*K.log(phi)) * batch_ratio / np.shape(Z)[0]

    loss = - hlik
    return loss

def update_params(mean_model, X, Z, y, ni, loss_ftn, optimizer, phi, lam, disp=True, batch_ratio=1.):  
    with tf.GradientTape() as tape:
        loss = loss_ftn(mean_model, X, Z, y, ni, phi, lam, disp, batch_ratio)
    if disp:
        gradients = tape.gradient(loss, mean_model.trainable_weights+[phi, lam])
        optimizer.apply_gradients(zip(gradients, mean_model.trainable_weights+[phi, lam])) 
    else:
        gradients = tape.gradient(loss, mean_model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, mean_model.trainable_weights))
    return loss

def train_one_epoch(mean_model, train_batch, loss_ftn, optimizer, phi, lam, disp=True, batch_ratio=1.):
    losses = [] 
    for step, (X_batch, Z_batch, y_batch) in enumerate(train_batch):
        ni_batch = K.transpose(Z_batch)@K.ones((np.shape(y_batch)[0],1))
        loss = update_params(mean_model, X_batch, Z_batch, y_batch, ni_batch, loss_ftn, optimizer, phi, lam, disp, batch_ratio)
        losses.append(loss)
    return losses

def adjust_wts(wts):    
    v_adjs = - np.mean(wts[-1])
    wts[-2] = wts[-2] - v_adjs
    wts[-1] = wts[-1] + v_adjs        
    return wts

def train_model(mean_model, train_batch, train_data, validation_data, 
                loss, optimizer, phi_init, lam_init, batch_ratio=1., 
                patience=0, pretrain=0, max_epochs=1000, moments_epochs=0,
                adjust=False, wts_init=None, seed=42):

    X_train, Z_train, y_train = train_data
    X_valid, Z_valid, y_valid = validation_data
    N_train, N_valid = np.shape(y_train)[0], np.shape(y_valid)[0]
    
    K.clear_session()
    seed_everything(seed)
    
    if wts_init is not None:
        mean_model.set_weights(wts_init)
    phi = tf.Variable(phi_init, name='phi', trainable=True, constraint=lambda x: tf.clip_by_value(x, 1e-18, np.infty))
    lam = tf.Variable(lam_init, name='lam', trainable=True, constraint=lambda x: tf.clip_by_value(x, 1e-18, np.infty))
    disp = True

    train_losses, valid_losses = [], []
    train_mse, valid_mse, phi_history, lam_history, compute_time = [np.zeros(max_epochs) for _ in range(5)]

    ni_valid = np.diag(Z_valid.T@Z_valid)
    min_valid_loss = np.infty
    for epoch in range(pretrain):
        train_loss = train_one_epoch(mean_model, train_batch, loss, optimizer, phi, lam, disp=False, batch_ratio=batch_ratio)
    
    patience_count = 0    
    temp_start = time.time()
    for epoch in range(max_epochs):

        if epoch < moments_epochs:
            train_loss = train_one_epoch(mean_model, train_batch, loss, optimizer, phi, lam, disp=False, batch_ratio=batch_ratio)
            out_train = mean_model([X_train, Z_train])
            wts = mean_model.get_weights()
            phi.assign(np.var(y_train-np.sum(out_train, axis=0).T))
            lam.assign(np.var(wts[-1]))
        else:
            train_loss = train_one_epoch(mean_model, train_batch, loss, optimizer, phi, lam, disp, batch_ratio=batch_ratio)
            if np.isnan(lam): 
                lam.assign(1e-18)
                disp = False
            out_train = mean_model([X_train, Z_train])
            wts = mean_model.get_weights()
            if adjust:
                wts = adjust_wts(wts)
                mean_model.set_weights(wts)
        valid_loss = nn_hlik_loss(mean_model, X_valid, Z_valid, y_valid, ni_valid, phi, lam, disp=True, batch_ratio=batch_ratio)
        out_valid = mean_model([X_valid, Z_valid])

        mu_train = np.sum(out_train, axis=0).T
        mu_valid = np.sum(out_valid, axis=0).T

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        train_mse[epoch] = np.mean((y_train-mu_train)**2)
        valid_mse[epoch] = np.mean((y_valid-mu_valid)**2)
        phi_history[epoch] = phi
        lam_history[epoch] = lam
        compute_time[epoch] = (time.time() - temp_start)

        if valid_loss > min_valid_loss:
            patience_count += 1
            if patience_count == patience: break
        else:
            min_valid_loss = valid_loss

    res = {
        "wts" : wts,
        "phi" : np.float32(phi),
        "lam" : np.float32(lam),
        "train_loss": train_losses[:(epoch+1)],
        "valid_loss": valid_losses[:(epoch+1)],
        "train_mse" : train_mse[:(epoch+1)],
        "valid_mse" : valid_mse[:(epoch+1)],
        "phi_history"  : phi_history[:(epoch+1)],
        "lam_history"  : lam_history[:(epoch+1)],
        "compute_time" : compute_time[:(epoch+1)]
    }
    
    return res
