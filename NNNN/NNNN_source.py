import time, random, os
import numpy as np
import tensorflow as tf
from tensorflow.linalg import slogdet, det, inv, solve, cholesky, cholesky_solve
from tensorflow.keras import backend as K
np.set_printoptions(precision=3,suppress=True)

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.random.set_seed(seed)
    
class config:
    seed = 42
    device = "cuda:0" 

def nn_hlik_loss_RBF(mean_model, X, ZL, y, ZLTZL, N, dist_square, phi, lamL, CL, CLinv, disp=True):
    
    # f(y|v)
    mu = K.transpose(K.sum(mean_model([X, ZL]), axis=0))
    hlik = - K.sum(K.square(y - mu)) / phi / np.shape(y)[0]
    
    # f(v)
    vL = mean_model.weights[-1]    
    hlik += - (K.transpose(vL) @ CLinv @ vL)[0,0] / lamL / N
    
    # c(phi, lam, lsq; y)
    if disp:        
        hlik += - slogdet(ZLTZL@CL*(lamL/phi) + K.eye(np.shape(ZLTZL)[0]))[1] / N
        hlik += - K.log(phi)

    loss = - hlik
    return loss

def update_params_RBF(optimizer, mean_model, X, ZL, y, ZLTZL, N, dist_square, phi, lamL, CL, CLinv, train_disp, train_lamL):
    with tf.GradientTape() as tape:
        loss = nn_hlik_loss_RBF(mean_model, X, ZL, y, ZLTZL, N, dist_square, phi, lamL, CL, CLinv, train_disp)
    trainable_wts = mean_model.trainable_weights
    if train_disp: trainable_wts += [phi]
    if train_lamL: trainable_wts += [lamL]
    gradients = tape.gradient(loss, trainable_wts)
    optimizer.apply_gradients(zip(gradients, trainable_wts))

    return loss

def train_one_epoch_RBF(train_batch, optimizer, mean_model, X, ZL, y, ZLTZL, N, dist_square, phi, lamL, CL, CLinv, train_disp=True, train_lamL=True):
    losses = [] 
    for step, (X_batch, ZL_batch, y_batch) in enumerate(train_batch):        
        loss = update_params_RBF(optimizer, mean_model, X_batch, ZL_batch, y_batch, ZLTZL, N, dist_square, phi, lamL, CL, CLinv, train_disp, train_lamL)
        losses.append(loss)
    return losses

def update_spatial_disp(lr_spatial, vL, ZLTZL, N, dist_square, phi, lamL, lsq, CL, CLinv, epsilon, train_lamL=True, train_lsq=True):
    dCLdg = - dist_square*CL
    A = ZLTZL@CL*lamL/phi + np.eye(np.shape(ZLTZL)[0], dtype=np.float32)
    if train_lsq:
        grad_lsq = ( (K.transpose(vL)@CLinv@dCLdg@CLinv@vL)[0,0]/lamL - np.trace(np.linalg.solve(A, ZLTZL@dCLdg))*lamL/phi ) / 2 / lsq / N
        log_lsq = np.log(lsq) - lr_spatial * grad_lsq
        lsq.assign(np.exp(log_lsq))
    if train_lamL:
        grad_lamL = ( (K.transpose(vL)@CLinv@vL)[0,0] + np.trace(np.linalg.solve(A, ZLTZL@CL))/phi ) * lamL / N
        log_lamL = np.log(lamL) - lr_spatial * grad_lamL
        lamL.assign(np.exp(log_lamL))
    
    CL = K.exp(-dist_square/2/lsq) + epsilon*K.eye(np.shape(dist_square)[0])
    CLinv = np.linalg.inv(CL)
    return lamL, lsq, CL, CLinv

def train_model_RBF(
    mean_model, train_batch, train_data, validation_data, dist_square,
    optimizer, lr_spatial, phi_init, lamL_init, lsq_init, phi_type='mme', lamL_type='mme', epsilon=1e-3,
    pretrain=0, moments_epochs=0, max_epochs=1000, patience = None,
    adjust=False, wts_init=None, verbose=False, seed=0):

    X_train, ZL_train, y_train = train_data
    X_valid, ZL_valid, y_valid = validation_data
    N_train, N_valid = np.shape(y_train)[0], np.shape(y_valid)[0]

    ZLTZL = ZL_train.T@ZL_train

    if patience >= max_epochs: patience = None
    if patience is not None:
        ZLTZLv = ZL_valid.T@ZL_valid

    if wts_init is not None:
        mean_model.set_weights(wts_init)

    phi  = tf.Variable(phi_init, name='phi', trainable=True, constraint=lambda x: tf.clip_by_value(x, 1e-18, np.infty), dtype=tf.float32)
    lamL = tf.Variable(lamL_init, name='lamL', trainable=True, constraint=lambda x: tf.clip_by_value(x, 1e-18, np.infty), dtype=tf.float32)
    lsq  = tf.Variable(lsq_init, name='lsq', trainable=True, constraint=lambda x: tf.clip_by_value(x, 1e-18, np.infty), dtype=tf.float32)
    
    if phi_type=='mme':
        train_disp = False
    if lamL_type=='mme':
        train_lamL = False

    CL = K.exp(-dist_square/2/lsq) + epsilon*K.eye(np.shape(dist_square)[0])
    CLinv = inv(CL)

    train_losses, valid_losses = [], []
    if patience is None:
        valid_losses = np.zeros(max_epochs)
    min_valid_loss = np.infty
    train_mse, valid_mse, phi_history, lamL_history, lsq_history, compute_time = [np.zeros(max_epochs) for _ in range(6)]

    K.clear_session()
    seed_everything(seed)    

    for epoch in range(pretrain):
        train_loss = train_one_epoch_RBF(train_batch, optimizer, mean_model, X_train, ZL_train, y_train, ZLTZL, N_train, dist_square, phi, lamL, CL, CLinv, train_disp=False, train_lamL=False)
    out_train = mean_model([X_train, ZL_train])  

    patience_count = 0
    temp_start = time.time()
    for epoch in range(max_epochs):

        if epoch < moments_epochs:
            train_loss = train_one_epoch_RBF(train_batch, optimizer, mean_model, X_train, ZL_train, y_train, ZLTZL, N_train, dist_square, phi, lamL, CL, CLinv, train_disp=False, train_lamL=False)
            phi.assign(np.var(y_train-np.sum(out_train, axis=0).T))            
            lamL.assign(np.var(mean_model.get_weights()[-1]))
            # lamL, lsq, CL, CLinv = update_spatial_disp2(lr_spatial, vL, ZLTZL, N_train, dist_square, phi, lam0, lamL, lsq, CL, CLinv, epsilon, train_lamL, train_lsq=True)
        else:
            train_loss = train_one_epoch_RBF(train_batch, optimizer, mean_model, X_train, ZL_train, y_train, ZLTZL, N_train, dist_square, phi, lamL, CL, CLinv, train_disp, train_lamL=False)
            wts = mean_model.get_weights()
            # add adjust process
            vL = wts[-1]
            lamL, lsq, CL, CLinv = update_spatial_disp(lr_spatial, vL, ZLTZL, N_train, dist_square, phi, lamL, lsq, CL, CLinv, epsilon, train_lamL, train_lsq=True)
            if np.isnan(lsq):
                break

        train_losses.append(train_loss)
        if patience is not None:
            valid_loss = nn_hlik_loss_RBF(mean_model, X_valid, ZL_valid, y_valid, ZLTZLv, N_valid, dist_square, phi, lamL, CL, CLinv, disp=True)
            valid_losses.append(valid_loss)
            if valid_loss > min_valid_loss and epoch > moments_epochs:
                patience_count += 1
                if patience_count == patience: break
            else: min_valid_loss = valid_loss 
            
        out_train = mean_model([X_train, ZL_train])                    
        out_valid = mean_model([X_valid, ZL_valid])
        mu_train = np.sum(out_train, axis=0).T
        mu_valid = np.sum(out_valid, axis=0).T
        
        train_mse[epoch] = np.mean((y_train-mu_train)**2)
        valid_mse[epoch] = np.mean((y_valid-mu_valid)**2)

        phi_history[epoch]  = phi
        lamL_history[epoch] = lamL
        lsq_history[epoch]  = lsq
        compute_time[epoch] = (time.time() - temp_start)

        if verbose: print(epoch, np.round([valid_mse[epoch], np.float32(phi), np.float32(lamL), np.float32(lsq)],3))
        
    res = {
        "wts" : mean_model.get_weights(),
        "phi" : np.float32(phi),
        "lamL" : np.float32(lamL),
        "lsq" : np.float32(lsq),
        "train_loss": train_losses[:(epoch+1)],
        "valid_loss": valid_losses[:(epoch+1)],
        "train_mse" : train_mse[:(epoch+1)],
        "valid_mse" : valid_mse[:(epoch+1)],
        "phi_history"  : phi_history[:(epoch+1)],
        "lamL_history" : lamL_history[:(epoch+1)],
        "lsq_history"  : lsq_history[:(epoch+1)],
        "compute_time" : compute_time[:(epoch+1)]
    }
    
    return res
    

def nn_hlik_loss_RBF2(mean_model, X, Z0, ZL, y, Z0TZ0, Z0TZL, ZLTZL, N, dist_square, phi, lam0, lamL, CL, CLinv, disp=True):
    
    # f(y|v)
    mu = K.transpose(K.sum(mean_model([X, Z0, ZL]), axis=0))
    hlik = - K.sum(K.square(y - mu)) / phi / np.shape(y)[0]
    
    # f(v)
    v0 = mean_model.weights[-2]
    vL = mean_model.weights[-1]
    diag_Z0TZ0_batch = np.diag(K.transpose(Z0)@Z0).copy()
    diag_Z0TZ0 = np.diag(Z0TZ0).copy()
    # To prevent divide by zero (when saturated 0/0 becomes 1 but it is okay becuase it leads to vi=0 for them)
    diag_Z0TZ0[diag_Z0TZ0==0], diag_Z0TZ0_batch[diag_Z0TZ0==0] = 1., 1.
    
    hlik += - (K.transpose(v0)@((diag_Z0TZ0_batch/diag_Z0TZ0)*v0))[0,0] / lam0 / N
    hlik += - (K.transpose(vL) @ CLinv @ vL)[0,0] / lamL / N
    
    # c(phi, lam, lsq; y)
    if disp:
        A = Z0TZ0*(lam0/phi) + K.eye(np.shape(Z0TZ0)[0])
        B = ZLTZL@CL*(lamL/phi) + K.eye(np.shape(ZLTZL)[0]) - Z0TZL.T@solve(A, Z0TZL)@CL*(lam0*lamL/phi**2)
        hlik += - (slogdet(A)[1] + slogdet(B)[1]) / N
        hlik += - K.log(phi)

    loss = - hlik
    return loss

def update_params_RBF2(optimizer, mean_model, X, Z0, ZL, y, Z0TZ0, Z0TZL, ZLTZL, N, dist_square, phi, lam0, lamL, CL, CLinv, train_disp, train_lamL):
    with tf.GradientTape() as tape:
        loss = nn_hlik_loss_RBF2(mean_model, X, Z0, ZL, y, Z0TZ0, Z0TZL, ZLTZL, N, dist_square, phi, lam0, lamL, CL, CLinv, train_disp)
    trainable_wts = mean_model.trainable_weights
    if train_disp: trainable_wts += [phi, lam0]
    if train_lamL: trainable_wts += [lamL]
    gradients = tape.gradient(loss, trainable_wts)
    optimizer.apply_gradients(zip(gradients, trainable_wts))

    return loss

def train_one_epoch_RBF2(train_batch, optimizer, mean_model, X, Z0, ZL, y, Z0TZ0, Z0TZL, ZLTZL, N, dist_square, phi, lam0, lamL, CL, CLinv, train_disp=True, train_lamL=True):
    losses = [] 
    for step, (X_batch, Z0_batch, ZL_batch, y_batch) in enumerate(train_batch):        
        loss = update_params_RBF2(optimizer, mean_model, X_batch, Z0_batch, ZL_batch, y_batch, Z0TZ0, Z0TZL, ZLTZL, N, dist_square, phi, lam0, lamL, CL, CLinv, train_disp, train_lamL)
        losses.append(loss)
    return losses    

def update_spatial_disp2(lr_spatial, vL, Z0TZ0, Z0TZL, ZLTZL, N, dist_square, phi, lam0, lamL, lsq, CL, CLinv, epsilon, train_lamL=True, train_lsq=True):
    dCLdg = -dist_square*CL
    A = ZLTZL/phi - Z0TZL.T@np.linalg.solve((Z0TZ0*(lam0/phi) + np.eye(np.shape(Z0TZ0)[0], dtype=np.float32)), Z0TZL)*(lam0*lamL/phi**2)
    B = A@CL + np.eye(np.shape(ZLTZL)[0], dtype=np.float32)
    if train_lsq:
        grad_lsq = ( (K.transpose(vL)@CLinv@dCLdg@CLinv@vL)[0,0]/lamL - - np.trace(np.linalg.solve(B, A@dCLdg)) ) / 2/ lsq / N
        log_lsq = np.log(lsq) - lr_spatial * grad_lsq
        lsq.assign(np.exp(log_lsq))
    if train_lamL:
        grad_lamL = ( (K.transpose(v)@CLinv@v)[0,0] + np.trace(np.linalg.solve(B, A@CL))/phi ) * lamL / N
        log_lamL = np.log(lamL) - lr_spatial * grad_lamL
        lamL.assign(np.exp(log_lamL))
    
    CL = K.exp(-dist_square/2/lsq) + epsilon*K.eye(np.shape(dist_square)[0])
    CLinv = np.linalg.inv(CL)
    return lamL, lsq, CL, CLinv

def train_model_RBF2(
    mean_model, train_batch, train_data, validation_data, dist_square,
    optimizer, lr_spatial, phi_init, lam0_init, lamL_init, lsq_init, phi_type='mme', lamL_type='mme', epsilon=1e-3,
    pretrain=0, moments_epochs=0, max_epochs=1000, patience = None,
    adjust=False, wts_init=None, verbose=False, seed=0):

    X_train, Z0_train, ZL_train, y_train = train_data
    X_valid, Z0_valid, ZL_valid, y_valid = validation_data
    N_train, N_valid = np.shape(y_train)[0], np.shape(y_valid)[0]
    
    Z0TZ0 = Z0_train.T@Z0_train
    Z0TZL = Z0_train.T@ZL_train
    ZLTZL = ZL_train.T@ZL_train

    if patience >= max_epochs: patience = None
    if patience is not None:
        Z0TZ0v = Z0_valid.T@Z0_valid
        Z0TZLv = Z0_valid.T@ZL_valid
        ZLTZLv = ZL_valid.T@ZL_valid

    if wts_init is not None:
        mean_model.set_weights(wts_init)

    phi  = tf.Variable(phi_init, name='phi', trainable=True, constraint=lambda x: tf.clip_by_value(x, 1e-18, np.infty), dtype=tf.float32)
    lam0 = tf.Variable(lam0_init, name='lam0', trainable=True, constraint=lambda x: tf.clip_by_value(x, 1e-18, np.infty), dtype=tf.float32)
    lamL = tf.Variable(lamL_init, name='lamL', trainable=True, constraint=lambda x: tf.clip_by_value(x, 1e-18, np.infty), dtype=tf.float32)
    lsq  = tf.Variable(lsq_init, name='lsq', trainable=True, constraint=lambda x: tf.clip_by_value(x, 1e-18, np.infty), dtype=tf.float32)
    
    if phi_type=='mme':
        train_disp = False
    if lamL_type=='mme':
        train_lamL = False

    CL = K.exp(-dist_square/2/lsq) + epsilon*K.eye(np.shape(dist_square)[0])
    CLinv = inv(CL)

    train_losses, valid_losses = [], []
    if patience is None:
        valid_losses = np.zeros(max_epochs)
    min_valid_loss = np.infty
    train_mse, valid_mse, phi_history, lam0_history, lamL_history, lsq_history, compute_time = [np.zeros(max_epochs) for _ in range(7)]

    K.clear_session()
    seed_everything(seed)    

    for epoch in range(pretrain):
        train_loss = train_one_epoch_RBF2(train_batch, optimizer, mean_model, X_train, Z0_train, ZL_train, y_train, Z0TZ0, Z0TZL, ZLTZL, N_train, dist_square, phi, lam0, lamL, CL, CLinv, train_disp=False, train_lamL=False)
    out_train = mean_model([X_train, Z0_train, ZL_train])  

    patience_count = 0
    temp_start = time.time()
    for epoch in range(max_epochs):

        if epoch < moments_epochs:
            train_loss = train_one_epoch_RBF2(train_batch, optimizer, mean_model, X_train, Z0_train, ZL_train, y_train, Z0TZ0, Z0TZL, ZLTZL, N_train, dist_square, phi, lam0, lamL, CL, CLinv, train_disp=False, train_lamL=False)
            phi.assign(np.var(y_train-np.sum(out_train, axis=0).T))
            lam0.assign(np.var(mean_model.get_weights()[-2]))
            lamL.assign(np.var(mean_model.get_weights()[-1]))
            # lamL, lsq, CL, CLinv = update_spatial_disp2(lr_spatial, vL, Z0TZ0, Z0TZL, ZLTZL, N_train, dist_square, phi, lam0, lamL, lsq, CL, CLinv, epsilon, train_lamL, train_lsq=True)
        else:
            train_loss = train_one_epoch_RBF2(train_batch, optimizer, mean_model, X_train, Z0_train, ZL_train, y_train, Z0TZ0, Z0TZL, ZLTZL, N_train, dist_square, phi, lam0, lamL, CL, CLinv, train_disp, train_lamL=False)
            wts = mean_model.get_weights()
            # add adjust process
            vL = wts[-1]
            if lamL_type=='mme':
                lamL.assign(np.var(wts[-1]))
            lamL, lsq, CL, CLinv = update_spatial_disp2(lr_spatial, vL, Z0TZ0, Z0TZL, ZLTZL, N_train, dist_square, phi, lam0, lamL, lsq, CL, CLinv, epsilon, train_lamL, train_lsq=True)
            if np.isnan(lsq):
                break  

        train_losses.append(train_loss)
        if patience is not None:
            valid_loss = nn_hlik_loss_RBF2(mean_model, X_valid, Z0_valid, ZL_valid, y_valid, Z0TZ0v, Z0TZLv, ZLTZLv, N_valid, dist_square, phi, lam0, lamL, CL, CLinv, disp=True)
            valid_losses.append(valid_loss)
            if valid_loss > min_valid_loss and epoch > moments_epochs:
                patience_count += 1
                if patience_count == patience: break
            else: min_valid_loss = valid_loss 
            
        out_train = mean_model([X_train, Z0_train, ZL_train])                    
        out_valid = mean_model([X_valid, Z0_valid, ZL_valid])
        mu_train = np.sum(out_train, axis=0).T
        mu_valid = np.sum(out_valid, axis=0).T
        
        train_mse[epoch] = np.mean((y_train-mu_train)**2)
        valid_mse[epoch] = np.mean((y_valid-mu_valid)**2)

        phi_history[epoch]  = phi
        lam0_history[epoch] = lam0
        lamL_history[epoch] = lamL
        lsq_history[epoch]  = lsq
        compute_time[epoch] = (time.time() - temp_start)

        if verbose: print(epoch, np.round([valid_mse[epoch], np.float32(phi), np.float32(lam0), np.float32(lamL), np.float32(lsq)],3))
        
    res = {
        "wts" : mean_model.get_weights(),
        "phi" : np.float32(phi),
        "lam0" : np.float32(lam0),
        "lamL" : np.float32(lamL),
        "lsq" : np.float32(lsq),
        "train_loss": train_losses[:(epoch+1)],
        "valid_loss": valid_losses[:(epoch+1)],
        "train_mse" : train_mse[:(epoch+1)],
        "valid_mse" : valid_mse[:(epoch+1)],
        "phi_history"  : phi_history[:(epoch+1)],
        "lam0_history" : lam0_history[:(epoch+1)],
        "lamL_history" : lamL_history[:(epoch+1)],
        "lsq_history"  : lsq_history[:(epoch+1)],
        "compute_time" : compute_time[:(epoch+1)]
    }
    
    return res
