import gc
import time
import torch
import numpy as np
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from transformers import AdamW, get_linear_schedule_with_warmup

from util import *
from training.freeze import *
from training.sampler import *


def custom_params(model, weight_decay=0, lr=1e-3, lr_transfo=3e-5, lr_decay=1):
    """
    Custom parameters for Bert Models to handle weight decay and differentiated learning rates.
    
    Arguments:
        model {torch model} -- Model to get parameters on
    
    Keyword Arguments:
        weight_decay {float} -- Weight decay (default: {0})
        lr {float} -- Learning rate of layers not belongign to the transformer, i.e. not pretrained (default: {1e-3})
        lr_transfo {float} -- Learning rate of the layer the of the transformer the closer to the output (default: {3e-5})
        lr_decay {float} -- How much to multiply the lr_transfo by when going deeper in the model (default: {1})
    
    Returns:
        torch opt_params -- Parameters to feed the optimizer for the model
    """
    opt_params = []
    no_decay = ["bias", "LayerNorm.weight"]
    nb_blocks = len(model.transformer.encoder.layer)
    
    for n, p in model.named_parameters():
        wd = 0 if any(nd in n for nd in no_decay) else weight_decay
        
        if "transformer" in n and "pooler" not in n:
            lr_ = lr_transfo
            if "transformer.embeddings" in n:
                lr_ = lr_transfo * lr_decay ** (nb_blocks)
            else:
                for i in range(nb_blocks):  # for bert base
                    if f"layer.{i}." in n:
                        lr_ = lr_transfo * lr_decay ** (nb_blocks - 1 - i)
                        break
        else:
            lr_ = lr

        opt_params.append({
         "params": [p], 
         "weight_decay": wd,
         'lr':lr_,
        })
        # print(n, lr_, wd)
    return opt_params


def predict(model, dataset, batch_size=32):
    """
    Usual predict torch function
    
    Arguments:
        model {torch model} -- Model to predict with
        dataset {torch dataset} -- Dataset to get predictions from
    
    Keyword Arguments:
        batch_size {int} -- Batch size (default: {32})
    
    Returns:
        numpy array -- Predictions
    """

    model.eval()
    preds = np.empty((0, NUM_CLASSES))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)

    with torch.no_grad():
        for x, _ in loader:
            prob = torch.softmax(model(x.cuda())[0], 1)
            probs = prob.detach().cpu().numpy()
            preds = np.concatenate([preds, probs])
            
    return preds


def predict_tta(model, dataset, batch_size=32, nb_tta=2):
    """
    Usual predict torch function, but looping over the batches to perform TTA
    
    Arguments:
        model {torch model} -- Model to predict with
        dataset {torch dataset} -- Dataset to get predictions from
    
    Keyword Arguments:
        batch_size {int} -- Batch size (default: {32})
        nb_tta {int} -- Number of TTA to do (default: {2})
    
    Returns:
        numpy array -- Predictions
    """
    model.eval()
    all_preds = []
    
    for tta in range(nb_tta):
        preds = np.empty((0, NUM_CLASSES))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)

        with torch.no_grad():
            for x, _ in loader:
                prob = torch.softmax(model(x.cuda())[0], 1)
                probs = prob.detach().cpu().numpy()
                preds = np.concatenate([preds, probs])
        all_preds.append(preds)
    return np.mean(np.array(all_preds), 0)


def fit(model, train_dataset, val_dataset, epochs=5, batch_size=8, acc_steps=1, weight_decay=0,
        warmup_prop=0, lr_transfo=1e-3, lr=5e-4, lr_decay=1, cp=False, model_name='model'):

    best_loss = 1000
    
    len_sampler = LenMatchBatchSampler(RandomSampler(train_dataset), batch_size=batch_size, drop_last=True)
    train_loader = DataLoader(train_dataset, batch_sampler=len_sampler, num_workers=NUM_WORKERS) 
    
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)

    opt_params = custom_params(model, lr=lr, weight_decay=weight_decay, lr_transfo=lr_transfo, lr_decay=lr_decay)
    optimizer = AdamW(opt_params, lr=lr, betas=(0.5, 0.999))
    
    num_warmup_steps = int(warmup_prop * epochs * len(train_loader) / acc_steps)
    num_training_steps = int(epochs * len(train_loader) / acc_steps)
    
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    loss_fct = nn.CrossEntropyLoss(reduction='mean').cuda()
    
    for epoch in range(epochs):
        model.train()
        start_time = time.time()
        
        optimizer.zero_grad()
        avg_loss = 0
        
        for step, (tokens, y_batch) in enumerate(train_loader): 
            tokens = trim_tensors(tokens)
            y_pred, _ = model(tokens.cuda())
            
            loss = loss_fct(y_pred, y_batch.long().cuda())
            loss.backward()
            avg_loss += loss.item() / len(train_loader)
            
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            if (step + 1) % acc_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                optimizer.zero_grad()
                
        model.eval()
        avg_val_loss = 0.

        with torch.no_grad():
            for tokens, y_batch in val_loader:                
                y_pred, _ = model(tokens.cuda())
                loss = loss_fct(y_pred.detach(), y_batch.long().cuda())
                avg_val_loss += loss.item() / len(val_loader)

        if avg_val_loss >= best_loss and cp:
            save_model_weights(model, f"{model_name}_cp.pt", verbose=0)
            best_loss = avg_val_loss
        
        dt = time.time() - start_time
        lr = scheduler.get_lr()[0]
        print(f'Epoch {epoch + 1}/{epochs} \t lr={lr:.1e} \t t={dt:.0f}s \t loss={avg_loss:.4f} \t val_loss={avg_val_loss:.4f}')
            
    del loss, tokens, y_batch, avg_val_loss, avg_loss, train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()