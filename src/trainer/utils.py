import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch, math
import torch.nn.functional as F
from torch import nn

#mapping q_forget tokens for llama, phi-mini
TOK_EQ = {
    'JensUn': [[2822, 4623], [1939, 2969]], #no way: 2822 16489
    'KLDiv': [[2822, 4623], [1939, 2969]],
    'JensUnEOT': [[2, 128009], [1939, 2969, 32000]],
    'JensUnHash': [[2], [396]],
    'JensUnComma': [[11], [1919]],
    'JensUnWhiteSpace': [[220], [259]],
    'JensUnLongString': [[40, 7846, 956, 1505, 430], []] # I couldn't find that 
}


def seed_everything(seed:int=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_kl_divergence(model, target_model, inputs):
    with torch.no_grad():
        ref_outputs = target_model(**inputs)
        ref_logits = ref_outputs.logits

    outputs = model(**inputs)
    logits = outputs.logits

    # 1. Align Logits and Labels (Causal Shift)
    # Logit at index t predicts label at index t+1
    shift_logits = logits[..., :-1, :].contiguous()
    shift_ref_logits = ref_logits[..., :-1, :].contiguous()
    shift_labels = inputs["labels"][..., 1:].contiguous()
    
    # 2. Create Mask
    mask = (shift_labels != -100)

    # 3. Compute log-probabilities for stability
    log_p = F.log_softmax(shift_ref_logits, dim=-1)
    log_q = F.log_softmax(shift_logits, dim=-1)
    
    # p and q as probabilities
    p = log_p.exp()
    q = log_q.exp()
    
    # JS Divergence = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
    # KL(P || M) = sum(P * (logP - logM))
    kl_p_q = (p * (log_p - log_q)).sum(dim=-1)


    # 4. Apply Mask and Average
    if mask.sum() > 0:
        loss = kl_p_q[mask].mean()
    else:
        loss = kl_p_q.sum() * 0.0 # Avoid None loss if no labels

    return loss, outputs

def compute_js_avg(model, inputs):
    # get the sum loss for each sequence in a batch
    # NOTE: not same as model(**inputs).loss but has sum loss for each seq in a batch

    # Compute probabilities from logits
    outputs = model(**inputs)
    logits = outputs.logits
    probs = F.softmax(logits, dim=-1)
        
    # Define the uniform distribution over the vocabulary
    uniform_dist = torch.full_like(probs, 1.0 / probs.size(-1))

    # Compute the midpoint distribution
    m = 0.5 * (probs + uniform_dist)

    # Compute KL divergence (adding a small value to avoid log(0))
    kl_p_m = F.kl_div(m.log(), probs, reduction='batchmean')
    kl_q_m = F.kl_div(m.log(), uniform_dist, reduction='batchmean')

    # Compute final JS Divergence
    js_div = 0.5 * (kl_p_m + kl_q_m)

    return js_div, outputs


def jensun_retain_loss(model, target_model, inputs):
    with torch.no_grad():
        ref_outputs = target_model(**inputs)
        ref_logits = ref_outputs.logits

    outputs = model(**inputs)
    logits = outputs.logits

    # 1. Align Logits and Labels (Causal Shift)
    # Logit at index t predicts label at index t+1
    shift_logits = logits[..., :-1, :].contiguous()
    shift_ref_logits = ref_logits[..., :-1, :].contiguous()
    shift_labels = inputs["labels"][..., 1:].contiguous()
    
    # 2. Create Mask
    mask = (shift_labels != -100)

    # 3. Compute log-probabilities for stability
    log_p = F.log_softmax(shift_ref_logits, dim=-1)
    log_q = F.log_softmax(shift_logits, dim=-1)
    
    # p and q as probabilities
    p = log_p.exp()
    q = log_q.exp()
    
    # Midpoint distribution m = 0.5 * (p + q)
    # We stay in probability space for m, but compute log(m) for KL
    m = 0.5 * (p + q)
    log_m = m.clamp(min=1e-10).log()

    # JS Divergence = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
    # KL(P || M) = sum(P * (logP - logM))
    kl_p_m = (p * (log_p - log_m)).sum(dim=-1)
    kl_q_m = (q * (log_q - log_m)).sum(dim=-1)
    
    js_div = 0.5 * (kl_p_m + kl_q_m)

    # 4. Apply Mask and Average
    if mask.sum() > 0:
        loss = js_div[mask].mean()
    else:
        loss = js_div.sum() * 0.0 # Avoid None loss if no labels

    return loss, outputs

def compute_uniform_ce_avg(model, inputs) -> (float, torch.Tensor):
    # Forward pass to get logits
    outputs = model(**inputs)
    logits = outputs.logits

    # Compute log probabilities from logits
    log_probs = F.log_softmax(logits, dim=-1)

    # Define the uniform target distribution over the vocabulary
    vocab_size = logits.size(-1)
    uniform_dist = torch.full_like(log_probs, 1.0 / vocab_size)

    # Compute cross-entropy loss: CE(target, log_probs)
    ce_loss = -(uniform_dist * log_probs).sum(dim=-1)  # sum over vocab
    ce_loss = ce_loss.mean()  # average over batch and sequence if needed

    return ce_loss, outputs



def compute_batch_nll(model, inputs) -> (float, torch.Tensor):
    # get the sum loss for each sequence in a batch
    # NOTE: not same as model(**inputs).loss but has sum loss for each seq in a batch
    outputs = model(**inputs)
    logits = outputs.logits
    labels = inputs["labels"]
    shifted_labels = labels[..., 1:].contiguous()
    logits = logits[..., :-1, :].contiguous()
    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    loss = loss_function(logits.transpose(-1, -2), shifted_labels).sum(dim=-1)
    return loss, outputs


def compute_dpo_loss(model, ref_model, win_inputs=None, lose_inputs=None, beta=1.0):
    if win_inputs is None and lose_inputs is None:
        raise ValueError("Both win_inputs and lose_inputs can't be None")

    win_log_ratio, lose_log_ratio = 0.0, 0.0
    win_outputs, lose_outputs = None, None

    if win_inputs is not None:
        win_loss, win_outputs = compute_batch_nll(model, win_inputs)
        with torch.no_grad():
            win_ref_loss, _ = compute_batch_nll(ref_model, win_inputs)
        win_log_ratio = -(win_loss - win_ref_loss)

    if lose_inputs is not None:
        lose_loss, lose_outputs = compute_batch_nll(model, lose_inputs)
        with torch.no_grad():
            lose_ref_loss, _ = compute_batch_nll(ref_model, lose_inputs)
        lose_log_ratio = -(lose_loss - lose_ref_loss)

    loss = -2 / beta * F.logsigmoid(beta * (win_log_ratio - lose_log_ratio)).mean()
    return loss, (win_outputs, lose_outputs)

def kldiv_forget_loss(model, forget_inputs, tokids=[2822, 4623]) -> (float, torch.Tensor):

    outputs = model(forget_inputs["input_ids"], attention_mask=forget_inputs["attention_mask"])
    logits = outputs.logits

    labels = forget_inputs["labels"]

    epsilon = 1e-9

    device = logits.device
    batch_size, seq_len, vocab_size = logits.size()

    # SHIFT for Causal Prediction
    # Input @ t predicts Target @ t+1
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Create the binary mask (Where is the response?)
    mask = (shift_labels != -100).float()
    
    # Ensure pattern is a tensor
    if not isinstance(tokids, torch.Tensor):
        pattern = torch.tensor(tokids, device=device, dtype=torch.long)
    else:
        pattern = tokids.to(device)
    
    pattern_len = len(tokids)

    peaked_dist = torch.zeros_like(shift_logits)

    # Iterate per batch because every sample has a different response length?
    for i in range(batch_size):
        # valid_indices is a 1D tensor of positions, e.g., [10, 11, 12, 13]
        valid_indices = torch.nonzero(mask[i], as_tuple=True)[0]
        num_slots = len(valid_indices)
        
        if num_slots == 0:
            continue
            
        repeats = math.ceil(num_slots / pattern_len)
        # .repeat generates [A, B, A, B, A, B]
        cyclic_targets = pattern.repeat(repeats)[:num_slots]
        #Create One-Hot for this specific sequence
        seq_target_probs = torch.zeros(num_slots, vocab_size, device=device)
        # Scatter 1.0 into the correct token columns
        seq_target_probs.scatter_(1, cyclic_targets.unsqueeze(1), 1.0)
                
        peaked_dist[i, valid_indices, :] = seq_target_probs
    
    # 3. COMPUTE JSD (Standard)
    model_probs = F.softmax(shift_logits, dim=-1)
    peaked_dist = torch.clamp(peaked_dist, min=epsilon)
    model_probs = torch.clamp(model_probs, min=epsilon)

    
    kl_p_q = torch.sum(peaked_dist * (torch.log(peaked_dist) - torch.log(model_probs)), dim=-1)
    
    # APPLY MASK
    masked_loss = kl_p_q * mask

    # Normalize by valid tokens only
    loss = masked_loss.sum() / (mask.sum() + 1e-8)

    return loss, outputs



def jensun_multitok_loss(model, forget_inputs, tokids=[2822, 4623]) -> (float, torch.Tensor):
    
    outputs = model(forget_inputs["input_ids"], attention_mask=forget_inputs["attention_mask"])
    logits = outputs.logits

    labels = forget_inputs["labels"]

    epsilon = 1e-9

    device = logits.device
    batch_size, seq_len, vocab_size = logits.size()

    # SHIFT for Causal Prediction
    # Input @ t predicts Target @ t+1
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Create the binary mask (Where is the response?)
    mask = (shift_labels != -100).float()
    
    # Ensure pattern is a tensor
    if not isinstance(tokids, torch.Tensor):
        pattern = torch.tensor(tokids, device=device, dtype=torch.long)
    else:
        pattern = tokids.to(device)
    
    pattern_len = len(tokids)

    peaked_dist = torch.zeros_like(shift_logits)

    # Iterate per batch because every sample has a different response length?
    for i in range(batch_size):
        valid_indices = torch.nonzero(mask[i], as_tuple=True)[0]
        num_slots = len(valid_indices)
        
        if num_slots == 0:
            continue
            
        repeats = math.ceil(num_slots / pattern_len)
        # .repeat generates [A, B, A, B, A, B]
        cyclic_targets = pattern.repeat(repeats)[:num_slots]
        #Create One-Hot for this specific sequence
        seq_target_probs = torch.zeros(num_slots, vocab_size, device=device)
        # Scatter 1.0 into the correct token columns
        seq_target_probs.scatter_(1, cyclic_targets.unsqueeze(1), 1.0)
                
        peaked_dist[i, valid_indices, :] = seq_target_probs
    
    # 3. COMPUTE JSD (Standard)
    model_probs = F.softmax(shift_logits, dim=-1)
    peaked_dist = torch.clamp(peaked_dist, min=epsilon)
    model_probs = torch.clamp(model_probs, min=epsilon)

    # Average distribution M
    m = 0.5 * (model_probs + peaked_dist)

    
    kl_p_m = torch.sum(model_probs * (torch.log(model_probs) - torch.log(m)), dim=-1)
    kl_q_m = torch.sum(peaked_dist * (torch.log(peaked_dist) - torch.log(m)), dim=-1)
    jsd = 0.5 * (kl_p_m + kl_q_m)
    
    # APPLY MASK
    masked_loss = jsd * mask

    # Normalize by valid tokens only
    loss = masked_loss.sum() / (mask.sum() + 1e-8)

    return loss, outputs



def js_multitok_loss_with_eos(model, forget_inputs, tokids=[2822, 4623]) -> (float, torch.Tensor):
    outputs = model(**forget_inputs)
    logits = outputs.logits  # Shape: [batch, seq_len, vocab_size]
    
    # 1. Determine the length to align (min of model output length and target length)
    num_targets = len(tokids)
    seq_len = logits.size(1)
    effective_len = min(seq_len, num_targets)
    
    # 2. Slice the logits to only focus on the first N tokens
    # Shape becomes: [batch, effective_len, vocab_size]
    sliced_logits = logits[:, :effective_len, :]
    
    # 3. Calculate probabilities for the sliced segment
    probs = F.softmax(sliced_logits, dim=-1)
    
    # 4. Construct the Target Distribution (Peaked)
    # We create a distribution that is 1.0 at the target token and 0.0 elsewhere
    peaked_dist = torch.zeros_like(probs)
    
    # Create a tensor for the target indices
    target_tensor = torch.tensor(tokids[:effective_len], device=logits.device)
    
    # We need to broadcast these targets across the batch dimension
    # Shape adjustment: [1, effective_len, 1] -> [batch, effective_len, 1]
    batch_size = logits.size(0)
    target_idxs = target_tensor.view(1, -1, 1).expand(batch_size, -1, -1)
    
    # Use scatter_ to place the 1.0s at the correct token indices
    peaked_dist.scatter_(2, target_idxs, 1.0)

    # 5. Jensen-Shannon Divergence Calculation
    epsilon = 1e-9
    
    # Midpoint distribution
    m = 0.5 * (probs + peaked_dist)
    
    # Clamp to avoid log(0) errors
    probs = torch.clamp(probs, min=epsilon)
    peaked_dist = torch.clamp(peaked_dist, min=epsilon)
    m = torch.clamp(m, min=epsilon)

    # Compute KL divergences manually (P||M and Q||M)
    kl_p_m = torch.sum(probs * (torch.log(probs) - torch.log(m)), dim=-1)
    kl_q_m = torch.sum(peaked_dist * (torch.log(peaked_dist) - torch.log(m)), dim=-1)
    
    # Average the divergences
    js_div = 0.5 * (kl_p_m.mean() + kl_q_m.mean())

    return js_div, outputs


