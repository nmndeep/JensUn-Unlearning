import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

#mapping q_forget tokens for llama, phi-mini
TOK_EQ = {
    'JensUn': [[2822, 4623], [1939, 2969]],
    'JensUnEOT': [[2822, 4623, 128009], [1939, 2969, 32000]],
    'JensUnHash': [[11], [396]],
    'JensUnComma': [[2], [1919]],
    'JensUnWhiteSpace': [[220], [259]],
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

    ref_probs = F.log_softmax(ref_outputs.logits, dim=-1)
    ref_probs = F.log_softmax(ref_outputs.logits, dim=-1)
    ref_probs = ref_probs.view(-1, ref_outputs.logits.shape[-1])

    outputs = model(**inputs)
    current_probs = F.log_softmax(outputs.logits, dim=-1)
    current_probs = current_probs.view(-1, outputs.logits.shape[-1])

    # minimum KL divergence
    return nn.functional.kl_div(
        current_probs, ref_probs, reduction="batchmean", log_target=True
    ), outputs


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
    epsilon = 1e-9
    ref_probs = F.softmax(ref_logits, dim=-1)


    outputs = model(**inputs)
    logits = outputs.logits
    current_probs = F.softmax(logits, dim=-1)

    # Compute the midpoint distribution
    m = 0.5 * (ref_probs + current_probs)
    m = torch.clamp(m, min=epsilon)  # Added this line

    # Clamp input distributions as well to avoid log(0)
    ref_probs = torch.clamp(ref_probs, min=epsilon)
    current_probs = torch.clamp(current_probs, min=epsilon)

    # Compute KL divergences manually (P‖M and Q‖M)
    kl_p_m = torch.sum(ref_probs * (torch.log(ref_probs) - torch.log(m)), dim=-1).mean()
    kl_q_m = torch.sum(current_probs * (torch.log(current_probs) - torch.log(m)), dim=-1).mean()

    # Compute final JS Divergence
    js_div = 0.5 * (kl_p_m + kl_q_m)

    return js_div, outputs

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



def jensun_multitok_loss(model, forget_inputs, tokids=[2822, 4623]) -> (float, torch.Tensor):

    outputs = model(**forget_inputs)
    logits = outputs.logits
    epsilon = 1e-9

    probs = F.softmax(logits, dim=-1)
    vocab_size = logits.size(-1)
    peaked_dist = torch.zeros_like(probs)

    targ_tokens = tokids

    seq_len = logits.shape[1]
    expected_tokens = torch.tensor(targ_tokens).repeat(seq_len // len(targ_tokens) + 1)[:seq_len]
    # probs[non_zero_indices] = equal_prob
    peaked_dist[..., expected_tokens] = 1.0  # Set probability 1.0 at resp tokens


    # Midpoint distribution
    m = 0.5 * (probs + peaked_dist)
    m = torch.clamp(m, min=epsilon)  # Ensure no zeros

    # Clamp input distributions as well to avoid log(0)
    probs = torch.clamp(probs, min=epsilon)
    peaked_dist = torch.clamp(peaked_dist, min=epsilon)

    # Compute KL divergences manually (P‖M and Q‖M)
    kl_p_m = torch.sum(probs * (torch.log(probs) - torch.log(m)), dim=-1)
    kl_q_m = torch.sum(peaked_dist * (torch.log(peaked_dist) - torch.log(m)), dim=-1)
    
    
    js_div = 0.5 * (kl_p_m.mean() + kl_q_m.mean())

    return js_div, outputs

