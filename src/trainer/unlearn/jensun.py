import copy
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Any, Dict, List, Optional, Tuple, Union

from trainer.unlearn.base import UnlearnTrainer
from trainer.utils import TOK_EQ, compute_kl_divergence, jensun_retain_loss, jensun_multitok_loss


class GradJSDiff(UnlearnTrainer):
    def __init__(self, gamma:float=1.0, alpha:float=1.0, retain_loss_type:str="JensUn", *args, **kwargs):
        """Initializes the GradJSDiff trainer for unlearning with JensUn.

        This trainer implements an unlearning strategy that combines a "forget" loss
        (using a multi-token Jensen-Shannon-like divergence) with a "retain" loss
        (which can be Negative Log Likelihood, KL Divergence, or another Jensen-Shannon variant).
        
        Args:
            gamma (float, optional): Weighting factor for the `forget_loss`. Defaults to 1.0.
            alpha (float, optional): Weighting factor for the `retain_loss`. Defaults to 1.0.
            retain_loss_type (str, optional): Specifies the type of loss to be used
                for the retain set.
                Supported types are "NLL" (Negative Log Likelihood), "KL" (Kullback-Leibler
                Divergence against a reference model), or "JensUn" (a Jensen-Shannon
                divergence against a reference model). Defaults to "JensUn".
            *args: Variable length argument list to pass to the parent `UnlearnTrainer` class.
            **kwargs: Arbitrary keyword arguments to pass to the parent `UnlearnTrainer` class.
        """
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.alpha = alpha 
        self.retain_loss_type = retain_loss_type
        self.ref_model = None
        self.fix_toks()
        if retain_loss_type == "KL" or retain_loss_type=='JensUn':
            self.ref_model = self._prepare_ref_model(self.model)

    def fix_toks(self):
        """Determines the specific target token ID for the unlearning process.

        This method inspects the name of the currently loaded model (e.g., "Phi" models)
        and the name of the inheriting class (e.g., `JensUnEOT`, `JensUnHash`)
        to select a corresponding token ID from the global `TOK_EQ` mapping.
        This token ID is used in the `jensun_multitok_loss` for the forget operation.

        Raises:
            KeyError: If the current class name or model configuration path
                      is not found in the `TOK_EQ` mapping.
        """
        if "Phi" in self.model.config._name_or_path:
            #tokens are different for Phi/Llama tokenizers
            self.tok_id = TOK_EQ[self.__class__.__name__][1]
        else:
            self.tok_id = TOK_EQ[self.__class__.__name__][0]

    def _prepare_ref_model(self, model):
        ref_model = copy.deepcopy(model).to(self.accelerator.device)
        ref_model.eval()
        if self.is_deepspeed_enabled:
            ref_model = self._prepare_deepspeed(ref_model)
        else:
            ref_model = self.accelerator.prepare_model(ref_model, evaluation_mode=True)
        return ref_model

    def compute_retain_loss(self, model:AutoModelForCausalLM, retain_inputs:Dict[str, Dict[str, torch.Tensor]]):
        retain_outputs = model(**retain_inputs)
        retain_loss = 0.0
        if self.retain_loss_type == "NLL":
            retain_loss += retain_outputs.loss
        elif self.retain_loss_type == "KL":
            kl_loss, retain_outputs = compute_kl_divergence(
                self.model, self.ref_model, retain_inputs
            )
            retain_loss += kl_loss
        elif self.retain_loss_type == "JensUn":
            js_loss, retain_outputs = jensun_retain_loss(
                self.model, self.ref_model, retain_inputs
            )
            retain_loss += js_loss
        else:
            raise NotImplementedError(
                f"{self.retain_loss_type} not implemented for retain set"
            )
        return retain_loss

    def compute_loss(self, model:AutoModelForCausalLM, inputs:Dict[str, Dict[str, torch.Tensor]], return_outputs:bool=False):
        """Computes the total loss for the JensUn unlearning strategy.

        Args:
            model (`torch.nn.Module`): The language model being trained/unlearned.
            inputs (`Dict[str, Dict[str, torch.Tensor]]`): A dictionary containing
                input tensors, expected to have two main keys:
                - "forget": A dictionary with "input_ids", "attention_mask", and "labels"
                            for the data to be forgotten.
                - "retain": A dictionary with "input_ids", "attention_mask", and "labels"
                            for the data to be retained.
            return_outputs (bool, optional): Whether to return the model's outputs
                from the `forget_loss` calculation along with the total loss. Defaults to False.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
                The total computed `loss` (a scalar `torch.Tensor`). If `return_outputs` is True,
                it returns a tuple containing the `loss` and the `forget_outputs` from the
                `jensun_multitok_loss` calculation.
        """
        forget_inputs = inputs["forget"]

        forget_inputs = {
            "input_ids": forget_inputs["input_ids"],
            "attention_mask": forget_inputs["attention_mask"],
            "labels": forget_inputs["labels"],
        }
        phimod=False
        if "Phi" in str(model):
            phimod=True
        forget_loss, forget_outputs = jensun_multitok_loss(model, forget_inputs, self.tok_id)

        retain_inputs = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        loss = self.gamma * forget_loss + self.alpha * retain_loss

        if self.accelerator.is_local_main_process:
            self.log({
                "retain_loss": retain_loss.item(),
                "forget_loss": forget_loss.item(),
            })

        return (loss, forget_outputs) if return_outputs else loss

class JensUn(GradJSDiff):
    """A specialized GradJSDiff trainer using "JensUn" for the retain loss.
       The default JensUn target tokens: 'No Idea'
    """
    def __init__(self, gamma=1.0, alpha=1.0, retain_loss_type="JensUn", *args, **kwargs):
        super().__init__(gamma=gamma, alpha=alpha, retain_loss_type='JensUn', *args, **kwargs)
class JensUnEOT(GradJSDiff):
    """A specialized GradJSDiff trainer focusing on unlearning based on End-Of-Text (EOT) tokens.
       The JensUn target tokens: 'No Idea <EOT>'
    """
    def __init__(self, gamma=1.0, alpha=1.0, retain_loss_type="JensUn", *args, **kwargs):
        super().__init__(gamma=gamma, alpha=alpha, retain_loss_type='JensUn', *args, **kwargs)
class JensUnHash(GradJSDiff):
    """A specialized GradJSDiff trainer focusing on unlearning for random tokens.
       The JensUn target tokens: '#'
    """
    def __init__(self, gamma=1.0, alpha=1.0, retain_loss_type="JensUn", *args, **kwargs):
        super().__init__(gamma=gamma, alpha=alpha, retain_loss_type='JensUn', *args, **kwargs)
class JensUnComma(GradJSDiff):
    """A specialized GradJSDiff trainer focusing on unlearning for random tokens.
       The JensUn target tokens: ','
    """
    def __init__(self, gamma=1.0, alpha=1.0, retain_loss_type="JensUn",  *args, **kwargs):
        super().__init__(gamma=gamma, alpha=alpha, retain_loss_type='JensUn', *args, **kwargs)
class JensUnWhiteSpace(GradJSDiff):
    """A specialized GradJSDiff trainer focusing on unlearning for random tokens.
       The JensUn target tokens: ' '
    """
    def __init__(self, gamma=1.0, alpha=1.0, retain_loss_type="JensUn", *args, **kwargs):
        super().__init__(gamma=gamma, alpha=alpha, retain_loss_type='JensUn', *args, **kwargs)