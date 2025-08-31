import copy
from trainer.utils import compute_kl_divergence
from trainer.unlearn.base import UnlearnTrainer
import torch

class GradLearn(UnlearnTrainer):
    def __init__(self, gamma=1.0, alpha=1.0, retain_loss_type="NLL", *args, **kwargs):
        """Initializes the GradLearn trainer.

        This trainer is designed for ReLearning strategy that focuses
        on learning from a "retain" set without an explicit "forget" loss
        component within the `compute_loss` method (as indicated by `forget_loss: 0.0`).

        Args:
            gamma (float, optional): A weighting factor for potential future use or
                compatibility with other unlearning strategies. Defaults to 1.0.
            alpha (float, optional): Another weighting factor for potential future use.
                Defaults to 1.0.
            retain_loss_type (str, optional): Specifies the type of loss to be used
                for the retain set. Currently, only "NLL" (Negative Log Likelihood,
                which is typically what `model.forward().loss` computes for classification/LM)
                is implicitly supported by the `compute_retain_loss` method. Defaults to "NLL".
            *args: Variable length argument list to pass to the parent `UnlearnTrainer` class.
            **kwargs: Arbitrary keyword arguments to pass to the parent `UnlearnTrainer` class.
        """
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.retain_loss_type = retain_loss_type
        self.ref_model = None

    def compute_retain_loss(self, model, retain_inputs):
        retain_outputs = model(**retain_inputs)
        retain_loss = 0.0
        retain_loss += retain_outputs.loss
        
        return retain_loss

    def compute_loss(self, model, inputs, return_outputs=False):
        """Computes the training loss for the GradLearn unlearning strategy.

        This method specifically focuses on minimizing the loss on the "retain" dataset,
        effectively performing a "learn-only" operation on the new disjoint set
        intended for relearning

        Args:
            model (`torch.nn.Module`): The language model being trained/unlearned.
            inputs (`Dict[str, torch.Tensor]`): A dictionary containing input tensors
                for the model.
            return_outputs (bool, optional): Whether to return the model's raw outputs
                along with the loss. 

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
                
        """
        retain_inputs = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        loss = retain_loss

        if self.accelerator.is_local_main_process:
            self.log({
                "retain_loss": retain_loss.item(),
                "forget_loss": 0.0,
            })

        return (loss, forget_outputs) if return_outputs else loss
