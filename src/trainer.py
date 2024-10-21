import inspect
import random
import warnings
from collections import defaultdict
from contextlib import nullcontext
from copy import deepcopy
from functools import wraps
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.utils import is_deepspeed_available, tqdm
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
import math
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput
from trl.import_utils import is_peft_available, is_wandb_available
from trl.models import PreTrainedModelWrapper, create_reference_model
from trl.trainer.utils import disable_dropout_in_model, pad_to_length, trl_sanitze_kwargs_for_tagging

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from torch.nn.utils.rnn import pad_sequence

if is_peft_available():
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training


if is_wandb_available():
    import wandb

if is_deepspeed_available():
    import deepspeed

from .losses import RANK_LOSS_TYPE
from .scores import _tdpo_get_batch_position_forward_kl,RANK_SCORE_TYPE,get_score_fn
from .rank_utils import calculate_pairwise_accuracy,compute_ndcg

@dataclass
class DRPODataCollatorWithPadding:
    r""" 
    DRPO DataCollator class that pads the tokenized inputs to the maximum length of the batch.
    Args:
        pad_token_id (`int` defaults to 0):
            The tokenizer's pad_token_id.
        label_pad_token_id (`int`, defaults to -100):
            The label used for masking.
        is_encoder_decoder (`Optional[bool]`, `optional`, defaults to `None`):
            Whether or not you model has an encoder_decoder architecture.
    """
    pad_token_id: int = 0
    label_pad_token_id: int = -100
    is_encoder_decoder: Optional[bool] = False

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # first, pad everything to the same length
        padded_batch = {}
        
        for k in features[0].keys():
            if k.endswith("_input_ids") or k.endswith("_attention_mask") or k.endswith("_labels"):
                
                if self.is_encoder_decoder:
                    to_pad = [torch.LongTensor(ex[k]) for ex in features]
                    if (k.startswith("prompt")) and (k.endswith("input_ids")):
                        padding_value = self.pad_token_id
                    elif k.endswith("_attention_mask"):
                        padding_value = 0
                    elif (k.startswith("chosen")) or (k.startswith("rejected")) or ("decoder" in k):
                        padding_value = self.label_pad_token_id
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")
                    padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                else:
                    # adapted from https://stackoverflow.com/questions/73256206
                    if "prompt" in k:
                        to_pad = [torch.LongTensor(ex[k][::-1]) for ex in features]
                    else:
                        to_pad = [torch.LongTensor(ex[k]) for ex in features]
                    if k.endswith("_input_ids"):
                        padding_value = self.pad_token_id
                    elif k.endswith("_labels"):
                        padding_value = self.label_pad_token_id
                    elif k.endswith("_attention_mask"):
                        padding_value = 0
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")

                    padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                    # for the prompt, flip back so padding is on left side
                    if "prompt" in k:
                        padded_batch[k] = padded_batch[k].flip(dims=[1])
            elif k.endswith("_logps"):
                # the cached reference model logprobs
                padded_batch[k] = torch.tensor([ex[k] for ex in features])

            elif k.startswith('rank_'):
                
                rank_list = [ feature[k] for feature in features]
                # [ [{},{}] , [{},{}] ]

                for rk in rank_list[0][0].keys():

                    padded_key = 'rank_'+rk
                    if rk.endswith('_input_ids') or rk.endswith('_attention_mask') or rk.endswith('_labels'):

                        to_pad = [ torch.LongTensor(ex[rk])  for ex_list in rank_list for ex in ex_list] 

                        if rk.endswith('_input_ids'):
                            padding_value = self.pad_token_id
                        elif rk.endswith('_labels'):
                            padding_value = -100
                        elif rk.endswith('_attention_mask'):
                            padding_value = 0
                        else:
                            raise ValueError(f"Unexpected key in batch '{rk}'")
                        
                        padded_batch[padded_key] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                        # print('pad value input,labels,mask -> ',self.tokenizer.pad_token_id,-100,0)
                    else:
                        padded_batch[padded_key] = [ rex[rk] for rex_list in rank_list for rex in rex_list]
            
            elif k.endswith('_scores'):
                scores = [ ex[k] for ex in features ]
                padded_batch[k] = torch.Tensor(scores)
            elif k.endswith('_index'):
                sft_indexes = [ ex[k] for ex in features ]
                padded_batch[k] = torch.Tensor(sft_indexes)
            else:
                padded_batch[k] = [ex[k] for ex in features]

        for k in padded_batch.keys():
            if k.startswith('rank_') and (k.endswith('_input_ids') or k.endswith('_attention_mask') or k.endswith('_labels')):
                BR,_ = padded_batch[k].shape
                B,_  = padded_batch['prompt_input_ids'].shape
                padded_batch[k] = padded_batch[k].reshape(B,BR // B,-1)        


        return padded_batch

class DRPOTrainer(Trainer):
    r"""
    Initialize DRPOTrainer.
    Args:
        model (`transformers.PreTrainedModel`, `nn.Module`, or `str`):
            The model to train. Can be a pretrained model, a PyTorch module, or a model identifier string.
        
        ref_model (`Optional[Union[PreTrainedModel, nn.Module, str]]`):
            The reference model used for comparison. Can be None, a pretrained model, a PyTorch module, or a model identifier string.
        
        beta (`float`, defaults to 0.1):
            The beta parameter used in the DRPO algorithm.
        
        use_ref (`bool`, defaults to True):
            Whether to use a reference model in the training process.
        
        mean_logps (`bool`, defaults to False):
            Whether to use mean log probabilities.
        
        loss_type (`Literal["diff", "diffndcg"]`, defaults to "diffndcg"):
            The type of loss function to use. Can be either "diff" or "diffndcg".
        
        score_type (`str`, defaults to 'dpo'):
            The type of scoring method to use.
        
        score_config (`Union[List, Dict]`, optional):
            Configuration for the scoring method.
        
        loss_config (`Dict`, optional):
            Configuration for the loss function.
        
        kl_coef (`Dict`, optional):
            Coefficient for KL divergence, if used.
        
        args (`TrainingArguments`):
            The arguments to use for training.
        
        data_collator (`Optional[DataCollator]`):
            The data collator to use for batching.
        
        label_pad_token_id (`int`, defaults to -100):
            The ID used for padding labels.
        
        padding_value (`int`, optional):
            The value to use for padding, if different from the tokenizer's pad token ID.
        
        truncation_mode (`str`, defaults to "keep_end"):
            The mode to use when truncating sequences.
        
        train_dataset (`Optional[Dataset]`):
            The dataset to use for training.
        
        eval_dataset (`Optional[Union[Dataset, Dict[str, Dataset]]]`):
            The dataset(s) to use for evaluation.
        
        tokenizer (`Optional[PreTrainedTokenizerBase]`):
            The tokenizer to use.
        
        model_init (`Optional[Callable[[], PreTrainedModel]]`):
            A function that instantiates the model.
        
        callbacks (`Optional[List[TrainerCallback]]`):
            A list of callbacks to use during training.
        
        optimizers (`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
            The optimizer and learning rate scheduler to use.
        
        preprocess_logits_for_metrics (`Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]`):
            A function to preprocess the logits before computing metrics.
        
        max_length (`Optional[int]`):
            The maximum length of the input sequences.
        
        max_prompt_length (`Optional[int]`):
            The maximum length of the prompt part of the input.
        
        max_target_length (`Optional[int]`):
            The maximum length of the target part of the input.
        
        peft_config (`Optional[Dict]`):
            Configuration for parameter-efficient fine-tuning.
        
        is_encoder_decoder (`Optional[bool]`):
            Whether the model is an encoder-decoder model.
        
        disable_dropout (`bool`, defaults to True):
            Whether to disable dropout in the model.
        
        generate_during_eval (`bool`, defaults to False):
            Whether to generate outputs during evaluation.
        
        compute_metrics (`Optional[Callable[[EvalLoopOutput], Dict]]`):
            A function to compute metrics during evaluation.
        
        precompute_ref_log_probs (`bool`, defaults to False):
            Whether to precompute log probabilities for the reference model.
        
        model_init_kwargs (`Optional[Dict]`):
            Additional keyword arguments for model initialization.
        
        ref_model_init_kwargs (`Optional[Dict]`):
            Additional keyword arguments for reference model initialization.
    """

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module, str] = None,
        ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        beta: float = 0.1,
        use_ref: bool = True,
        mean_logps: bool = False,
        loss_type: Literal["diff", "diffndcg"] = "diffndcg",
        score_type:str = 'dpo',
        score_config : Union[List,Dict] = None,
        loss_config : Dict = None ,
        kl_coef : Dict = None ,
        args: TrainingArguments = None,
        data_collator: Optional[DRPODataCollatorWithPadding] = None,
        label_pad_token_id: int = -100,
        padding_value: int = None,
        truncation_mode: str = "keep_end",
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        max_length: Optional[int] = None,
        max_prompt_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        peft_config: Optional[Dict] = None,
        is_encoder_decoder: Optional[bool] = None,
        disable_dropout: bool = True,
        generate_during_eval: bool = False,
        compute_metrics: Optional[Callable[[EvalLoopOutput], Dict]] = None,
        precompute_ref_log_probs: bool = False,
        model_init_kwargs: Optional[Dict] = None,
        ref_model_init_kwargs: Optional[Dict] = None,
    ):
        if model_init_kwargs is None:
            model_init_kwargs = {}
        elif not isinstance(model, str):
            raise ValueError("You passed model_kwargs to the DPOTrainer. But your model is already instantiated.")

        if ref_model_init_kwargs is None:
            ref_model_init_kwargs = {}
        elif not isinstance(ref_model, str):
            raise ValueError(
                "You passed ref_model_kwargs to the DPOTrainer. But your ref_model is already instantiated."
            )
        if isinstance(model, str):
            warnings.warn(
                "You passed a model_id to the DPOTrainer. This will automatically create an "
                "`AutoModelForCausalLM` or a `PeftModel` (if you passed a `peft_config`) for you."
            )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)

        if isinstance(ref_model, str):
            warnings.warn(
                "You passed a ref model_id to the DPOTrainer. This will automatically create an "
                "`AutoModelForCausalLM`"
            )
            ref_model = AutoModelForCausalLM.from_pretrained(ref_model, **ref_model_init_kwargs)

        if not is_peft_available() and peft_config is not None:
            raise ValueError(
                "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
            )
        elif is_peft_available() and peft_config is not None:
            # if model is a peft model and we have a peft_config, we merge and unload it first
            if isinstance(model, PeftModel):
                model = model.merge_and_unload()

            if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
                _support_gc_kwargs = hasattr(
                    args, "gradient_checkpointing_kwargs"
                ) and "gradient_checkpointing_kwargs" in list(
                    inspect.signature(prepare_model_for_kbit_training).parameters
                )

                preprare_model_kwargs = {"use_gradient_checkpointing": args.gradient_checkpointing}

                if _support_gc_kwargs:
                    preprare_model_kwargs["gradient_checkpointing_kwargs"] = args.gradient_checkpointing_kwargs

                model = prepare_model_for_kbit_training(model, **preprare_model_kwargs)
            
            elif getattr(args, "gradient_checkpointing", False):
                # For backward compatibility with older versions of transformers
                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()
                else:

                    def make_inputs_require_grad(module, input, output):
                        output.requires_grad_(True)

                    model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

            # get peft model with the given config
            model = get_peft_model(model, peft_config)

        # For models that use gradient_checkpoiting, we need to attach a hook that enables input
        # to explicitly have `requires_grad=True`, otherwise training will either silently
        # fail or completely fail.
        elif getattr(args, "gradient_checkpointing", False):
            # For backward compatibility with older versions of transformers
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        if generate_during_eval and not is_wandb_available():
            raise ValueError(
                "`generate_during_eval=True` requires Weights and Biases to be installed."
                " Please install `wandb` to resolve."
            )

        if model is not None:
            self.is_encoder_decoder = model.config.is_encoder_decoder
        elif is_encoder_decoder is None:
            raise ValueError("When no model is provided, you need to pass the parameter is_encoder_decoder.")
        else:
            self.is_encoder_decoder = is_encoder_decoder

        self.is_peft_model = is_peft_available() and isinstance(model, PeftModel)
        self.use_ref = use_ref
        self.mean_logps = mean_logps
        
        self.score_type = score_type
        if score_config is None:
            score_config = {}
        self.score_fn = get_score_fn(score_type,score_config)
        # self.rank_coef = rank_coef
        self.kl_coef = kl_coef
        self.add_klloss = False
        if self.kl_coef is not None and self.kl_coef > 0 :
            self.add_klloss = True
            # 
        # print('*********************** use ref        -> ',self.use_ref)
        # print('*********************** use mean logps -> ',self.mean_logps)
        # print('*********************** score_type       -> ',self.score_type )
        # print('*********************** kl_coef       -> ',self.kl_coef )
        
        if use_ref:
            if ref_model:
                self.ref_model = ref_model
            elif self.is_peft_model or precompute_ref_log_probs:
                # The `model` with adapters turned off will be used as the reference model
                self.ref_model = None
            else:
                self.ref_model = create_reference_model(model)
        else:
            self.ref_model = None
            
        if data_collator is None:
            if tokenizer is None:
                raise ValueError(
                    "max_length or a tokenizer must be specified when using the default DRPODataCollatorWithPadding"
                )
            if max_length is None:
                warnings.warn(
                    "When using DRPODataCollatorWithPadding, you should set `max_length` in the DPOTrainer's init"
                    " it will be set to `512` by default, but you should do it yourself in the future.",
                    UserWarning,
                )
                max_length = 512
            if max_prompt_length is None:
                warnings.warn(
                    "When using DRPODataCollatorWithPadding, you should set `max_prompt_length` in the DPOTrainer's init"
                    " it will be set to `128` by default, but you should do it yourself in the future.",
                    UserWarning,
                )
                max_prompt_length = 128

            if max_target_length is None and self.is_encoder_decoder:
                warnings.warn(
                    "When using DRPODataCollatorWithPadding with an encoder decoder architecture, you should set `max_target_length` in the DPOTrainer's init"
                    " it will be set to `128` by default, but you should do it yourself in the future.",
                    UserWarning,
                )
                max_target_length = 128

            data_collator = DRPODataCollatorWithPadding(
                pad_token_id=tokenizer.pad_token_id,
                label_pad_token_id=label_pad_token_id,
                is_encoder_decoder=self.is_encoder_decoder,
            )
            print('***************<pad id>*****************',tokenizer.pad_token_id,self.is_encoder_decoder)

            if args.remove_unused_columns:
                args.remove_unused_columns = False
                # warn users
                warnings.warn(
                    "When using DRPODataCollatorWithPadding, you should set `remove_unused_columns=False` in your TrainingArguments"
                    " we have set it for you, but you should do it yourself in the future.",
                    UserWarning,
                )
            self.use_dpo_data_collator = True
        else:
            self.use_dpo_data_collator = False

        if disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        self.max_length = max_length
        self.generate_during_eval = generate_during_eval
        print('****************generate_during_eval *******',self.generate_during_eval)
        self.label_pad_token_id = label_pad_token_id
        self.padding_value = padding_value if padding_value is not None else tokenizer.pad_token_id
        self.max_prompt_length = max_prompt_length
        self.truncation_mode = truncation_mode
        self.max_target_length = max_target_length
        self.tokenizer = tokenizer
        self.precompute_ref_log_probs = precompute_ref_log_probs

        # Since ref_logs are precomputed on the first call to get_train/eval_dataloader
        # keep track of first called to avoid computation of future calls
        self._precomputed_train_ref_log_probs = False
        self._precomputed_eval_ref_log_probs = False

        if loss_type not in ["diff", "diffndcg"]:
            warnings.warn(
                "You are using a loss type that does not support label smoothing."
            )

        self.beta = beta
        self.loss_type = loss_type
        
        self.loss_func = RANK_LOSS_TYPE[loss_type](**loss_config)
        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        # tokenize the dataset
        train_dataset = train_dataset.map(self.tokenize_row)
        
        if eval_dataset is not None:
            eval_dataset = eval_dataset.map(self.tokenize_row)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        if not hasattr(self, "accelerator"):
            raise AttributeError(
                "Your `Trainer` does not have an `accelerator` object. Consider upgrading `transformers`."
            )

        # Deepspeed Zero-3 does not support precompute_ref_log_probs
        if self.is_deepspeed_enabled:
            if self.accelerator.state.deepspeed_plugin.zero_stage == 3 and self.precompute_ref_log_probs:
                raise ValueError(
                    "You cannot use `precompute_ref_log_probs=True` with Deepspeed ZeRO-3. Please set `precompute_ref_log_probs=False`."
                )

        if self.ref_model is None:
            if not (self.is_peft_model or self.precompute_ref_log_probs) and self.use_ref:
                raise ValueError(
                    "No reference model and model is not a Peft model. Try setting `precompute_ref_log_probs=True`"
                )
        else:
            if self.is_deepspeed_enabled:
                self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

    def _prepare_deepspeed(self, model: PreTrainedModelWrapper):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        }
                    )

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        return model

    # fix to drpo
    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Subclass of transformers.src.transformers.trainer.get_train_dataloader to precompute `ref_log_probs`.
        """

        if self.precompute_ref_log_probs and not self._precomputed_train_ref_log_probs:
            dataloader_params = {
                "batch_size": self.args.per_device_train_batch_size,
                "collate_fn": self.data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "shuffle": False,
            }

            # prepare dataloader
            data_loader = self.accelerator.prepare(DataLoader(self.train_dataset, **dataloader_params))

            reference_logps = []
            for padded_batch in tqdm(iterable=data_loader, desc="Train dataset reference log probs"):
                reference_logp  = self.compute_reference_log_probs(padded_batch)
                reference_logp  = self.accelerator.gather_for_metrics(reference_logp)

                reference_logps.append(reference_logp.cpu())

            all_reference_chosen_logps = torch.cat(reference_logps).float().numpy()

            self.train_dataset = self.train_dataset.add_column(
                name="rank_reference_logps", column=all_reference_chosen_logps
            )

            self._precomputed_train_ref_log_probs = True

        return super().get_train_dataloader()

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass of transformers.src.transformers.trainer.get_eval_dataloader to precompute `ref_log_probs`.

        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
                by the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        if self.precompute_ref_log_probs and not self._precomputed_eval_ref_log_probs:
            dataloader_params = {
                "batch_size": self.args.per_device_eval_batch_size,
                "collate_fn": self.data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "shuffle": False,
            }

            # prepare dataloader
            data_loader = self.accelerator.prepare(DataLoader(eval_dataset, **dataloader_params))

            reference_logps = []
            for padded_batch in tqdm(iterable=data_loader, desc="Eval dataset reference log probs"):
                reference_logp = self.compute_reference_log_probs(padded_batch)
                reference_logp = self.accelerator.gather_for_metrics(
                    reference_logp
                )
                reference_logps.append(reference_logp.cpu())

            all_reference_logps = torch.cat(reference_logps).float().numpy()

            eval_dataset = eval_dataset.add_column(name="rank_reference_logps", column=all_reference_logps)

            # Save calculated reference_chosen_logps and reference_rejected_logps to the eval_dataset for subsequent runs
            if self.eval_dataset is not None:
                self.eval_dataset = eval_dataset
            self._precomputed_eval_ref_log_probs = True

        return super().get_eval_dataloader(eval_dataset=eval_dataset)

    def build_tokenized_answer(self, prompt, answer):
        """
        Llama tokenizer does satisfy `enc(a + b) = enc(a) + enc(b)`.
        It does ensure `enc(a + b) = enc(a) + enc(a + b)[len(enc(a)):]`.
        Reference:
            https://github.com/EleutherAI/lm-evaluation-harness/pull/531#issuecomment-1595586257
        """

        full_tokenized = self.tokenizer(prompt + answer, add_special_tokens=False)
        prompt_input_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]

        answer_input_ids = full_tokenized["input_ids"][len(prompt_input_ids) :]
        answer_attention_mask = full_tokenized["attention_mask"][len(prompt_input_ids) :]

        # Concat tokens to form `enc(a) + enc(a + b)[len(enc(a)):]`
        full_concat_input_ids = np.concatenate([prompt_input_ids, answer_input_ids])

        # Prepare input tokens for token by token comparison
        full_input_ids = np.array(full_tokenized["input_ids"])

        if len(full_input_ids) != len(full_concat_input_ids):
            raise ValueError("Prompt input ids and answer input ids should have the same length.")

        # On some tokenizers, like Llama-2 tokenizer, there are occasions where tokens
        # can be merged together when tokenizing prompt+answer. This could result
        # on the last token from the prompt being different when tokenized on its own
        # vs when done as prompt+answer.
        response_token_ids_start_idx = len(prompt_input_ids)

        # If tokenized prompt is different than both prompt+answer, then it means the
        # last token has changed due to merging.
        if prompt_input_ids != full_tokenized["input_ids"][:response_token_ids_start_idx]:
            response_token_ids_start_idx -= 1

        prompt_input_ids = full_tokenized["input_ids"][:response_token_ids_start_idx]
        prompt_attention_mask = full_tokenized["attention_mask"][:response_token_ids_start_idx]

        if len(prompt_input_ids) != len(prompt_attention_mask):
            raise ValueError("Prompt input ids and attention mask should have the same length.")

        answer_input_ids = full_tokenized["input_ids"][response_token_ids_start_idx:]
        answer_attention_mask = full_tokenized["attention_mask"][response_token_ids_start_idx:]

        return dict(
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            input_ids=answer_input_ids,
            attention_mask=answer_attention_mask,
        )

    def tokenize_row(self, feature, model: Union[PreTrainedModel, nn.Module] = None) -> Dict:
        """Tokenize a single row from a DPO specific dataset.

        At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
        in case the prompt + chosen or prompt + rejected responses is/are too long. First
            we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

        We also create the labels for the chosen/rejected responses, which are of length equal to
            the sum of the length of the prompt and the chosen/rejected response, with
            label_pad_token_id  for the prompt tokens.
        """
        batch = {}
        prompt = feature["prompt"]
        generations = feature["generations"]
        scores = feature["scores"]
        sft_index =feature["sft_index"]

        if not self.is_encoder_decoder:
            # Check issues below for more details
            #  1. https://github.com/huggingface/trl/issues/907
            #  2. https://github.com/EleutherAI/lm-evaluation-harness/pull/531#issuecomment-1595586257
            #  3. https://github.com/LianjiaTech/BELLE/issues/337
            if not isinstance(prompt, str):
                raise ValueError(f"prompt should be an str but got {type(prompt)}")
            prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)
            prompt_tokens = {f"prompt_{k}": v for k, v in prompt_tokens.items()}

            if not isinstance(generations, list):
                raise ValueError(f"generations should be an str but got {type(generations)}")
            if self.tokenizer.bos_token_id is not None:
                # print(self.tokenizer.bos_token_id)
                prompt_tokens["prompt_input_ids"] = [self.tokenizer.bos_token_id] + prompt_tokens["prompt_input_ids"]
                prompt_tokens["prompt_attention_mask"] = [1] + prompt_tokens["prompt_attention_mask"]
            
            rank_list = []
            # print('prompt->',prompt)
            for response in generations:
                rank_prompt_tokens = prompt_tokens
                # print('response -> ',response)
                res_tokens = self.build_tokenized_answer(prompt, response)
                # add BOS token to head of prompt
                if self.tokenizer.bos_token_id is not None:
                    # print(self.tokenizer.bos_token_id)
                    res_tokens["prompt_input_ids"] = [self.tokenizer.bos_token_id] + res_tokens["prompt_input_ids"]
                    res_tokens["prompt_attention_mask"] = [1] + res_tokens["prompt_attention_mask"]

                # add EOS token to end of answer
                res_tokens["input_ids"].append(self.tokenizer.eos_token_id)
                res_tokens["attention_mask"].append(1)

                response_length = len(res_tokens["input_ids"])
                # if combined sequence is too long, truncate the prompt
                for answer_tokens in [res_tokens, rank_prompt_tokens]:
                    if len(answer_tokens["prompt_input_ids"]) + response_length > self.max_length:
                        if self.truncation_mode == "keep_start":
                            for k in ["prompt_input_ids", "prompt_attention_mask"]:
                                answer_tokens[k] = answer_tokens[k][: self.max_prompt_length]
                        elif self.truncation_mode == "keep_end":
                            for k in ["prompt_input_ids", "prompt_attention_mask"]:
                                answer_tokens[k] = answer_tokens[k][-self.max_prompt_length :]
                        else:
                            raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")

                # if that's still too long, truncate the response
                if len(res_tokens["prompt_input_ids"]) + response_length > self.max_length:
                    for k in ["input_ids", "attention_mask"]:
                        res_tokens[k] = res_tokens[k][: self.max_length - self.max_prompt_length]

                res_sequence_tokens = {
                    f"target_{k}": res_tokens[f"prompt_{k}"] + res_tokens[k] for k in ["input_ids", "attention_mask"]
                }
                res_sequence_tokens["target_labels"] = res_sequence_tokens["target_input_ids"][:]
                res_sequence_tokens["target_labels"][: len(res_tokens["prompt_input_ids"])] = [self.label_pad_token_id] * len(res_tokens["prompt_input_ids"])
                # print(res_sequence_tokens.keys())
                rank_list.append(res_sequence_tokens)

            batch["rank_combined_list"] = rank_list
            batch["target_scores"] = scores
            batch["sft_index"] = sft_index

            for type_key, tokens in prompt_tokens.items():
                if type_key == "token_type_ids":
                    continue
                batch[f"{type_key}"] = tokens
        else:
            raise NotImplementedError

        return batch

    def compute_reference_log_probs(self, padded_batch: Dict) -> Dict:
        """Computes log probabilities of the reference model for a single padded batch of a DPO specific dataset."""
        # compute reference logps
        with torch.no_grad():
            if self.ref_model is None:
                with self.accelerator.unwrap_model(
                    self.model
                ).disable_adapter() if self.is_peft_model else nullcontext():
                    reference_logps , _ = self.concatenated_forward(self.model, padded_batch)
            else:
                reference_logps , _ = self.concatenated_forward(self.ref_model, padded_batch)

        return reference_logps

    def drpo_loss(
        self,
        pred_scores: torch.FloatTensor,
        scores: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:

        labels = scores
        losses = self.loss_func(pred_scores,labels)
  
        return losses

    @staticmethod
    def get_rank_batch_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
        label_pad_token_id: int = -100,
        is_encoder_decoder: bool = False,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size*K, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size*K, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

        Returns:
            A tensor of shape (batch_size*K,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        if not is_encoder_decoder:
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]
        loss_mask = labels != label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == label_pad_token_id] = 0


        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)
    

    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]],average_log_prob=False
    ):
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        N,R,L = batch['rank_target_input_ids'].shape

        rank_target_input_ids = batch['rank_target_input_ids'].reshape(N*R, -1)
        rank_target_attention_mask = batch['rank_target_attention_mask'].reshape(N*R, -1)
        rank_target_labels = batch['rank_target_labels'].reshape(N*R, -1)


        outputs = model(rank_target_input_ids,
                           attention_mask=rank_target_attention_mask
                           )
        all_logits = outputs.logits
        all_logps = self.get_rank_batch_logps(
            all_logits,
            rank_target_labels,
            average_log_prob=average_log_prob,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )
        all_logps = all_logps.reshape(N,R)

        return all_logps,all_logits

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        
        metrics = {}
        prefix = "eval_" if train_eval == "eval" else ""
        
        N,R,_ = batch['rank_target_input_ids'].shape
        
        label_scores = batch['target_scores'].detach()
        max_idx = torch.argmax(label_scores, dim=-1, keepdim=True)
        min_idx = torch.argmin(label_scores, dim=-1, keepdim=True)

        policy_logps,all_logits = self.concatenated_forward(model, batch,average_log_prob=self.mean_logps)

        if self.use_ref:
            if "rank_reference_logps" in batch:
                reference_logps = batch["rank_reference_logps"]
            else:
                with torch.no_grad():
                    if self.ref_model is None:
                        with self.accelerator.unwrap_model(self.model).disable_adapter():
                            reference_logps, ref_logits = self.concatenated_forward(self.model, batch, average_log_prob=self.mean_logps)
                    else:
                        reference_logps, ref_logits = self.concatenated_forward(self.ref_model, batch)
        else:
            reference_logps = None
            ref_logits = None
        
        pred_scores,score_metrics = self.score_fn(
            policy_logps = policy_logps,
            reference_logps = reference_logps,
            policy_logits = all_logits,
            reference_logits = ref_logits,
            label_scores = batch['target_scores'],
            mean_probs = self.mean_logps
        )
        metrics.update(score_metrics)

        drpo_losses =  self.drpo_loss(
            pred_scores,
            batch['target_scores'],
        )
        drpo_losses_item = drpo_losses.detach().cpu().mean()
        
        if reference_logps is not None:
            rewards = self.beta * (policy_logps - reference_logps).detach()
        else:
            rewards = None
            
        losses = 0
        losses += drpo_losses

        metrics[f"{prefix}_drpo_loss"] = drpo_losses_item
        

        max_props = policy_logps.gather(-1, max_idx).detach().squeeze()
        min_props = policy_logps.gather(-1, min_idx).detach().squeeze()
        props_accuracies = (max_props > min_props).float()

        if rewards is not None:
            max_rewards = rewards.gather(-1, max_idx).detach().squeeze()
            min_rewards = rewards.gather(-1, min_idx).detach().squeeze()
            reward_accuracies = (max_rewards > min_rewards).float()
            metrics[f"{prefix}rewards/max"] = max_rewards.cpu().mean()
            metrics[f"{prefix}rewards/min"] = min_rewards.cpu().mean()
            metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.cpu().mean()
            metrics[f"{prefix}rewards/margins"] = (max_rewards - min_rewards).cpu().mean()
        
        metrics[f"{prefix}logps/max"] = max_props.detach().cpu().mean()
        metrics[f"{prefix}logps/min"] = min_props.detach().cpu().mean()
        metrics[f"{prefix}logps/acc"] = props_accuracies.detach().cpu().mean()

        metrics[f"{prefix}logps/sum_acc_v2"] = calculate_pairwise_accuracy(policy_logps,batch['target_scores']).detach().cpu().mean()
        metrics[f"{prefix}score/sum_acc_v2"] = calculate_pairwise_accuracy(pred_scores,batch['target_scores']).detach().cpu().mean()
        metrics[f"{prefix}logps/ndcg"] = compute_ndcg(policy_logps,label_scores).detach().cpu().mean()
        metrics[f"{prefix}score/ndcg"] = compute_ndcg(pred_scores,label_scores).detach().cpu().mean()


        return losses.mean(), metrics

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if not self.use_dpo_data_collator:
            warnings.warn(
                "compute_loss is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )

        loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")

        if self.accelerator.is_main_process:
            self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return (loss, metrics)
        return loss

    def get_batch_samples(self, model, batch: Dict[str, torch.LongTensor]) -> Tuple[str, str]:
        """Generate samples from the model and reference model for the given batch of inputs."""

        policy_output = model.generate(
            input_ids=batch["prompt_input_ids"],
            attention_mask=batch["prompt_attention_mask"],
            max_length=self.max_length,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # if reference_output in batch use that otherwise use the reference model
        if "reference_output" in batch:
            reference_output = batch["reference_output"]
        else:
            if self.ref_model is None:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    reference_output = self.model.generate(
                        input_ids=batch["prompt_input_ids"],
                        attention_mask=batch["prompt_attention_mask"],
                        max_length=self.max_length,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
            else:
                reference_output = self.ref_model.generate(
                    input_ids=batch["prompt_input_ids"],
                    attention_mask=batch["prompt_attention_mask"],
                    max_length=self.max_length,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

        policy_output = pad_to_length(policy_output, self.max_length, self.tokenizer.pad_token_id)
        policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)

        reference_output = pad_to_length(reference_output, self.max_length, self.tokenizer.pad_token_id)
        reference_output_decoded = self.tokenizer.batch_decode(reference_output, skip_special_tokens=True)

        return policy_output_decoded, reference_output_decoded
    
    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):

        st_time = time.time()
        if ignore_keys is None:
            if hasattr(model, "config"):
                ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        with torch.no_grad():
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="eval")

        # force log the metrics
        metrics['rank_time'] = time.time()-st_time
        print('run time ',time.time()-st_time)
        if self.accelerator.is_main_process:
            self.store_metrics(metrics, train_eval="eval")
        
        if prediction_loss_only:
            return (loss.detach(), None, None)

        return (loss.detach(), None, None)

    def store_metrics(self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:

        """
        Overriding built-in evaluation loop to store metrics for each batch.
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.
        Works both with or without labels.
        """

        initial_output = super().evaluation_loop(
            dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix
        )

        return initial_output

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training, including stored metrics.
        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        
        train_eval = "train" if "loss" in logs else "eval"
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]
        return super().log(logs)

    @wraps(Trainer.push_to_hub)
    def push_to_hub(self, commit_message: Optional[str] = "End of training", blocking: bool = True, **kwargs) -> str:
        """
        Overwrite the `push_to_hub` method in order to force-add the tag "sft" when pushing the
        model on the Hub. Please refer to `~transformers.Trainer.push_to_hub` for more details.
        """
        kwargs = trl_sanitze_kwargs_for_tagging(tag_names=self._tag_names, kwargs=kwargs)

        return super().push_to_hub(commit_message=commit_message, blocking=blocking, **kwargs)
