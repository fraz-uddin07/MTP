# models/student/medusa_llama_student.py

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType

# -----------------------
# MEDUSA head: SiLU -> +residual -> vocab
#   p_t^(k) = softmax(W2^(k) * ( SiLU(W1^(k) * h_t) + h_t ))
#   with init: W1^(k) = 0, W2^(k) = LM head
# -----------------------
class _MedusaSingleHead(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, lm_head: nn.Module | None):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_model, bias=True)
        # mirror LM head bias presence if available
        use_bias_w2 = bool(getattr(lm_head, "bias", None) is not None) if lm_head is not None else False
        self.w2 = nn.Linear(d_model, vocab_size, bias=use_bias_w2)
        self._init_from_lm_head(lm_head)

    def _init_from_lm_head(self, lm_head: nn.Module | None):
        # W1 = 0 so SiLU(W1*h) == 0 -> output starts identical to LM head
        with torch.no_grad():
            self.w1.weight.zero_()
            if self.w1.bias is not None:
                self.w1.bias.zero_()

            if lm_head is not None and hasattr(lm_head, "weight"):
                if self.w2.weight.shape == lm_head.weight.shape:
                    self.w2.weight.copy_(lm_head.weight.detach())
                else:
                    # fallback if shapes differ (shouldn't for standard CausalLM)
                    print(f"[warn] Medusa head W2 shape {self.w2.weight.shape} != LM head shape {lm_head.weight.shape}. Using Xavier init.")
                    nn.init.xavier_uniform_(self.w2.weight)
                # bias
                if self.w2.bias is not None:
                    if getattr(lm_head, "bias", None) is not None and self.w2.bias.shape == lm_head.bias.shape:
                        self.w2.bias.copy_(lm_head.bias.detach())
                    else:
                        print(f"[warn] Medusa head W2 bias shape mismatch or LM head has no bias. Zeroing bias.")
                        self.w2.bias.zero_()
            else:
                print(f"[warn] No LM head provided or it lacks weights. Using Xavier init for Medusa head W2.")
                nn.init.xavier_uniform_(self.w2.weight)
                if self.w2.bias is not None:
                    self.w2.bias.zero_()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B*S, D] or [B, S, D] - reshape handled in MultiTokenHead
        z = F.silu(self.w1(h)) + h      # residual before vocab proj
        return self.w2(z)               # output shape matches input h shape except last dim -> V


class MultiTokenHead(nn.Module):
    def __init__(self, d_model, vocab_size, offsets=(1, 2), lm_head: nn.Module | None = None):
        super().__init__()
        self.offsets = tuple(int(o) for o in offsets)
        if not self.offsets:
             raise ValueError("Offsets cannot be empty.")
        # Ensure offsets are positive integers
        if not all(isinstance(o, int) and o > 0 for o in self.offsets):
             raise ValueError(f"Offsets must be positive integers, got: {self.offsets}")

        self.heads = nn.ModuleDict()
        print(f"[info] Initializing Medusa heads for offsets: {self.offsets}")
        for off in self.offsets:
            self.heads[str(off)] = _MedusaSingleHead(d_model, vocab_size, lm_head)

    def forward(self, hidden_states: torch.Tensor, last_only: bool = False):
        """
        Calculates logits for specified future tokens using Medusa heads.

        Args:
            hidden_states: [B, S, D] tensor from the backbone's last layer.
            last_only: If True, only use the hidden state of the last token [:, 
            :, :].

        Returns:
            dict: {offset_k: [B, S_or_1, V] logits tensor for offset k}
        """
        # Select relevant hidden states
        hs = hidden_states[:, -1:, :] if last_only else hidden_states # Shape [B, S_out, D] where S_out is 1 or S
        B, S_out, D = hs.shape

        # Reshape for efficient linear layer application if needed
        # Note: If last_only is True, S_out=1, reshape has minimal impact.
        # If last_only is False, we process all sequence positions.
        x = hs.reshape(-1, D) # Shape [B * S_out, D]

        output_logits = {}
        for off_str, head_module in self.heads.items():
            # Apply the specific head: [B * S_out, D] -> [B * S_out, V]
            head_logits_flat = head_module(x)
            # Reshape back to [B, S_out, V]
            head_logits = head_logits_flat.reshape(B, S_out, -1)
            # Store with integer offset key
            output_logits[int(off_str)] = head_logits

        return output_logits

# -----------------------
# UPDATED MODEL CLASS with hidden_states fix
# -----------------------
class MedusaLlamaStudent(nn.Module):
    """
    Llama model augmented with Medusa heads and optional QLoRA adapters.
    """
    def __init__(
        self,
        model_name="NousResearch/Llama-2-7b-chat-hf", # Default to 7B Llama
        offsets=(1, 2),
        lora=True,
        qlora=True,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        lora_targets=("q_proj", "v_proj"), # Correct defaults for Llama
    ):
        super().__init__()
        self.model_name = model_name
        self.offsets = tuple(offsets) # Store offsets

        # --- QLoRA Setup ---
        quantization_config = None
        if qlora:
            # Check for GPU availability for QLoRA
            if not torch.cuda.is_available():
                raise RuntimeError("QLoRA requires CUDA. No GPU detected.")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                # Use bfloat16 for compute if supported, otherwise float16
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            print("[info] QLoRA enabled. Loading base model in 4-bit.")
        else:
            print("[info] QLoRA disabled. Loading base model in default precision (likely fp16/bf16).")

        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",  # Handles device placement for base model (required for QLoRA)
            trust_remote_code=True # Often needed for chat models
        )
        # --- End QLoRA Setup ---

        # Apply LoRA adapters if requested
        if lora:
            print(f"[info] Applying LoRA adapters with r={lora_r}, alpha={lora_alpha}, targets={lora_targets}")
            cfg = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                target_modules=list(lora_targets),
                task_type=TaskType.CAUSAL_LM,
            )
            self.backbone = get_peft_model(self.backbone, cfg)
            print("[info] LoRA adapters applied.")
            # Optional: Print trainable parameters for verification
            # self.backbone.print_trainable_parameters()
        elif qlora:
             # If QLoRA is enabled but LoRA isn't, PEFT isn't applied yet.
             # We still need trainable parameters for the trainer later.
             # This scenario might need adjustments depending on how PEFT handles QLoRA without LoRA layers.
             # Typically QLoRA implies LoRA. If LoRA is truly disabled, ensure some part is trainable or adjust trainer.
             print("[warn] QLoRA is enabled but LoRA is disabled. Ensure some parameters are trainable or adjust trainer.")


        self.config = self.backbone.config
        # Handle potential variations in config attribute names
        d_model = getattr(self.config, "hidden_size", None)
        vocab_size = getattr(self.config, "vocab_size", None)
        if d_model is None or vocab_size is None:
             raise ValueError(f"Could not determine hidden_size or vocab_size from model config: {self.config}")

        # Get device from backbone (crucial after device_map)
        try:
             # Find the first parameter's device - more robust for device_map
             self.device = next(self.backbone.parameters()).device
        except StopIteration:
             print("[warn] Could not determine backbone device automatically. Defaulting.")
             self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[info] Backbone placed on device: {self.device}")


        # --- Initialize Medusa Head ---
        # IMPORTANT: Pass the potentially PEFT-wrapped LM head
        # PEFT handles accessing the underlying lm_head correctly
        lm_head_module = getattr(self.backbone, 'lm_head', None)
        if lm_head_module is None:
             # Try accessing through base_model if PEFT wrapped it deeper
             if hasattr(self.backbone, 'base_model') and hasattr(self.backbone.base_model, 'lm_head'):
                  lm_head_module = self.backbone.base_model.lm_head
             else:
                  raise AttributeError("Could not find 'lm_head' on the backbone model or its base_model.")

        self.mtp_head = MultiTokenHead(
            d_model, vocab_size, offsets, lm_head=lm_head_module
        )
        # Move the new head module to the *same device* as the backbone's parameters
        self.mtp_head.to(self.device)
        print(f"[info] Medusa head initialized and moved to device: {self.device}")
        # --- End Medusa Head Init ---


        # --- Tokenizer Setup ---
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            print("[info] Tokenizer has no pad_token. Setting pad_token = eos_token.")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Ensure padding side is right for causal LMs during generation/batching
        self.tokenizer.padding_side = "right"
        # --- End Tokenizer Setup ---

        # No need for a final self.to(self.device) as device_map handles backbone
        # and we manually moved mtp_head.

    def forward(self, input_ids, attention_mask=None, **kwargs):
        """
        Forward pass for training. Returns standard LM logits and Medusa head logits.
        """
        # Get outputs from the PEFT-wrapped backbone
        # We need hidden states for Medusa heads, so request them
        # Pass any additional kwargs (like use_cache=False during training if needed)
        backbone_outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,  # <-- REQUEST HIDDEN STATES
            return_dict=True,
            **kwargs
        )

        # --- CORRECTED hidden_states ACCESS ---
        # outputs.hidden_states is a tuple of states from all layers (including embeddings)
        # We need the last layer's output
        if not backbone_outputs.hidden_states:
             raise ValueError("Backbone did not return hidden states. Ensure `output_hidden_states=True`.")
        # Index -1 gets the last tuple element, which is the last layer's hidden state
        hidden_states = backbone_outputs.hidden_states[-1] # Shape: [B, S, D]
        # --- END CORRECTION ---

        # Calculate standard LM logits using the potentially PEFT-wrapped lm_head
        lm_logits = self.backbone.lm_head(hidden_states)     # Shape: [B, S, V]

        # Calculate Medusa head logits using the same hidden states
        # last_only=False to get logits for all sequence positions (needed for loss calc)
        mtp_logits_full = self.mtp_head(hidden_states, last_only=False) # Dict: {k: [B, S, V]}

        # Prepare output dictionary (compatible with trainer and loss function)
        # Also include 'mtp_logits' for the last token only, potentially useful for inference/eval
        mtp_logits_last = {k: v[:, -1:, :] for k, v in mtp_logits_full.items()} # Dict: {k: [B, 1, V]}

        return {
            "lm_logits": lm_logits,             # For backbone loss (KL divergence)
            "mtp_logits_full": mtp_logits_full, # For Medusa head loss (CE)
            "mtp_logits": mtp_logits_last       # Logits for last token only (inference)
        }

    def save_pretrained(self, save_directory):
        """Saves PEFT adapters, Medusa head, and metadata."""
        os.makedirs(save_directory, exist_ok=True)
        print(f"[info] Saving model components to {save_directory}")

        # Save PEFT adapter state + config (saves only adapters, not base model)
        if hasattr(self.backbone, "save_pretrained"):
            try:
                self.backbone.save_pretrained(save_directory)
                print("[info] PEFT adapters saved successfully.")
            except Exception as e:
                print(f"[error] Failed to save PEFT adapters: {e}")
        else:
            # Fallback for non-PEFT model (shouldn't happen if LoRA enabled)
            print("[warn] Backbone doesn't have save_pretrained (not a PEFT model?). Saving full state dict.")
            torch.save(self.backbone.state_dict(), os.path.join(save_directory, "backbone.bin"))

        # Save the custom Medusa head separately
        try:
            mtp_head_path = os.path.join(save_directory, "mtp_head.pt")
            torch.save(self.mtp_head.state_dict(), mtp_head_path)
            print(f"[info] Medusa head saved successfully to {mtp_head_path}")
        except Exception as e:
            print(f"[error] Failed to save Medusa head state dict: {e}")

        # Save metadata to reconstruct the class
        try:
            meta_path = os.path.join(save_directory, "mtp_meta.json")
            with open(meta_path, "w") as f:
                json.dump(
                    {
                        "model_name": self.model_name, # Base model name
                        "offsets": list(self.mtp_head.offsets) # Medusa offsets used
                    },
                    f,
                    indent=4
                )
            print(f"[info] Metadata saved successfully to {meta_path}")
        except Exception as e:
            print(f"[error] Failed to save metadata: {e}")

        # Save tokenizer
        if hasattr(self.tokenizer, "save_pretrained"):
             try:
                  self.tokenizer.save_pretrained(save_directory)
                  print("[info] Tokenizer saved successfully.")
             except Exception as e:
                  print(f"[warn] Failed to save tokenizer: {e}")