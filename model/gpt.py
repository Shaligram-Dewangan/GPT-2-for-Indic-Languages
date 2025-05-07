import torch
import torch.nn as nn
from torch.nn import functional as F
import inspect

from .block import Block
from config import GPTConfig


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte = nn.Embedding(config.vocab_size, config.d_model),
                wpe = nn.Embedding(config.max_sequence_len, config.d_model),
                h = nn.ModuleList([
                    Block(config) for i in range(config.n_layer)
                ]),
                ln_f = nn.LayerNorm(config.d_model)
            )
        )
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "SCALE_INIT_FLAG"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.max_sequence_len, f"Cannot forward the sequence len of {T}, max sequence length is only {self.config.max_sequence_len}"
        
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss
    

    def configure_optimizers(self, config, device):
        """Configure the optimizer with weight decay applied selectively to 2D parameters."""

        # Collect all parameters that require gradients
        param_dict = {name: param for name, param in self.named_parameters() if param.requires_grad}

        # Split parameters into decay and no-decay groups
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        # Define optimizer parameter groups
        optim_groups = [
            {'params': decay_params, 'weight_decay': config.weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        # Logging parameter counts
        print(f"Number of decayed parameter tensors: {len(decay_params)}, total parameters: {sum(p.numel() for p in decay_params):,}")
        print(f"Number of non-decayed parameter tensors: {len(nodecay_params)}, total parameters: {sum(p.numel() for p in nodecay_params):,}")

        # Check if the fused version of AdamW is available and use it on CUDA
        use_fused = 'fused' in inspect.signature(torch.optim.AdamW).parameters and device == "cuda"
        print(f"Using fused AdamW: {use_fused}")

        # Initialize AdamW optimizer
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=config.max_lr,
            betas=(config.beta_1, config.beta_2),
            eps=config.eps,
            fused=use_fused
        )

        return optimizer


    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from Hugging Face"""
        from transformers import GPT2LMHeadModel

        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}, f"Invalid model_type: {model_type}"
    
        print(f"Loading weights from pretrained GPT: {model_type}")

        model_configs = {
        'gpt2':        {'n_layer': 12, 'n_head': 12, 'd_model': 768},   # 124M params
        'gpt2-medium': {'n_layer': 24, 'n_head': 16, 'd_model': 1024},  # 350M params
        'gpt2-large':  {'n_layer': 36, 'n_head': 20, 'd_model': 1280},  # 774M params
        'gpt2-xl':     {'n_layer': 48, 'n_head': 25, 'd_model': 1600},  # 1558M params
        }

        config_args = model_configs[model_type]
        config_args.update({
            'vocab_size': 50257,
            'max_sequence_len': 1024,
            'mlp_hidden_dim': 4 * model_configs[model_type]['d_model']
        })

        # Create a from-scratch initialized GPT model
        config = GPTConfig(**config_args)
        model = GPT(config)

        # Load state dictionary and filter unnecessary keys
        sd = {k: v for k, v in model.state_dict().items() if not k.endswith('.attn.bias')}

        # Initialize Hugging Face model and load pretrained weights
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = {
            k: v for k, v in model_hf.state_dict().items()
            if not k.endswith(('.attn.masked_bias', '.attn.bias'))
        }

        # Ensure key counts match between models
        assert len(sd_hf) == len(sd), f"Mismatched keys: {len(sd_hf)} != {len(sd)}"

        # Handle weight transpositions (OpenAI Conv1D to Linear conversion)
        transposed_layers = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        for k in sd_hf:
            with torch.no_grad():
                if any(k.endswith(layer) for layer in transposed_layers):
                    # Transpose specific Conv1D weights to match Linear layer shape
                    assert sd_hf[k].shape[::-1] == sd[k].shape, f"Shape mismatch on transposed key: {k}"
                    sd[k].copy_(sd_hf[k].T)
                else:
                    # Standard weight copy for other parameters
                    assert sd_hf[k].shape == sd[k].shape, f"Shape mismatch on key: {k}"
                    sd[k].copy_(sd_hf[k])

        return model
