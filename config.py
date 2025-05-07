from dataclasses import dataclass

@dataclass
class GPTConfig:

    # Data
    shard_size = int(1e8)
    
    # Model
    max_sequence_len: int = 1024
    vocab_size: int = 50000 + 256 + 1
    n_layer: int = 12
    n_head: int = 12
    d_model: int = 768
    mlp_hidden_dim: int = 4 * d_model
    activation: str = "GELU"

    # Optimizer
    max_lr = 6e-4
    min_lr = max_lr * 0.1
    beta_1 = 0.9
    beta_2 = 0.95
    eps = 1e-8
    weight_decay = 0.1
    
    # Training
    warmup_steps = 1440
    max_steps = 39459
    total_batch_size = 524288
    micro_batch_size = 64        # B = 64
    sequence_len = 1024          # T = 1024

    # Training Config
    resume_from_checkpoint = False
    checkpoint_path = "logs/exp_2025-05-05_10-50-34/checkpoints/model_039000.pt"

