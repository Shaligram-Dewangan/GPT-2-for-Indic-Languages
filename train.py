import os
import time
import math
import datetime
import torch
from itertools import islice
from torch.utils.tensorboard import SummaryWriter

from model.gpt import GPT
from data.dataloader import DataLoader
from evaluation.hellaswag import Hellaswag
from config import GPTConfig


# tensorboard logging
expt_name = f"exp_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
writer = SummaryWriter(log_dir=f"logs/{expt_name}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

torch.manual_seed(1337)
torch.cuda.manual_seed(1337)

config = GPTConfig(vocab_size=50304)

B = config.micro_batch_size
T = config.sequence_len

assert config.total_batch_size % (B * T) == 0, "make sure total batch size is divisible by (B * T)"
grad_accum_steps = config.total_batch_size // (B * T)
print(f"total desired batch size: {config.total_batch_size}")
print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoader(B=B, T=T, split="train")
val_loader = DataLoader(B=B, T=T, split="val")

hs_eng_eval = Hellaswag(language="english")
hs_hi_eval = Hellaswag(language="hindi")

if device == "cuda": torch.set_float32_matmul_precision("high")

model = GPT(config)
model.to(device)

use_torch_compile = True
if device == "cuda" and use_torch_compile:
    model = torch.compile(model)

def get_lr(it):
    if it < config.warmup_steps:
        return config.max_lr * (it+1) / config.warmup_steps
    if it > config.max_steps:
        return config.min_lr
    decay_ratio = (it - config.warmup_steps) / (config.max_steps - config.warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.max_lr - config.min_lr)

# Optimize
optimizer = model.configure_optimizers(config, device=device)

def load_checkpoint(filepath, model, optimizer=None, map_location=None):
    checkpoint = torch.load(filepath, map_location=torch.device(map_location), weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    step = checkpoint.get('step', 0)
    loss = checkpoint.get('loss', None)

    if 'rng_state' in checkpoint:
        torch.set_rng_state(checkpoint['rng_state'])
    if torch.cuda.is_available() and checkpoint.get('cuda_rng_state') is not None:
        torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state'])

    print(f"Checkpoint loaded from step {step}, with loss: {loss}")
    return step, loss

start_step = 0
if config.resume_from_checkpoint:
    start_step, _ = load_checkpoint(config.checkpoint_path, model, optimizer, map_location=device)
    train_loader = DataLoader(B=B, T=T, split="train", step=start_step, config=config)

for step in range(start_step, config.max_steps): 
    
    last_step = (step == config.max_steps - 1)

    # ----- Evaluation -----
    if step % 250 == 0 or last_step:
        t0 = time.time()
        

        # Validation Loss
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                if device == "cuda":
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        logits, loss = model(x, y)
                else:
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        writer.add_scalar("Evaluation/ Validation Loss", val_loss_accum.item(), step)
        print(f"\nvalidation loss: {val_loss_accum.item():.4f}")


        # HellaSwag Evaluation
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(islice(hs_eng_eval.iterate_examples("val"), None)):
            _, tokens, mask, label = hs_eng_eval.render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            with torch.no_grad():
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(tokens)
                pred_norm = hs_eng_eval.get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        acc_norm = num_correct_norm / num_total
        writer.add_scalar("Evaluation/ HellaSwag-Eng", acc_norm, step)
        print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")


        # Hindi-HellaSwag Evaluation
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(islice(hs_hi_eval.iterate_examples("val"), None)):
            _, tokens, mask, label = hs_hi_eval.render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            with torch.no_grad():
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(tokens)
                pred_norm = hs_hi_eval.get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        acc_norm = num_correct_norm / num_total
        writer.add_scalar("Evaluation/ Hindi-HellaSwag", acc_norm, step)
        print(f"Hindi-HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")


        t1 = time.time()
        print(f"Eval dt: {(t1-t0):.2f}s (Val + eng_HS + hi_HS)\n")
    # ----- ---------- -----

    t0 = time.time()

    optimizer.zero_grad()
    loss_accum = 0

    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        if device == "cuda":
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = model(x, y)
        else:
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.step()

    if device == "cuda": torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0) # in seconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps
    tokens_per_sec = tokens_processed / dt
    print(f"step {step} | loss: {loss_accum.item():.6f} | norm: {norm:.4f} | lr: {lr:.4e} | dt: {(dt):.2f}s | tok/sec: {tokens_per_sec:.2f}")

    # save checkpoint
    if step > 0 and (step % 1000 == 0 or last_step):
        os.makedirs(f"logs/{expt_name}/checkpoints/", exist_ok=True)
        checkpoint_path = f"logs/{expt_name}/checkpoints/model_{step:06d}.pt"
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': model.config,
            'step': step,
            'loss': loss_accum.item(),
            'rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint model_{step:06d}.pt saved!")
    
    # tensorboard logging
    writer.add_scalar("Loss", loss_accum.item(), step)
    writer.add_scalar("Train Metrics/ Norm", norm, step)
    writer.add_scalar("Train Metrics/ Learning Rate", lr, step)
    writer.add_scalar("Performance/ Step Time (ms)", dt * 1000, step)
    writer.add_scalar("Performance/ Tokens_per_Second", tokens_per_sec, step)

writer.close()