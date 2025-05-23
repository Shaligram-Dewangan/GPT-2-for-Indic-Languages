import os
import json
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

class Hellaswag:

    def __init__(self, language):
        self.DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "eval_datasets")

        self.language = language
        assert language == "english" or language == "hindi", f"'language' must be 'english' or 'hindi', and not {self.language}"
        self.eval_name = "hellaswag" if self.language == "english" else "hindi_hellaswag"

        self.tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")

    def render_example(self, example):
        """
        Given the example as a dictionary, render it as three torch tensors:
        - tokens (the tokens of context + completion, of size 4xN, as there are always 4 candidates)
        - mask (is 1 in the region of the candidate completion, where we evaluate likelihoods)
        - label (the index of the correct completion, which we hope has the highest likelihood)
        """
        ctx = example["ctx"]
        label = example["label"]
        endings = example["endings"]

        # data needed to reproduce this eval on the C size
        data = {
            "label": label,
            "ctx_tokens": None,
            "ending_tokens": [],
        }

        # gather up all the tokens
        ctx_tokens = self.tokenizer.encode(ctx).ids
        data["ctx_tokens"] = ctx_tokens
        tok_rows = []
        mask_rows = []
        for end in endings:
            end_tokens = self.tokenizer.encode(" " + end).ids
            tok_rows.append(ctx_tokens + end_tokens)
            mask_rows.append([0]*len(ctx_tokens) + [1]*len(end_tokens))
            data["ending_tokens"].append(end_tokens)

        # have to be careful during the collation because the number of tokens in each row can differ
        max_len = max(len(row) for row in tok_rows)
        tokens = torch.zeros((4, max_len), dtype=torch.long)
        mask = torch.zeros((4, max_len), dtype=torch.long)
        for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
            tokens[i, :len(tok_row)] = torch.tensor(tok_row)
            mask[i, :len(mask_row)] = torch.tensor(mask_row)

        return data, tokens, mask, label

    def iterate_examples(self, split):
        with open(os.path.join(self.DATA_CACHE_DIR, f"{self.eval_name}_{split}.jsonl"), "r") as f:
            for line in f:
                example = json.loads(line)
                yield example

    def get_most_likely_row(self, tokens, mask, logits):
        """inputs: tokens, mask, and logits
           returns: index of the completion with the lowest loss"""
        
        # evaluate the autoregressive loss at all positions
        shift_logits = (logits[..., :-1, :]).contiguous()
        shift_tokens = (tokens[..., 1:]).contiguous()
        flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_shift_tokens = shift_tokens.view(-1)
        shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
        shift_losses = shift_losses.view(tokens.size(0), -1)

        # now get the average loss just for the completion region (where mask == 1), in each row
        shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
        masked_shift_losses = shift_losses * shift_mask

        # sum and divide by the number of 1s in the mask
        sum_loss = masked_shift_losses.sum(dim=1)
        avg_loss = sum_loss / shift_mask.sum(dim=1)

        # now we have a loss for each of the 4 completions
        # the one with the lowest loss should be the most likely
        pred_norm = avg_loss.argmin().item()
        return pred_norm