import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from model.gpt import GPT
from config import GPTConfig

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

torch.manual_seed(42)
torch.cuda.manual_seed(42)

start_sequennce = "Hello, I'm a language model,"
start_sequennce = "नमस्ते, मैं एक भाषा प्रतिरूप हूँ,"

num_samples = 5
max_len = 32

config = GPTConfig(vocab_size=50304)
model = GPT(config)
model.eval()
model.to(device)

model_path = "logs/server_exp_2025-05-05_10-50-34/checkpoints/model_039000.pt"
state_dict = torch.load(model_path, map_location=device, weights_only=False)
config = state_dict['config']

model_state_dict = state_dict['model_state_dict']
unwanted_prefix = '_orig_mod.'
for k,v in list(model_state_dict.items()):
    if k.startswith(unwanted_prefix):
        model_state_dict[k[len(unwanted_prefix):]] = model_state_dict.pop(k)

model.load_state_dict(state_dict['model_state_dict'])

tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")
tokens = tokenizer.encode(start_sequennce).ids
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_samples, 1)
x = tokens.to(device)

while x.size(1) < max_len:
    with torch.no_grad():
        logits = model(x)
        logits = logits[0][:, -1, :]

        probalities = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probalities, 50, dim=-1)
        ix = torch.multinomial(topk_probs, 1)
        xcol = torch.gather(topk_indices, -1, ix)
        
        x = torch.cat((x, xcol), dim=1)

for i in range(num_samples):
    tokens = x[i, :max_len].tolist()
    decoded = tokenizer.decode(tokens)
    print(">", decoded)