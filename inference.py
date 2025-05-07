import torch
import torch.nn.functional as F
import tiktoken
import torch.optim.adamw

from model.gpt import GPT
from config import GPTConfig


num_samples = 5
max_len = 30
start_sequennce = "Hello, I'm a language model,"
start_sequennce = "नमस्ते, मैं एक भाषा प्रतिरूप हूँ,"
start_sequennce = "नमस्ते"
start_sequennce = "Hello"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = GPT.from_pretrained("gpt2")
# config=GPTConfig()
# model = GPT(config)
model.eval()
model.to(device)

enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode(start_sequennce)
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_samples, 1)
x = tokens.to(device)

torch.manual_seed(42)
torch.cuda.manual_seed(42)

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
    decoded = enc.decode(tokens)
    print(">", decoded)