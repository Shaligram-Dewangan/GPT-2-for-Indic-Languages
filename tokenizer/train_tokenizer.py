from tokenizers import Regex
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Split
from tokenizers.decoders import BPEDecoder


ascii_chars = [chr(i) for i in range(32, 127)]
devanagari_chars = ['अ','आ','इ','ई','उ','ऊ','ए','ऐ','ओ','औ','ं','ः','क','ख','ग','घ','ङ','च','छ','ज','झ','ञ','ट','ठ','ड','ढ','ण','ड़','ढ़','त','थ','द','ध','न','प','फ','ब','भ','म','य','र','ल','व','श','ष','स','ह','क्ष','त्र','ज्ञ','श्र','ऋ',' ा',' ि',' ी','◌ु','◌ू','◌े','◌ै',' ो',' ौ','०','१','२','३','४','५','६','७','८','९','।', '॥','◌़','ऽ','ॐ','◌॒','◌॑','◌᳚','ऌ','ॠ','ॡ','◌ॢ','◌ॣ','◌ृ','◌ॄ','◌्','◌ऀ','ँ','◌ॅ',' ॉ']

tokenizer = Tokenizer(BPE())

tokenizer.decoder = BPEDecoder()

custom_regex_pattern = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|\s[\p{L}\p{M}]+|\s\d+|[\p{L}\p{M}]+|\d+|[\p{P}।॥]+"""
tokenizer.pre_tokenizer = Split(pattern=Regex(custom_regex_pattern), behavior="isolated")

trainer = BpeTrainer(vocab_size=50304,
                     show_progress=True,
                     special_tokens=['<|endoftext|>'],
                     initial_alphabet=devanagari_chars+ascii_chars)

print("Starting tokenizer training...")

tokenizer.train(files=["data_1B.txt"], trainer=trainer)
print("Training complete.")

# Save the tokenizer
tokenizer.save("tokenizer.json")
print("Tokenizer saved to tokenizer.json")