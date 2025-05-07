from tokenizers import Tokenizer
from transformers import GPT2Tokenizer

# Hindi and English text from a random Wikipedia page
hindi_text = """भूगोल (अंग्रेज़ी: Geography) शब्द दो शब्दों से मिलकर बना है- भू + गोल। यहाँ भू शब्द का तात्पर्य पृथ्वी और गोल शब्द का तात्पर्य उसके गोल आकार से है। यह एक विज्ञान है जिसके द्वारा पृथ्वी की सतह के स्वरुप और उसके प्राकृतिक विभागों (जैसे पहाड़, महाद्वीप, देश, नगर, नदी, समुद्र, झील, जल-संधियाँ, वन आदि) का ज्ञान होता है।[1]प्राकृतिक विज्ञानों के निष्कर्षों के बीच कार्य-कारण संबंध स्थापित करते हुए पृथ्वीतल की विभिन्नताओं का मानवीय दृष्टिकोण से अध्ययन ही भूगोल का सार तत्व है। पृथ्वी की सतह पर जो स्थान विशेष हैं उनकी समताओं तथा विषमताओं का कारण और उनका स्पष्टीकरण भूगोल का निजी क्षेत्र है। सर्वप्रथम प्राचीन यूनानी विद्वान"""
english_text = """Geography (from Ancient Greek γεωγραφία geōgraphía; combining gê 'Earth' and gráphō 'write', literally 'Earth writing') is the study of the lands, features, inhabitants, and phenomena of Earth.[1][2] Geography is an all-encompassing discipline that seeks an understanding of Earth and its human and natural complexities—not merely where objects are, but also how they have changed and come to be. While geography is specific to Earth, many concepts can be applied more broadly to other celestial bodies in the field of planetary science.[3] Geography has been called "a bridge between natural science and social science disciplines."[4] Origins of many of the concepts in geography can be traced to Greek"""

# Load the trained tokenizer
tokenizer = Tokenizer.from_file("tokenizer.json")

print("My BPE tokenizer:\n")

text_num_words = len(hindi_text.split())
encoding = tokenizer.encode(hindi_text)
token_count = len(encoding.ids)
hindi_token_per_word = token_count / text_num_words

print("  Hindi:")
print(f"    token count = {token_count}")
print(f"    word count = {text_num_words}")
print(f"    token/word = {hindi_token_per_word:.2f}\n")

text_num_words = len(english_text.split())
encoding = tokenizer.encode(english_text)
token_count = len(encoding.ids)
english_token_per_word = token_count / text_num_words

print("  English:")
print(f"    token count = {token_count}")
print(f"    word count = {text_num_words}")
print(f"    token/word = {english_token_per_word:.2f}\n")

print(f"Average token/word = {(hindi_token_per_word + english_token_per_word) / 2:.2f}\n")


# Load GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

print("OpenAI's GPT-2 BPE tokenizer:\n")

text_num_words = len(hindi_text.split())
tokens = tokenizer.tokenize(hindi_text)
ids = tokenizer.encode(hindi_text)
token_count = len(ids)
hindi_token_per_word = token_count / text_num_words

print("  Hindi:")
print(f"    token count = {token_count}")
print(f"    word count = {text_num_words}")
print(f"    token/word = {hindi_token_per_word:.2f}\n")

text_num_words = len(english_text.split())
tokens = tokenizer.tokenize(english_text)
ids = tokenizer.encode(english_text)
token_count = len(ids)
english_token_per_word = token_count / text_num_words

print("  English:")
print(f"    token count = {token_count}")
print(f"    word count = {text_num_words}")
print(f"    token/word = {english_token_per_word:.2f}\n")

print(f"Average token/word = {(hindi_token_per_word + english_token_per_word) / 2:.2f}")