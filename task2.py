# Import necessary libraries
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load the pre-trained GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Put model in evaluation mode
model.eval()

def generate_text(prompt, max_length=150, num_return_sequences=1):
    """
    Generate coherent paragraphs of text based on a user prompt.
    
    Args:
        prompt (str): Input text prompt.
        max_length (int): Maximum length of generated text.
        num_return_sequences (int): Number of generated sequences to return.
    
    Returns:
        List of generated text sequences.
    """
    # Encode input prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    # Generate text using the model
    output_sequences = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        temperature=0.7,        # creativity of generation
        top_k=50,               # limit sampling to top_k tokens
        top_p=0.95,             # nucleus sampling (probability mass)
        do_sample=True,         # enable sampling
        num_return_sequences=num_return_sequences,
        pad_token_id=tokenizer.eos_token_id  # pad token for GPT2
    )
    
    # Decode generated sequences
    generated_texts = []
    for generated_sequence in output_sequences:
        text = tokenizer.decode(generated_sequence, skip_special_tokens=True)
        generated_texts.append(text)
    
    return generated_texts

# Example usage: generate 1 paragraph for a given prompt
user_prompt = "Artificial intelligence is transforming the world by"
generated_paragraphs = generate_text(user_prompt, max_length=150, num_return_sequences=1)

print("User prompt:", user_prompt)
print("\nGenerated text:\n")
print(generated_paragraphs[0])