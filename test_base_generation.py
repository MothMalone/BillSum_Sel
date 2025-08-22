#!/usr/bin/env python3
"""
Quick debug script to test base model generation
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_base_model_generation():
    """Test basic generation with the base model."""
    logger.info("Loading base model for generation test...")
    
    model_name = "meta-llama/Llama-2-7b-hf"
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Test simple generation
    test_text = "The bill proposes to increase funding for education."
    prompt = f"Summarize the following bill:\n\n{test_text}\n\nSummary:"
    
    logger.info(f"Testing with prompt: {prompt[:100]}...")
    
    inputs = tokenizer(
        prompt,
        max_length=400,
        truncation=True,
        return_tensors="pt"
    ).to(model.device)
    
    logger.info(f"Input tokens shape: {inputs['input_ids'].shape}")
    logger.info(f"Input text length: {len(prompt)}")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,  # Use greedy for more predictable output
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode full output
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info(f"Full output length: {len(full_output)}")
    logger.info(f"Full output: {full_output}")
    
    # Extract generated part
    generated_part = full_output[len(prompt):].strip()
    logger.info(f"Generated part length: {len(generated_part)}")
    logger.info(f"Generated part: '{generated_part}'")
    
    return generated_part

if __name__ == "__main__":
    test_base_model_generation()
