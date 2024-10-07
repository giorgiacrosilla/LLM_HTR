import os
from PIL import Image
import requests
from transformers import AutoModelForCausalLM, AutoProcessor

# Model setup
model_id = "microsoft/Phi-3-vision-128k-instruct"
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda", trust_remote_code=True, torch_dtype="auto", _attn_implementation='flash_attention_2')
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

# Input and output folder paths
input_folder = "IAM/IAMa_cropped"  
output_folder = "transcriptions_IAM3_phi" 

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Generation arguments
generation_args = {
    "max_new_tokens": 500,
    "temperature": 0.0,
    "do_sample": False,
}

# System prompt
system_prompt = "You are an expert in transcribing handwritten text. Your task is to accurately transcribe the handwritten content in the provided image, respecting line breaks and without describing any fields or layout elements."

# Iterate over images in the input folder
for image_file in os.listdir(input_folder):
    if image_file.endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):  
        image_path = os.path.join(input_folder, image_file)

        # Load the image
        image = Image.open(image_path)

        # Create messages for the prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "<|image_1|>\nPlease transcribe the handwritten text in this image accurately, respecting line breaks. Do not describe the fields of the image('body text' or 'signature'), stick only to the text as it is."}
        ]

        # Generate the prompt with the image
        prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Process the inputs
        inputs = processor(prompt, [image], return_tensors="pt").to("cuda:0")

        # Generate transcription
        generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args)

        # Remove input tokens
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]

        # Decode the response
        response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        # Save the transcription to a text file
        output_file_path = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}.txt")
        with open(output_file_path, "w") as f:
            f.write(response)

        print(f"Processed {image_file}, transcription saved to {output_file_path}")