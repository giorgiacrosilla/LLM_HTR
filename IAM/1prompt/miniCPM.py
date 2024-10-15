import os
from PIL import Image
import torch
from transformers import AutoModel, AutoTokenizer

# Initialize the model and tokenizer
model = AutoModel.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True,
    attn_implementation='flash_attention_2', torch_dtype=torch.bfloat16)
model = model.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True)

def get_model_response(image, prompt, previous_msgs=[]):
    """Get a response from the model."""
    msgs = previous_msgs + [{'role': 'user', 'content': prompt}]
    res = model.chat(
        image=image,
        msgs=msgs,
        tokenizer=tokenizer,
        sampling=False,
        stream=False
    )
    generated_text = "".join(res)
    return generated_text

def transcribe_image(image_path):
    """Transcribe text from an image."""
    image = Image.open(image_path).convert('RGB')
    
    system_prompt = """You are an AI assistant specialized in transcribing handwritten text from images. Please follow these guidelines:
1. Examine the image carefully and identify all handwritten text.
2. Transcribe ONLY the handwritten text. Ignore any printed or machine-generated text in the image.
3. Maintain the original structure of the handwritten text, including line breaks and paragraphs.
4. Do not attempt to correct spelling or grammar in the handwritten text. Transcribe it exactly as written.
5. Do not describe the image or its contents.
6. Do not introduce or contextualize the transcription.
Please begin your response directly with the transcribed text. Remember, your goal is to provide an accurate transcription of ONLY the handwritten portions of the text, preserving its original form as much as possible."""

    transcription_prompt = f"""{system_prompt}\n\nPlease transcribe only the handwritten text in this image accurately. Please follow these guidelines:
             - In the image you can find both printed and handwritten text. The handwritten portion of the text is below the printed one.
             - Exclude from the transcription the signature part containing the word 'name:' and any text that follows it.
             - Maintain the original line breaks and formatting exactly as seen in the image.
             - Output ONLY the handwritten text without any additional comments, explanations.
             - Output ONLY the handwritten text without describing the fields of the image."""
    
    # Get transcription
    transcription = get_model_response(image, transcription_prompt)

    print(f"Processing image: {image_path}")
    print("Transcription:")
    print(transcription)

    return transcription

def process_images_in_folder(input_folder, output_folder):
    """Process all images in the input folder and save transcriptions to the output folder."""
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            print(f"Processing image: {filename}")

            transcription = transcribe_image(image_path)

            # Save transcription to output folder
            output_file_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.txt")
            with open(output_file_path, "w", encoding='utf-8') as f:
                f.write(transcription)

            print(f"Transcription saved to: {output_file_path}")
            print(f"Transcription process completed for: {filename}\n" + "=" * 50 + "\n")

# Specify the input and output folder paths
input_folder_path = 'aachen_validation_set'
output_folder_path = 'transcription_IAM1_minicpm'

# Process images in the specified folder
process_images_in_folder(input_folder_path, output_folder_path)