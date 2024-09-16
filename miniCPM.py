import os
from PIL import Image
import torch
from transformers import AutoModel, AutoTokenizer

# Initialize the model and tokenizer
model = AutoModel.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True,
    attn_implementation='flash_attention_2', torch_dtype=torch.bfloat16)
model = model.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True)

def transcribe_image(image_path):
    """Transcribe text from an image using the model."""
    image = Image.open(image_path).convert('RGB')
    question = 'Please consider only the handwritten portion of the image and transcribe it as accurately as possible. Respect end of line and start of new lines.Do not describe the fields of the image("body text" or "signature"), stick only to the text as it is.'
    msgs = [{'role': 'user', 'content': [image, question]}]

    # Get the model response
    res = model.chat(
        image=None,
        msgs=msgs,
        tokenizer=tokenizer,
        sampling=False,
        stream=False
    )

    # Collect the generated text
    generated_text = ""
    for new_text in res:
        generated_text += new_text
        print(new_text, flush=True, end='')

    return generated_text

def process_images(input_folder, output_folder):
    """Process all images in the input folder and save transcriptions to the output folder."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            transcription = transcribe_image(image_path)
            
            # Save the transcription to a file
            transcription_filename = os.path.splitext(filename)[0] + '.txt'
            transcription_path = os.path.join(output_folder, transcription_filename)
            
            with open(transcription_path, 'w') as file:
                file.write(transcription)
            print(f"Transcription saved to {transcription_path}")

# Define input and output folders
input_folder = 'formsA-D_cropped'
output_folder = 'transcriptions_IAM_minicpm'

# Process images
process_images(input_folder, output_folder)

