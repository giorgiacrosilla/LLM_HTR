import os
import requests
from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoTokenizer, AutoProcessor, CLIPImageProcessor

# Load the model and processor
checkpoint = "Intel/llava-gemma-2b"
model = LlavaForConditionalGeneration.from_pretrained(checkpoint)
processor = AutoProcessor.from_pretrained(checkpoint)

# Input and output folder paths
input_folder = "IAM/formsA-D_cropped"  # Folder containing the images
output_folder = "transcriptions_IAM_llava"  # Folder where transcriptions will be saved

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Prepare the template for transcription
prompt_template = processor.tokenizer.apply_chat_template(
    [{'role': 'user', 'content': "<image>\nPlease consider only the handwritten portion of the image and transcribe it as accurately as possible. Respect end of line and start of new lines.Do not describe the fields of the image(""body text"" or ""signature""), stick only to the text as it is."}],
    tokenize=False,
    add_generation_prompt=True
)

# Iterate over all images in the input folder
for image_file in os.listdir(input_folder):
    if image_file.endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):  # Supported image formats
        image_path = os.path.join(input_folder, image_file)

        # Load the image
        image = Image.open(image_path)

        # Prepare inputs
        inputs = processor(text=prompt_template, images=image, return_tensors="pt")

        # Generate the transcription
        generate_ids = model.generate(**inputs, max_length=5000)
        output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        # Save the transcription to a text file in the output folder
        output_file_path = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}.txt")
        with open(output_file_path, "w") as f:
            f.write(output)

        print(f"Processed {image_file}, transcription saved to {output_file_path}")
