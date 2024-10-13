import os
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import difflib

# Load the model and processor
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

# Create necessary directories
input_folder = "IAMa_cropped"
transcription_txt_folder = "transcription_IAM2_qwen"
refinement_folder1 = "transcription_IAM2_qwen/refinement_folder1"
refinement_folder2 = "transcription_IAM2_qwen/refinement_folder2"
final_transcription_folder = "transcription_IAM2_qwen/final_transcription_folder"
os.makedirs(transcription_txt_folder, exist_ok=True)
os.makedirs(refinement_folder1, exist_ok=True)
os.makedirs(refinement_folder2, exist_ok=True)
os.makedirs(final_transcription_folder, exist_ok=True)

# System prompt: General guidelines for the model's behavior
system_prompt = """You are an AI assistant specialized in transcribing handwritten text from images. Please follow these guidelines:
1. Examine the image carefully and identify all handwritten text.
2. Transcribe ONLY the handwritten text. Ignore any printed or machine-generated text in the image.
3. Maintain the original structure of the handwritten text, including line breaks and paragraphs.
4. Do not attempt to correct spelling or grammar in the handwritten text. Transcribe it exactly as written.
5. Do not describe the image or its contents.
6. Do not introduce or contextualize the transcription.
Please begin your response directly with the transcribed text. Remember, your goal is to provide an accurate transcription of ONLY the handwritten portions of the text, preserving its original form as much as possible."""

user_prompt = "Please transcribe the handwritten text in this image as accurately as possible, respecting line breaks. Do not describe the fields of the image(i.e.'body text' or 'signature'), stick only to the text as it is."


def generate_transcription(image, message):
    """Helper function to generate a transcription or refinement."""
    # Prepare inputs for model inference
    text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(message)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to("cuda")

    # Generate transcription or refinement
    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    transcription = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    return transcription


def process_image(image_path, transcription_txt_folder):
    # Get the image file name
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Check if transcription already exists
    txt_file_path = os.path.join(transcription_txt_folder, f"{image_name}.txt")
    if os.path.exists(txt_file_path):
        print(f"Transcription already exists for {image_name}. Skipping.")
        return False

    # Load the image
    image = Image.open(image_path)

    # Combine system and user prompts for the first transcription
    message = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": user_prompt}
        ]}
    ]

    # First step: Generate initial transcription
    initial_transcription = generate_transcription(image, message)
    print(f"Processing image: {image_name}")
    print("Initial transcription:")
    print(initial_transcription)
    print("\n" + "="*50 + "\n")

    # Save initial transcription in refinement_folder1
    initial_file_path = os.path.join(refinement_folder1, f"{image_name}_initial.txt")
    with open(initial_file_path, "w", encoding='utf-8') as f:
        f.write(initial_transcription)

    # Refinement loop
    refined_transcription = initial_transcription
    refinement_prompts = [
        "Review the original image and your previous transcription. Focus on correcting any spelling errors, punctuation mistakes, or missed words. Ensure the transcription accurately reflects the handwritten text.",
        "Examine the structure of the transcription. Are paragraphs and line breaks correctly represented? Adjust the layout to match the original handwritten text more closely.",
        "Make a final pass over the transcription, comparing it closely with the original image. Make any last corrections or improvements to ensure the highest possible accuracy. Delete any introduction or contextualization you might have added to the transcribed text."
    ]

    for i, prompt in enumerate(refinement_prompts):
        # In each refinement step, pass the image and updated transcription for further review
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "assistant", "content": refined_transcription},  # Previous transcription
            {"role": "user", "content": [
                {"type": "image", "image": image},  # Include image here
                {"type": "text", "text": prompt}    # Prompt for refinement
            ]}
        ]
        
        # Generate new transcription using the image and updated message
        new_transcription = generate_transcription(image, messages)
        
        # Compare the new transcription with the previous one
        diff = list(difflib.unified_diff(refined_transcription.splitlines(), new_transcription.splitlines(), lineterm=''))
        
        if diff:
            print(f"Refinement step {i+1} - Changes made:")
            for line in diff:
                print(line)
            refined_transcription = new_transcription
        else:
            print(f"Refinement step {i+1} - No changes made.")
        
        # Save each refinement step in the appropriate folder
        if i == 0:
            refinement_file_path = os.path.join(refinement_folder1, f"{image_name}.txt")
        elif i == 1:
            refinement_file_path = os.path.join(refinement_folder2, f"{image_name}.txt")
        else:
            refinement_file_path = os.path.join(transcription_txt_folder, f"{image_name}.txt")
        
        with open(refinement_file_path, "w", encoding='utf-8') as f:
            f.write(refined_transcription)
        
        print("\n" + "="*50 + "\n")

    # Save final refined transcription to final_transcription_folder
    final_file_path = os.path.join(final_transcription_folder, f"{image_name}.txt")
    with open(final_file_path, "w", encoding='utf-8') as f:
        f.write(refined_transcription)

    print(f"Final refined text transcription saved to {final_file_path}")


if __name__ == "__main__":
    for image_file in os.listdir(input_folder):
        image_path = os.path.join(input_folder, image_file)
        if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            process_image(image_path, transcription_txt_folder)
