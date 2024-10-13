import os
from PIL import Image
import torch
from transformers import AutoModel, AutoTokenizer
import difflib

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

def transcribe_and_refine_image(image_path):
    """Transcribe text from an image and perform 3 refinement steps."""
    image = Image.open(image_path).convert('RGB')
    
    system_prompt = """You are an AI assistant specialized in transcribing handwritten text from images. Please follow these guidelines:
1. Examine the image carefully and identify all handwritten text.
2. Transcribe ONLY the handwritten text. Ignore any printed or machine-generated text in the image.
3. Maintain the original structure of the handwritten text, including line breaks and paragraphs.
4. Do not attempt to correct spelling or grammar in the handwritten text. Transcribe it exactly as written.
5. Do not describe the image or its contents.
6. Do not introduce or contextualize the transcription.
Please begin your response directly with the transcribed text. Remember, your goal is to provide an accurate transcription of ONLY the handwritten portions of the text, preserving its original form as much as possible."""

    transcription_prompt = f"{system_prompt}\n\nPlease transcribe as accurately as possible only the handwritten portions of the provided image."
    
    # Get initial transcription
    initial_transcription = get_model_response(image, transcription_prompt)

    print(f"Processing image: {image_path}")
    print("Initial transcription:")
    print(initial_transcription)
    print("\n" + "="*50 + "\n")

    # Save initial transcription in refinement_folder1
    initial_file_path = os.path.join(refinement_folder1, f"{os.path.basename(image_path)}_initial.txt")
    os.makedirs(refinement_folder1, exist_ok=True)
    with open(initial_file_path, "w", encoding='utf-8') as f:
        f.write(initial_transcription)

    # Refinement loop
    refined_transcription = initial_transcription
    refinement_prompts = [
        f"{system_prompt}\n\nReview the original image and your previous transcription. Focus on correcting any spelling errors, punctuation mistakes, or missed words. Ensure the transcription accurately reflects the handwritten text.",
        f"{system_prompt}\n\nExamine the structure of the transcription. Are paragraphs and line breaks correctly represented? Adjust the layout to match the original handwritten text more closely.",
        f"{system_prompt}\n\nMake a final pass over the transcription, comparing it closely with the original image. Make any last corrections or improvements to ensure the highest possible accuracy. Delete any introduction or contextualization you might have added to the transcribed text."
    ]

    # Initialize the message list with only the initial transcription
    messages = [{"role": "user", "content": initial_transcription}]

    for i, prompt in enumerate(refinement_prompts):
        # Generate new transcription using the image and updated messages
        new_transcription = get_model_response(image, prompt, messages)

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
            refinement_file_path = os.path.join(refinement_folder1, f"{os.path.basename(image_path)}.txt")
        elif i == 1:
            refinement_file_path = os.path.join(refinement_folder2, f"{os.path.basename(image_path)}.txt")
        else:
            refinement_file_path = os.path.join(refinement_folder3, f"{os.path.basename(image_path)}.txt")
        
        os.makedirs(os.path.dirname(refinement_file_path), exist_ok=True)
        with open(refinement_file_path, "w", encoding='utf-8') as f:
            f.write(refined_transcription)
        
        print("\n" + "="*50 + "\n")

    # Save final refined transcription to final_transcription_folder
    final_file_path = os.path.join(final_transcription_folder, f"{os.path.basename(image_path)}.txt")
    os.makedirs(final_transcription_folder, exist_ok=True)
    with open(final_file_path, "w", encoding='utf-8') as f:
        f.write(refined_transcription)

    print(f"Final refined text transcription saved to {final_file_path}")

def process_images_in_folder(input_folder, output_folder):
    """Process all images in the input folder and save transcriptions to the output folder."""
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            print(f"Processing image: {filename}")

            transcribe_and_refine_image(image_path)

            print(f"Transcription process completed for: {filename}\n" + "=" * 50 + "\n")

# Specify the input and output folder paths
input_folder_path = 'IAMa_cropped'
output_folder_path = 'transcription_IAM2_minicpm'
refinement_folder1 = "transcription_IAM2_minicpm/refinement1"
refinement_folder2 = "transcription_IAM2_minicpm/refinement2"
refinement_folder3 = "transcription_IAM2_minicpm/refinement3"
final_transcription_folder = "transcription_IAM2_minicpm/final"

# Process images in the specified folder
process_images_in_folder(input_folder_path, output_folder_path)