import os
import torch
import torchvision.transforms as T
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import logging
from difflib import unified_diff

# Set up logging
logging.basicConfig(filename='transcription.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure CUDA memory is cleared
torch.cuda.empty_cache()

# Set CUDA memory allocator configuration
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# Define constants for image normalization
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    """Builds the image transformation pipeline."""
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform

def load_image(image_file, input_size=448):
    """Load and preprocess the image."""
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    pixel_values = transform(image).unsqueeze(0)  # Adding batch dimension
    return pixel_values

def print_diff(old_text, new_text):
    """Print the difference between two texts."""
    diff = unified_diff(old_text.splitlines(keepends=True), 
                        new_text.splitlines(keepends=True))
    print(''.join(diff))

# Load the model and tokenizer
path = 'OpenGVLab/InternVL2-8B'
model = AutoModel.from_pretrained(path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, use_flash_attn=True, trust_remote_code=True).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

# Define the input and output folder paths
input_folder = "IAMa_cropped"
output_folder = "transcriptions_IAM2_internvl"
refinement_folder1 = os.path.join(output_folder, "refinement1")
refinement_folder2 = os.path.join(output_folder, "refinement2")
final_transcription_folder = os.path.join(output_folder, "final_transcriptions")

for folder in [output_folder, refinement_folder1, refinement_folder2, final_transcription_folder]:
    os.makedirs(folder, exist_ok=True)

# Define a system prompt to guide the model's behavior
system_prompt = """You are an AI assistant specialized in transcribing handwritten text from images. Please follow these guidelines:
1. Examine the image carefully and identify all handwritten text.
2. Transcribe ONLY the handwritten text. Ignore any printed or machine-generated text in the image.
3. Maintain the original structure of the handwritten text, including line breaks and paragraphs.
4. Do not attempt to correct spelling or grammar in the handwritten text. Transcribe it exactly as written.
5. Do not describe the image or its contents.
6. Do not introduce or contextualize the transcription.
Please begin your response directly with the transcribed text. Remember, your goal is to provide an accurate transcription of ONLY the handwritten portions of the text, preserving its original form as much as possible."""

# Set generation parameters
generation_config = dict(max_new_tokens=1024, do_sample=True)

# Iterate through each image in the input folder
for image_file in os.listdir(input_folder):
    if image_file.endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
        image_path = os.path.join(input_folder, image_file)
        image_name = os.path.splitext(image_file)[0]
        
        # Define the output file path for transcription
        output_file_path = os.path.join(final_transcription_folder, f"{image_name}.txt")

        # Check if the transcription file already exists
        if os.path.exists(output_file_path):
            print(f"Transcription for {image_file} already exists, skipping...")
            continue  # Skip this image if transcription exists

        try:
            # Load and preprocess the image
            pixel_values = load_image(image_path).to(torch.bfloat16).cuda()

            # Define the task-specific prompt
            question = 'Please transcribe as accurately as possible only the handwritten portions of the provided image.'

            # Combine the system prompt with the specific task question
            full_prompt = system_prompt + "\n" + question

            # Transcribe the image using the model (initial transcription)
            response = model.chat(tokenizer, pixel_values, full_prompt, generation_config)
            initial_transcription = response.strip()
            
            print("\nInitial transcription:")
            print(initial_transcription)
            logging.info(f"Initial transcription for {image_name} completed")

            # Save initial transcription in refinement_folder1
            initial_file_path = os.path.join(refinement_folder1, f"{image_name}_initial.txt")
            with open(initial_file_path, "w", encoding='utf-8') as f:
                f.write(initial_transcription)

            # Refinement loop
            refined_transcription = initial_transcription
            refinement_prompts = [
                "Review the original image and your previous transcription. Focus on correcting any spelling errors, punctuation mistakes, or missed words. Ensure the transcription accurately reflects the handwritten text.",
                "Examine the structure of the transcription. Are paragraphs and line breaks correctly represented? Adjust the layout to match the original handwritten text more closely.",
                "Make a final pass over the transcription, comparing it closely with the original image. Make any last corrections or improvements to ensure the highest possible accuracy. Delete any introduction or contextualization you might have added to the transcribed text.",
            ]

            for i, prompt in enumerate(refinement_prompts):
                full_prompt = system_prompt + "\n" + prompt + "\n\nPrevious transcription:\n" + refined_transcription

                response = model.chat(tokenizer, pixel_values, full_prompt, generation_config)
                new_transcription = response.strip()
                
                print(f"\nRefinement step {i+1}:")
                print_diff(refined_transcription, new_transcription)
                
                if new_transcription != refined_transcription:
                    logging.info(f"Refinement step {i+1} for {image_name} - Changes made")
                    refined_transcription = new_transcription
                else:
                    logging.info(f"Refinement step {i+1} for {image_name} - No changes made")
                
                # Save each refinement step in the appropriate folder
                if i == 0:
                    refinement_file_path = os.path.join(refinement_folder1, f"{image_name}.txt")
                elif i == 1:
                    refinement_file_path = os.path.join(refinement_folder2, f"{image_name}.txt")
                else:
                    refinement_file_path = os.path.join(final_transcription_folder, f"{image_name}.txt")
                
                with open(refinement_file_path, "w", encoding='utf-8') as f:
                    f.write(refined_transcription)

            print("\nFinal transcription:")
            print(refined_transcription)
            logging.info(f"Final refined text transcription saved for {image_name}")

        except Exception as e:
            logging.error(f"Error processing image {image_name}: {str(e)}")
            print(f"Error processing image {image_name}. Check the log file for details.")

print("All images processed.")