import os
import torch
import torchvision.transforms as T
from PIL import Image
from transformers import AutoModel, AutoTokenizer

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

def load_image(image_file, input_size=448, max_num=12):
    """Load and preprocess the image."""
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    pixel_values = transform(image).unsqueeze(0)  # Adding batch dimension
    return pixel_values

# Load the model and tokenizer
path = 'OpenGVLab/InternVL2-8B'
model = AutoModel.from_pretrained(path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, use_flash_attn=True, trust_remote_code=True).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

# Define the input and output folder paths
input_folder = "IAMa_cropped"  
output_folder = "transcriptions_IAM3_internvl"  
os.makedirs(output_folder, exist_ok=True)

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
    if image_file.endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):  # Check for supported image formats
        image_path = os.path.join(input_folder, image_file)
        
        # Define the output file path for transcription
        output_file_path = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}.txt")

        # Check if the transcription file already exists
        if os.path.exists(output_file_path):
            print(f"Transcription for {image_file} already exists, skipping...")
            continue  # Skip this image if transcription exists

        # Load and preprocess the image
        pixel_values = load_image(image_path).to(torch.bfloat16).cuda()

        question = 'Please transcribe as accurately as possible only the handwritten portions of the provided image, respecting line breaks.'

        # Combine the system prompt with the specific task question
        full_prompt = system_prompt + "\n" + question

        response = model.chat(tokenizer, pixel_values, full_prompt, generation_config)
        transcription = response.strip()

        with open(output_file_path, "w") as f:
            f.write(transcription)

        print(f"Processed {image_file}, transcription saved to {output_file_path}")

print("All images processed.")
