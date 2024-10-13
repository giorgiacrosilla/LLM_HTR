import os
from vllm import LLM
from vllm.sampling_params import SamplingParams
import base64

# Initialize model and sampling parameters
model_name = "mistralai/Pixtral-12B-2409"
sampling_params = SamplingParams(max_tokens=8192, temperature= 0)

# Initialize the LLM
llm = LLM(model=model_name, tokenizer_mode="mistral", max_model_len=16500)

# Define input and output folders
input_folder = "aachen_validation_set" 
output_folder = "transcriptions_IAM_pixtral" 
os.makedirs(output_folder, exist_ok=True)

# Define the system and user prompts
system_prompt = """You are an AI assistant specialized in transcribing handwritten text from images. Please follow these guidelines:
1. Examine the image carefully and identify all handwritten text.
2. Transcribe ONLY the handwritten text.
3. Maintain the original structure of the handwritten text, including line breaks and paragraphs.
4. Do not attempt to correct spelling or grammar in the handwritten text. Transcribe it exactly as written.
5. Do not describe the image or its contents.
6. Do not introduce or contextualize the transcription.
Please begin your response directly with the transcribed text. Remember, your goal is to provide an accurate transcription of ONLY the handwritten portions of the text, preserving its original form as much as possible.
"""

user_prompt = "Please transcribe the handwritten text in this image as accurately as possible, respecting line breaks. Do not describe the fields of the image('body text' or 'signature'), stick only to the text as it is."

# Function to convert image to base64-encoded string
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded_string}"

def process_image(image_path, output_folder):
    # Get the image file name without extension
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    # Convert image to base64-encoded format
    image_base64 = image_to_base64(image_path)

    # Prepare the message with system prompt and user prompt
    messages = [
        {"role": "system", "content": system_prompt},  
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": image_base64}}  # Image as base64-encoded string
            ]
        }
    ]

    # Generate the transcription using the model
    outputs = llm.chat(messages, sampling_params=sampling_params)

    # Extract transcription from the model output
    transcription = outputs[0].outputs[0].text

    # Save the transcription to a text file in the output folder
    output_file_path = os.path.join(output_folder, f"{image_name}.txt")
    with open(output_file_path, "w") as f:
        f.write(transcription)

    print(f"Processed {image_path}, transcription saved to {output_file_path}")

    # Generate the transcription using the model
    outputs = llm.chat(messages, sampling_params=sampling_params)

    # Extract transcription from the model output
    transcription = outputs[0].outputs[0].text

    # Save the transcription to a text file in the output folder
    output_file_path = os.path.join(output_folder, f"{image_name}.txt")
    with open(output_file_path, "w") as f:
        f.write(transcription)

    print(f"Processed {image_path}, transcription saved to {output_file_path}")

# Main function to iterate over all images in the input folder
def main():
    # Loop through all image files in the input folder
    for image_file in os.listdir(input_folder):
        if image_file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):  # Supported image formats
            image_path = os.path.join(input_folder, image_file)
            process_image(image_path, output_folder)

if __name__ == "__main__":
    main()
