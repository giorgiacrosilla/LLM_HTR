import os
from vllm import LLM
from vllm.sampling_params import SamplingParams

# Initialize model and sampling parameters
model_name = "mistralai/Pixtral-12B-2409"
sampling_params = SamplingParams(max_tokens=8192)

# Initialize the LLM
llm = LLM(model=model_name, tokenizer_mode="mistral", max_model_len=16500)

# Define input and output folders
input_folder = "IAM/formsA-D_cropped"  # Folder containing the images
output_folder = "transcriptions_IAM_pixtral"  # Folder to save the transcriptions
os.makedirs(output_folder, exist_ok=True)

# Transcription prompt
prompt = "Please take in consideration only the handwritten portion of the image and transcribe it as accurately as possible. Answer with a transcription of the text only, respecting end of line and start of new lines."

# Function to process each image
def process_image(image_path, output_folder):
    # Get the image file name
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    # Prepare the input for the model
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"file://{image_path}"}},  # Use local image path
            ]
        }
    ]

    # Generate the transcription
    outputs = llm.chat(messages, sampling_params=sampling_params)

    # Extract transcription from output
    transcription = outputs[0].outputs[0].text

    # Save the transcription to a text file
    output_file_path = os.path.join(output_folder, f"{image_name}.txt")
    with open(output_file_path, "w") as f:
        f.write(transcription)

    print(f"Processed {image_path}, transcription saved to {output_file_path}")

# Main function to iterate over all images in the input folder
def main():
    # Iterate through all image files in the input folder
    for image_file in os.listdir(input_folder):
        if image_file.endswith((".png", ".jpg", ".jpeg", ".bmp")):  # Supported image formats
            image_path = os.path.join(input_folder, image_file)
            process_image(image_path, output_folder)

if __name__ == "__main__":
    main()
