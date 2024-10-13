import os
import base64
import difflib
import logging
from vllm import LLM
from vllm.sampling_params import SamplingParams

# Set up logging
logging.basicConfig(filename='transcription_log.txt', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize model and sampling parameters
model_name = "mistralai/Pixtral-12B-2409"
sampling_params = SamplingParams(max_tokens=8192)

# Initialize the LLM
try:
    llm = LLM(model=model_name, tokenizer_mode="mistral", max_model_len=16500)
except Exception as e:
    logging.error(f"Error initializing LLM: {str(e)}")
    raise

# Define input and output folders
input_folder = "IAMa_cropped"  # Folder containing the images
output_folder = "transcriptions_IAM3_pixtral"  # Main output folder
os.makedirs(output_folder, exist_ok=True)

# Create subfolders for different stages of transcription
refinement_folder1 = os.path.join(output_folder, "refinement_1")
refinement_folder2 = os.path.join(output_folder, "refinement_2")
transcription_txt_folder = os.path.join(output_folder, "transcription_txt")
final_transcription_folder = os.path.join(output_folder, "final_transcriptions")

for folder in [refinement_folder1, refinement_folder2, transcription_txt_folder, final_transcription_folder]:
    os.makedirs(folder, exist_ok=True)

# Define the system and user prompts
system_prompt = """You are an AI assistant specialized in transcribing handwritten text from images. Please follow these guidelines:
1. Examine the image carefully and identify all handwritten text.
2. Transcribe ONLY the handwritten text. Ignore any printed or machine-generated text in the image.
3. Maintain the original structure of the handwritten text, including line breaks and paragraphs.
4. Do not attempt to correct spelling or grammar in the handwritten text. Transcribe it exactly as written.
5. Do not describe the image or its contents.
6. Do not introduce or contextualize the transcription.
Please begin your response directly with the transcribed text. Remember, your goal is to provide an accurate transcription of ONLY the handwritten portions of the text, preserving its original form as much as possible."""

user_prompt = "Please transcribe the handwritten text in this image as accurately as possible, respecting line breaks. "

# Function to convert image to base64-encoded string
def image_to_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded_string}"
    except Exception as e:
        logging.error(f"Error encoding image {image_path}: {str(e)}")
        raise

def print_diff(old, new):
    diff = list(difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm=''))
    for line in diff:
        if line.startswith('+'):
            print(f"\033[92m{line}\033[0m")  # Green for additions
        elif line.startswith('-'):
            print(f"\033[91m{line}\033[0m")  # Red for deletions
        else:
            print(line)

def process_image(image_path):
    # Get the image file name without extension
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    print(f"\nProcessing image: {image_name}")
    logging.info(f"Processing image: {image_name}")

    try:
        # Convert image to base64-encoded format
        image_base64 = image_to_base64(image_path)

        # Prepare the message with system prompt and user prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": image_base64}}
                ]
            }
        ]

        # Generate the initial transcription using the model
        outputs = llm.chat(messages, sampling_params=sampling_params)
        initial_transcription = outputs[0].outputs[0].text
        
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
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_base64}},
                        {"type": "text", "text": "Please transcribe the handwritten text in this image accurately, respecting line breaks."}
                    ]
                },
                {"role": "assistant", "content": refined_transcription},
                {"role": "user", "content": prompt}
            ]

            outputs = llm.chat(messages, sampling_params=sampling_params)
            new_transcription = outputs[0].outputs[0].text
            
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
                refinement_file_path = os.path.join(transcription_txt_folder, f"{image_name}.txt")
            
            with open(refinement_file_path, "w", encoding='utf-8') as f:
                f.write(refined_transcription)

        # Save final refined transcription to final_transcription_folder
        final_file_path = os.path.join(final_transcription_folder, f"{image_name}.txt")
        with open(final_file_path, "w", encoding='utf-8') as f:
            f.write(refined_transcription)

        print("\nFinal transcription:")
        print(refined_transcription)
        logging.info(f"Final refined text transcription saved for {image_name}")

    except Exception as e:
        logging.error(f"Error processing image {image_name}: {str(e)}")
        print(f"Error processing image {image_name}. Check the log file for details.")

# Main function to iterate over all images in the input folder
def main():
    logging.info("Starting transcription process")
    print("Starting transcription process")
    # Loop through all image files in the input folder
    for image_file in os.listdir(input_folder):
        if image_file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):  # Supported image formats
            image_path = os.path.join(input_folder, image_file)
            process_image(image_path)
    logging.info("Transcription process completed")
    print("Transcription process completed")

if __name__ == "__main__":
    main()