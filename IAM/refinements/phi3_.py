import os
import difflib
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

# Model setup
model_id = "microsoft/Phi-3-vision-128k-instruct"
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda", trust_remote_code=True, torch_dtype="auto")
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

# Input and output folder paths
input_folder = "IAMa_cropped"
output_folder = "transcriptions_IAM2_phi"
refinement_folders = [
    os.path.join(output_folder, "refinement_step1"),
    os.path.join(output_folder, "refinement_step2"),
    os.path.join(output_folder, "refinement_step3")
]
final_transcription_folder = os.path.join(output_folder, "final_transcriptions")

# Ensure output folders exist
os.makedirs(output_folder, exist_ok=True)
for folder in refinement_folders:
    os.makedirs(folder, exist_ok=True)
os.makedirs(final_transcription_folder, exist_ok=True)

# Generation arguments
generation_args = {
    "max_new_tokens": 500,
    "temperature": 0.0,
    "do_sample": False,
}

# System prompt
system_prompt = """You are an AI assistant specialized in transcribing handwritten text from images. Please follow these guidelines:
1. Examine the image carefully and identify all handwritten text.
2. Transcribe ONLY the handwritten text. Ignore any printed or machine-generated text in the image.
3. Maintain the original structure of the handwritten text, including line breaks and paragraphs.
4. Do not attempt to correct spelling or grammar in the handwritten text. Transcribe it exactly as written.
5. Do not describe the image or its contents.
6. Do not introduce or contextualize the transcription.
Please begin your response directly with the transcribed text. Remember, your goal is to provide an accurate transcription of ONLY the handwritten portions of the text, preserving its original form as much as possible."""

def generate_transcription(image, messages):
    prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(prompt, [image], return_tensors="pt").to("cuda:0")
    generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args)
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    return processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

def show_differences(old_text, new_text):
    differ = difflib.Differ()
    diff = list(differ.compare(old_text.splitlines(), new_text.splitlines()))
    return '\n'.join(diff)

# Refinement prompts
refinement_prompts = [
    "Review the original image and your previous transcription. Focus on correcting any spelling errors, punctuation mistakes, or missed words. Ensure the transcription accurately reflects the handwritten text. Provide only the refined transcription.",
    "Examine the structure of the transcription. Are paragraphs and line breaks correctly represented? Adjust the layout to match the original handwritten text more closely. Provide only the refined transcription.",
    "Make a final pass over the transcription, comparing it closely with the original image. Make any last corrections or improvements to ensure the highest possible accuracy. Provide only the final refined transcription."
]

# Iterate over images in the input folder
for image_file in os.listdir(input_folder):
    if image_file.endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
        image_path = os.path.join(input_folder, image_file)
        image = Image.open(image_path)

        print(f"Processing image: {image_file}")

        # Initial transcription
        initial_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "<|image_1|>\nPlease transcribe the handwritten text in this image accurately, respecting line breaks.B"}
        ]
        transcription = generate_transcription(image, initial_messages)

        print("Initial transcription:")
        print(transcription)
        print("\n" + "="*50 + "\n")

        # Save initial transcription
        initial_file_path = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}_initial.txt")
        with open(initial_file_path, "w", encoding='utf-8') as f:
            f.write(transcription)

        # Refinement steps
        previous_transcription = transcription
        for i, (refinement_prompt, refinement_folder) in enumerate(zip(refinement_prompts, refinement_folders), 1):
            refinement_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "assistant", "content": previous_transcription},
                {"role": "user", "content": f"<|image_1|>\n{refinement_prompt}"}
            ]
            transcription = generate_transcription(image, refinement_messages)

            print(f"Refinement step {i}:")
            print(transcription)
            print("\nDifferences:")
            print(show_differences(previous_transcription, transcription))
            print("\n" + "="*50 + "\n")

            # Save refined transcription
            refinement_file_path = os.path.join(refinement_folder, f"{os.path.splitext(image_file)[0]}.txt")
            with open(refinement_file_path, "w", encoding='utf-8') as f:
                f.write(transcription)

            previous_transcription = transcription

        # Save final refined transcription
        final_file_path = os.path.join(final_transcription_folder, f"{os.path.splitext(image_file)[0]}.txt")
        with open(final_file_path, "w", encoding='utf-8') as f:
            f.write(transcription)

        print(f"Final refined transcription saved to {final_file_path}")