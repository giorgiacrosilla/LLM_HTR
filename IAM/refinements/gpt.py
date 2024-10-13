import os
import base64
import difflib
from openai import OpenAI

# Initialize the OpenAI client with your API key
client = OpenAI(api_key='')

# Define folder paths
image_folder = "IAM/IAMa_cropped"
transcription_txt_folder = "transcriptions_IAM2_gpt4mini"
refinement_folder1 = "transcriptions_IAM2_gpt4mini/refinement_step1"
refinement_folder2 = "transcriptions_IAM2_gpt4mini/refinement_step2"
final_transcription_folder = "transcriptions_IAM2_gpt4mini/final_transcriptions"

# Create necessary folders
for folder in [transcription_txt_folder, refinement_folder1, refinement_folder2, final_transcription_folder]:
    os.makedirs(folder, exist_ok=True)

# Transcription prompts
initial_prompt = """Please transcribe the handwritten text in this image accurately, respecting line breaks. Do not describe any fields or layout elements, focus solely on the handwritten content."""

system_prompt = """Please follow these guidelines:
1. Examine the image carefully and identify all handwritten text.
2. Transcribe ONLY the handwritten text. Ignore any printed or machine-generated text in the image.
3. Maintain the original structure of the handwritten text, including line breaks and paragraphs.
4. Do not attempt to correct spelling or grammar in the handwritten text. Transcribe it exactly as written.
Please begin your response directly with the transcribed text. Remember, your goal is to provide an accurate transcription of ONLY the handwritten portions of the text, preserving its original form as much as possible."""

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def process_image(image_path):
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    txt_file_path = os.path.join(transcription_txt_folder, f"{image_name}.txt")

    if os.path.exists(txt_file_path):
        print(f"Transcription for {image_name} already exists, skipping...")
        return

    base64_image = encode_image(image_path)

    initial_completion = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": initial_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        max_tokens=1000,
    )

    initial_transcription = initial_completion.choices[0].message.content
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
        "Make a final pass over the transcription, comparing it closely with the original image. Make any last corrections or improvements to ensure the highest possible accuracy. Delete any introduction or contextualization you might have added to the transcribed text .",
    ]

    for i, prompt in enumerate(refinement_prompts):
        refinement_response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                        {"type": "text", "text": "Please transcribe the handwritten text in this image accurately, respecting line breaks."}
                    ]
                },
                {"role": "assistant", "content": refined_transcription},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
        )

        new_transcription = refinement_response.choices[0].message.content
        
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

def main():
    image_count = 0  

    for filename in os.listdir(image_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(image_folder, filename)
            process_image(image_path)
            image_count += 1 
            
    print(f"Processed {image_count} images.")

if __name__ == "__main__":
    main()
