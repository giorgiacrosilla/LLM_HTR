from openai import OpenAI
import os
import base64

# Initialize the OpenAI client with your API key
client = OpenAI(api_key='')


image_folder = "IAM/aachen_validation_set"
transcription_txt_folder = "transcriptions_IAM_gpt4"


os.makedirs(transcription_txt_folder, exist_ok=True)

# Transcription prompts
initial_prompt = """ Please transcribe the handwritten text in this image as accurately as possible, respecting line breaks."""

system_prompt = """"You are an AI assistant specialized in transcribing handwritten text from images. Please follow these guidelines:
1. Examine the image carefully and identify all handwritten text.
2. Transcribe ONLY the handwritten text.
3. Maintain the original structure of the handwritten text, including line breaks and paragraphs.
4. Do not attempt to correct spelling or grammar in the handwritten text. Transcribe it exactly as written.
5. Do not describe the image or its contents.
6. Do not introduce or contextualize the transcription.
Please begin your response directly with the transcribed text. Remember, your goal is to provide an accurate transcription of ONLY the handwritten portions of the text, preserving its original form as much as possible."
"""

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
        model="gpt-4o",
        temperature=0,  # Setting the temperature to 0 for deterministic results
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


    transcription = initial_completion.choices[0].message.content


    with open(txt_file_path, "w", encoding='utf-8') as f:
        f.write(transcription)
    
    print(f"Transcription for {image_name} saved to {txt_file_path}")


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
