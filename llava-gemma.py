import requests
from PIL import Image
from transformers import (
  LlavaForConditionalGeneration,
  AutoTokenizer,
  AutoProcessor,
  CLIPImageProcessor
)
#In this repo, needed for version < 4.41.1
#from processing_llavagemma import LlavaGemmaProcessor
#processor = LlavaGemmaProcessor( tokenizer=AutoTokenizer.from_pretrained(checkpoint), image_processor=CLIPImageProcessor.from_pretrained(checkpoint))

checkpoint = "Intel/llava-gemma-2b"

# Load model
model = LlavaForConditionalGeneration.from_pretrained(checkpoint)
processor = AutoProcessor.from_pretrained(checkpoint)

# Prepare inputs
# Use gemma chat template
prompt = processor.tokenizer.apply_chat_template(
    [{'role': 'user', 'content': "<image>\nThe images here are handwritten letters sent from Belle Greene to Mr.Berenson. In these letters generally the first part of the letter (starting with date and 'dear,') is on the right side, and the following part on the left side. Please provide only the transcription of the letters, please stick to the real text of the image, pay attention to underlined words, uppercase and lowercase letters. Do not describe the image."}],
    tokenize=False,
    add_generation_prompt=True
)
url = "https://github.com/giorgiacrosilla/InternshipITatti/blob/main/letter1.jpg?raw=true"
image = Image.open(requests.get(url, stream=True).raw)
inputs = processor(text=prompt, images=image, return_tensors="pt")

# Generate
generate_ids = model.generate(**inputs, max_length=100)
output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(output)