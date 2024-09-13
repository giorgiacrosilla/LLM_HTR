from vllm import LLM
from vllm.sampling_params import SamplingParams

model_name = "mistralai/Pixtral-12B-2409"

sampling_params = SamplingParams(max_tokens=8192)

llm = LLM(model=model_name, tokenizer_mode="mistral")

prompt = "Please transcribe the handwritten text in the image."
image_url = "image2.png"

messages = [
    {
        "role": "user",
        "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": image_url}}]
    },
]

outputs = vllm_model.model.chat(messages, sampling_params=sampling_params)

print(outputs[0].outputs[0].text)