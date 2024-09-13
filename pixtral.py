from vllm import LLM
from vllm.sampling_params import SamplingParams

def initialize_llm(model_name, gpu_memory_utilization=0.9):
    try:
        # Initialize the LLM with GPU memory settings if supported
        llm = LLM(model=model_name, tokenizer_mode="mistral", gpu_memory_utilization=gpu_memory_utilization)
        return llm
    except Exception as e:
        print(f"Failed to initialize LLM: {e}")
        raise

def run_chat(llm, messages, sampling_params):
    try:
        # Ensure 'chat' method is available in LLM and correctly used
        outputs = llm.chat(messages, sampling_params=sampling_params)
        return outputs
    except Exception as e:
        print(f"Error during chat: {e}")
        raise

def main():
    model_name = "mistralai/Pixtral-12B-2409"
    sampling_params = SamplingParams(max_tokens=8192)

    # Initialize the model
    llm = initialize_llm(model_name)

    # Define the prompt and image URL
    prompt = "Please transcribe the handwritten text in the image."
    image_url = "image2.png"

    # Prepare messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        },
    ]

    # Run the chat method and print the outputs
    try:
        outputs = run_chat(llm, messages, sampling_params)
        # Ensure outputs are in expected format
        print(outputs[0].outputs[0].text)
    except Exception as e:
        print(f"An error occurred while processing the chat: {e}")

if __name__ == "__main__":
    main()
