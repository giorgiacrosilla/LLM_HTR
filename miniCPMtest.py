import torch
from PIL import Image
import requests
from io import BytesIO
from transformers import AutoModel, AutoTokenizer

# Carica il modello e il tokenizer
model = AutoModel.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True,
    attn_implementation='sdpa', torch_dtype=torch.bfloat16) # sdpa or flash_attention_2, no eager
model = model.eval().cuda()
tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True)

# URL dell'immagine
image_url = 'https://iiif.itatti.harvard.edu/iiif/2/bellegreene-full!32044150446383_001.jpg/full/full/0/default.jpg'

# Scarica l'immagine
response = requests.get(image_url)
image = Image.open(BytesIO(response.content)).convert('RGB')

# Definisci il messaggio
question = 'Please transcribe'
msgs = [{'role': 'user', 'content': [image, question]}]

# Esegui il modello
res = model.chat(
    image=None,
    msgs=msgs,
    tokenizer=tokenizer
)

## se desideri utilizzare lo streaming, assicurati che sampling=True e stream=True
## il model.chat restituirà un generatore
res = model.chat(
    image=None,
    msgs=msgs,
    tokenizer=tokenizer,
    sampling=False,
    stream=False
)

# Accumula e stampa il testo generato
generated_text = ""
for new_text in res:
    generated_text += new_text
    print(new_text, flush=True, end='')
