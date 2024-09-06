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

# Preprocessa l'immagine (se necessario per il tuo modello)
# Questo potrebbe includere la conversione in tensore e il ridimensionamento
# Modifica questa parte secondo le necessità del tuo modello

# Definisci il messaggio
question = 'Please transcribe'
# Assicurati che il formato dei messaggi sia corretto
msgs = [{'role': 'user', 'content': question}]

# Esegui il modello
# Assicurati che il metodo `model.chat` sia corretto e compatibile con il tuo modello
# Se il modello non supporta `chat`, potrebbe essere necessario usare un altro metodo
try:
    res = model.chat(
        image=None,  # L'immagine potrebbe dover essere passata in un altro formato
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
    
except AttributeError:
    print("Il modello non supporta il metodo `chat`. Verifica la documentazione del modello.")

