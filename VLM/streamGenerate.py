import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'
# import torch
from models import load_model
from PIL import Image
from data_utils import convert_chat_prompt, TTFTTextStreamer
from utils import Colors

tokenizer, processor, model = load_model('/root/LLM4Beginner/ckpts/VLMFinetune')

print(Colors.GREEN + "This chat is without history context. (Ctrl + C to exit)" + Colors.RESET)

while True:
    prompt = input(Colors.GREEN + 'Text Input >>> ' + Colors.RESET)
    image_path = input(Colors.GREEN + 'Image Path List (Split with | and Enter for None) >>> ' + Colors.RESET).replace("'", "").replace('"', '')
    try:
        image = [Image.open(i.strip()).convert('RGB') for i in image_path.split('|')]
        print(f'Reveived {len(image)} Images.')
    except:
        image = None
        print('Receive 0 Images.')
    inputs = convert_chat_prompt(prompt, image, tokenizer, processor, model.config)
    streamer = TTFTTextStreamer(tokenizer, skip_prompt=True)
    model.generate(**inputs, streamer=streamer, max_new_tokens = 500)
    print()
