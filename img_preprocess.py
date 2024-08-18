from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os 
import torch
import numpy as np
import pickle

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

images = []
image_names = []
for thing in os.scandir('imgs'):
    if thing.is_file():
        data = Image.open(thing.path)
        images.append(data)
        image_names.append(thing.name)
inputs = processor(text=[''], images=images, return_tensors="pt", padding=True)
outputs = model(**inputs)
# print(outputs.image_embeds.shape)

image_embeds = outputs.image_embeds.detach()
# print(image_embeds)

labelled_embeds = {}

for nm, em in zip(image_names, image_embeds):
    labelled_embeds[nm] = em
# print(nm, labelled_embeds[nm].shape)
with open('img_embeds.pickle', 'wb') as f:
    pickle.dump(labelled_embeds, f)