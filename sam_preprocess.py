from PIL import Image
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import random

SAM2_DIR = './segment-anything-2/'
checkpoint = SAM2_DIR+"checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"

imgs_path = 'imgs_kitchenvideo'
masks_path = 'img_masks.pickle'

# Copied from the sam2 repository
def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask 
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) 

    ax.imshow(img)

def show_anns_and_img(image, masks):
    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    plt.show() 

def test_masking():
    with open(masks_path, 'rb') as f:
        segmented_scenes = pickle.load(f)
    first_img = [_ for _ in segmented_scenes][random.randrange(0, len(segmented_scenes))]
    show = segmented_scenes[first_img]
    show_anns_and_img(Image.open(imgs_path+'/'+first_img),show)
    
def main():
    segmented_scenes = {}
    for thing in os.scandir(imgs_path):
        if thing.is_file():
            generator = SAM2AutomaticMaskGenerator(build_sam2(model_cfg, checkpoint, device ='cuda', apply_postprocessing=False))
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                print('Processing', thing.path)
                image = Image.open(thing.path)
                image = np.array(image.convert("RGB"))
                # Use a point mask crawl over the image, requesting a mask every X pixels, rather than the preset automatic mask generator
                masks = generator.generate(image)
                segmented_scenes[thing.name] = masks
            del generator
    print(segmented_scenes)
    print('done')
    print('saving to pickle')
    with open(masks_path, 'wb') as f:
        pickle.dump(segmented_scenes, f)
    print('saved')
    print('showing demo')
    first_img = [_ for _ in segmented_scenes][0]
    show = segmented_scenes[first_img]
    show_anns_and_img(Image.open(imgs_path+'/'+first_img),show)

if __name__ == '__main__':
    # test_masking()
    main()
