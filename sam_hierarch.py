import pickle
import string
import random
import torch
import math
from PIL import Image
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import matplotlib.pyplot as plt 
import os
import time

import inspect
import sys


global BIGINT, SHOW_OBJS, DEFRAG_FREQ, DO_DEFRAG
BIGINT = 9999999999
SHOW_OBJS = False
DEFRAG_FREQ = 3
DO_DEFRAG = True

cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

my_clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

masks_path = 'imgs_tunanscene/pickle/img_masks.pickle'
world_path = 'world_collections'

world_name_code = 'tunan_mae'
scene_dir_path = 'imgs_tunanscene/'
scenes = list(sorted([thing.name for thing in os.scandir(scene_dir_path) if thing.is_file()]))

def generate_obj_id():
    alphabet = string.ascii_lowercase + string.digits
    return ''.join(random.choices(alphabet, k=8))

def similarity(embed_1, embed_2):
    return cos(torch.flatten(embed_1), torch.flatten(embed_2))

class ImageWorld:
    def __init__(self, name, init_img=None, init_masks=None):
        self.name = name
        self.imgs = []
        self.entities = {}
        self.entity_sizes = {}
        self.entity_ims = {}
        if init_img is not None and init_masks is not None:
            self.add_img(init_img, init_masks)
    
    def defrag(self):
        print('Defragmenting')
        rejects = []
        for entity_name in self.entities:
            rejects += self.entities[entity_name].reject_outliers()
        for reject in rejects:
            reject_orig_frame = reject[1]['frame']
            reject_orig_img = self.imgs[reject_orig_frame]
            self.add_img(reject_orig_img, [None], *reject, reject_orig_frame)

    def add_img(self, new_img, new_masks, new_repr=None, new_mask_data=None, new_mise_repr=None, curr_frame=None):
        if not curr_frame:
            curr_frame = len(self.imgs)
        if curr_frame == 0:
            init_mode = True
        else:
            init_mode = False
        self.imgs.append(new_img)
        for mask in new_masks:
            mask_data = None
            repr = None
            mise_repr = None
            img_of_obj = None
            if new_repr is None:
                if new_masks is None:
                    raise RuntimeError('No mask supplied for new object/new image')
                print('New object coming in, analysing mask')
                
                seg = np.array(mask['segmentation'])
                idx=(seg==False)
                t_new_img = np.copy(new_img)
                t_new_img[idx] = [255,255,255]

                bbox = [int(_) for _ in mask['bbox']]
                print(bbox)
                img_of_obj = t_new_img[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2], ]
                print('Created cutout of object')
                inputs = processor(text=[''], images=img_of_obj, return_tensors="pt", padding=True)
                repr = my_clip(**inputs).image_embeds.detach()[0]
                print('Created latent space repr of object with CLIP')
                inputs = processor(text=[''], images=t_new_img, return_tensors="pt", padding=True)
                mise_repr = my_clip(**inputs).image_embeds.detach()[0]
                print('Created mise en scene embedding')
                mask_data = {'frame': curr_frame, 'mask': mask, 'img': img_of_obj, 'img_context': t_new_img}
            else:
                try:
                    assert (new_repr is not None) and (new_mask_data is not None) and (new_mise_repr is not None)
                except AssertionError:
                    raise RuntimeError('Must supply img representation, img mask data and mise en scene data for image if not generating from mask')
                print('Repr, mask and mise data already exists, using that instead.')
                repr = new_repr
                mask_data = new_mask_data
                mise_repr = new_mise_repr
                img_of_obj = new_mask_data['img']    
            matched = False
            if init_mode:
                print('skipping object matching for first image.')
            else:
                print('Matching object to known entities')
                # Only add to an entity once per new_img (object permanence)
                matches = []
                for entity_name in self.entities:

                    e = self.entities[entity_name]
                    match_val = e.admit_decision(repr, mise_repr)
                    if match_val > -BIGINT:
                        matched = True
                        matches.append([entity_name, match_val])
                if matched:
                    max_match = max(matches, key=lambda x: x[1])[0]
                    print('Found match in', max_match)
                    self.entities[max_match].add_member(repr, mask_data, mise_repr)
                    self.entity_sizes[max_match] += 1
                    self.entity_ims[max_match].append(img_of_obj)
            if (not matched) or init_mode:
                entity_name = generate_obj_id()
                print('No match found, making new entity', entity_name)

                self.entities[entity_name] = TrackedEntity(entity_name, repr, mask_data, mise_repr)
                print('Generated entity', entity_name)
                self.entity_sizes[entity_name] = 1
                self.entity_ims[entity_name] = [img_of_obj]

                if SHOW_OBJS:
                    plt.figure(figsize=(20, 20))
                    plt.imshow(img_of_obj)
                    plt.axis('off')
                    plt.show()

class TrackedEntity:
    def __init__(self, name, init_repr, init_mask, init_mise):
        self.name = name
        self.data = []
        self.similarities = {}
        self.avg_similarity = BIGINT
        self.confidence_prior = 0.5  # usually 0.85 for clip
        self.inner_cohesion_prior = 0.05
        self.rolling_average_range = 10
        self.masks = []
        self.mise_en_scene = []
        self.add_member(init_repr, init_mask, init_mise)
    
    def add_member(self, member, mask, mise):
        self.data.append(member)
        self.masks.append(mask)
        self.mise_en_scene.append(mise)
        # self.generate_similarities()
    
    def reject_outliers(self):
        print('Defrag in progress for', self.name)
        rejections = []
        done = False
        while not done:
            if len(self.data) < 2:
                print('Categ is a monoid, skipping defrag')
                break
            avgs = {}
            self.generate_similarities()
            for entity_id in range(len(self.data)):
                entity_sims = [_[1] for _ in self.similarities[entity_id].items()]
                entity_avg = sum(entity_sims)/len(entity_sims)
                avgs[entity_id] = entity_avg
            max_outlier = sorted(avgs.items(), key=lambda x: x[1])[0]
            counterfactual_sims = [_[1] for thing in self.similarities for _ in self.similarities[thing].items() if _[0] != max_outlier and thing != max_outlier]
            counterfactual_avg = sum(counterfactual_sims)/len(counterfactual_sims)
            print('max_out - ', max_outlier)
            print('counter_avg - ', counterfactual_avg)
            print('avg_sim - ',self.avg_similarity)
            if (self.avg_similarity - max_outlier[1]) > self.inner_cohesion_prior:
                print('Outlier found, purging')
                rejections.append([self.data[max_outlier[0]], self.masks[max_outlier[0]], self.mise_en_scene[max_outlier[0]]])
                del self.data[max_outlier[0]], self.masks[max_outlier[0]], self.mise_en_scene[max_outlier[0]]
                self.generate_similarities()
            else:
                done = True
        print('Done')
        return rejections
    
    def generate_similarities(self):
        for i in range(len(self.data)):
            if i not in self.similarities:
                self.similarities[i] = {}
            for j in range(i, len(self.data)):
                if j in self.similarities[i]:
                    continue
                else:
                    self.similarities[i][j] = similarity(self.data[i], self.data[j])
        sims = []
        for entity in self.similarities:
            sims+=[_[1] for _ in self.similarities[entity].items()]
        self.avg_similarity = sum(sims)/len(sims)

    def admit_decision(self, proposed_member, proposed_mise_en_scene):
        # Use geographic and spatial priors to modify conf threshold
        if len(self.data) == 0:
            raise RuntimeError('Trying to decide whether to admit member to empty categ')
        elif len(self.data) == 1:
            if similarity(proposed_member, self.data[0]) > self.confidence_prior:
                return similarity(proposed_member, self.data[0])
            else:
                return -BIGINT
        else:
            new_sims = []
            for member in self.data[-self.rolling_average_range:]:  # Only use the last X reprs to construct rolling average similarity
                new_sims.append(similarity(member, proposed_member))
            avg_sim = sum(new_sims)/len(new_sims)

            new_sim_mises = []
            for mise_embedding in self.mise_en_scene[-self.rolling_average_range:]:  # Only use the last X reprs to construct rolling average similarity
                new_sim_mises.append(similarity(mise_embedding, proposed_mise_en_scene))
            avg_sim_mise = sum(new_sim_mises)/len(new_sim_mises)

            avg_overall_sim = (avg_sim*1.75+avg_sim_mise*0.25)/2
            if avg_overall_sim > self.confidence_prior:
                return sum(new_sims)/len(new_sims) 
            else:
                return -BIGINT

def main():
    with open(masks_path, 'rb') as f:
        image_masks = pickle.load(f)
    world = ImageWorld(world_name_code+'_'+str(time.time()))
    i = 0
    for scene in scenes:
        scene_orig_img = Image.open(scene_dir_path+scene)
        scene_orig_img = np.array(scene_orig_img.convert("RGB"))
        mask = image_masks[scene]
        world.add_img(scene_orig_img, mask)
        if i>0 and i%DEFRAG_FREQ==0 and DO_DEFRAG:
            world.defrag()
        i += 1
    entity_sizes = list(sorted(world.entity_sizes.items(), key=lambda x: x[1], reverse=True))
    print(len(world.entities))
    print('\n'.join([f'{item[0]} -> {item[1]}' for item in entity_sizes]))
    print('Saving world')
    try:
        os.rmdir(world_path+'/'+world.name)
    except FileNotFoundError:
        pass
    os.mkdir(world_path+'/'+world.name)
    with open(f'{world_path}/{world.name}/sam_hierarch_source_{world.name}.py', 'w') as f:
        f.write(inspect.getsource(sys.modules[__name__]))
    for entity in world.entities:
        elabel = str(world.entity_sizes[entity])+'_'+entity
        os.mkdir(world_path+'/'+world.name+'/'+elabel)
        i = 0
        for im in world.entity_ims[entity]:
            res = Image.fromarray(im)
            res.save(world_path+'/'+world.name+'/'+elabel+'/'+f'{i}.png')
            i += 1
    print('World saved. Done.')

if __name__ == '__main__':
    main()
