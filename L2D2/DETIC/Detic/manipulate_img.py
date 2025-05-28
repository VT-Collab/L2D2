from detectron2.utils.logger import setup_logger
setup_logger()

import sys
import numpy as np
import torch
import cv2
import pygame
import json

# detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

# Detic libraries
sys.path.insert(0, 'DETIC/Detic/third_party/CenterNet2')
from centernet.config import add_centernet_config
from DETIC.Detic.detic.config import add_detic_config
from DETIC.Detic.detic.config import add_detic_config
from DETIC.Detic.detic.modeling.utils import reset_cls_test
from DETIC.Detic.detic.modeling.text.text_encoder import build_text_encoder

class DETICSegment:
    
    def __init__(self, cfg):
        self.robot_cfg = cfg

        self.config_path = "DETIC/Detic/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml"
        self.weights_path = "DETIC/Detic/models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"
        self.image_path = self.robot_cfg.img_path
        if self.robot_cfg.get_prompts:
            self.classes = self.get_prompts(self.image_path)
        else:
            self.classes = json.load(open('data/{}/{}/object_prompts.json'.format(self.robot_cfg.task, self.robot_cfg.alg), 'r'))
        self.cfg = None
        self.predictor = None
        self.metadata = None
        self.classifier = None
        self.masks = None
        self.num_classes = len(self.classes)

        self.text_encoder = build_text_encoder(pretrain=True)
        self.text_encoder.eval()
        self.setup_config()
        self.m = []
        self.predictor = DefaultPredictor(self.cfg)
        for idx in range(self.num_classes):
            self.classifier = self.get_clip_embeddings(self.classes[idx])
            reset_cls_test(self.predictor.model, self.classifier, 1)
            
            image = cv2.imread(self.image_path)
            self.run_inference(image)
            obx, oby = self.get_objects(self.masks)
            self.m.append([obx, oby])

    # Config Setup
    def setup_config(self):
        cfg = get_cfg()
        add_centernet_config(cfg)
        add_detic_config(cfg)
        cfg.merge_from_file(self.config_path)
        cfg.MODEL.WEIGHTS = self.weights_path
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
        cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
        cfg.freeze()
        self.cfg = cfg
        self.metadata = MetadataCatalog.get("__unused")
        self.metadata.thing_classes = self.classes
    
    # Prompt to Embeddings
    def get_clip_embeddings(self, text_prompt):
        # print(text_prompt)
        texts = ['a ' + text_prompt]
        embeddings = self.text_encoder(texts).detach().permute(1, 0).contiguous().cpu()
        return embeddings

    # Prediction   
    def run_inference(self, im):
        outputs = self.predictor(im)

        # Visualize the results
        v = Visualizer(im[:, :, ::-1], self.metadata)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        boxes = outputs["instances"].pred_boxes
        self.masks = outputs["instances"].pred_masks
        self.masks = self.masks.cpu().detach().numpy()
        return boxes
    
    def get_boxes(self, image):
        IMG_POS = []
        for idx in range(self.num_classes):
            self.classifier = self.get_clip_embeddings(self.classes[idx])
            reset_cls_test(self.predictor.model, self.classifier, 1)
            
            box = self.run_inference(image)
            box = box.tensor.detach()
            if len(box)>0:
                IMG_POS += box[0]
        return torch.FloatTensor(IMG_POS)

    def get_objects(self, mask):
        color = np.array([30/255, 144/255, 255/255, 0.6])
        object_indices_x, object_indices_y = np.where(mask.mean(axis=0) != 0)
        return object_indices_x, object_indices_y
    
    def get_prompts(self, img_path):
        pygame.init()
        X = 256
        Y = 256
        screen = pygame.display.set_mode((X, Y))
        pygame.display.set_caption('image')

        img = pygame.image.load(img_path).convert()
        screen.blit(img, (0, 0))
        pygame.display.flip()
        prompts = []
        print("[*] What are the objects of interest in this environment?")
        while True:
            obj = input()
            if obj == '':
                pygame.quit()
                break            
            prompts.append(obj)
            print("[*] Enter any other objects of interest or press ENTER")
        
        json.dump(prompts, open('data/{}/{}/object_prompts.json'.format(self.robot_cfg.task, self.robot_cfg.alg), 'w'))

        return prompts

    

    def segment_image(self, center=None, seed=None):
        np.random.seed(seed+np.random.randint(0, 1000))
        image = cv2.imread(self.robot_cfg.img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        background = cv2.imread(self.robot_cfg.bg_path)
        background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)

        temp = image.copy()
        IMG_POS = []
        
        for obj_x, obj_y in self.m:
            
            if center is None:
                new_x_pos = obj_x + np.random.randint(-128, 128)
                new_y_pos = obj_y + np.random.randint(-128, 128)
            else:
                new_x_pos = obj_x + int(center[0])
                new_y_pos = obj_y + int(center[1])
            max_x, min_x = np.max(new_x_pos), np.min(new_x_pos)
            if max_x >= 255:
                new_x_pos -= (max_x - 255)
            if min_x < 0:
                new_x_pos += abs(min_x)

            max_y, min_y = np.max(new_y_pos), np.min(new_y_pos)
            if max_y >= 255:
                new_y_pos -= (max_y - 255)
            if min_y < 0:
                new_y_pos += abs(min_y)

            max_x, min_x = np.max(new_x_pos), np.min(new_x_pos)
            max_y, min_y = np.max(new_y_pos), np.min(new_y_pos)
            
            background[new_x_pos, new_y_pos, :] = temp[obj_x, obj_y, :]

        return background
