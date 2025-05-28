import numpy as np
import json
import torch
import os, sys
import cv2 
import socket
import pickle 

from models import BC_Vis, Model
from utils import Camera, reconstruct_traj

curr_dir = os.getcwd()
child = 'FastSAM'
child_path = os.path.join(curr_dir, child)
sys.path.append(child_path)
from DETIC.Detic.manipulate_img import DETICSegment


def get_demos(cfg):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    server_address = ('172.16.0.3', 10980)
    s.bind(server_address)
    print('Starting Port {} on Port {}'.format(*server_address))
    s.listen()
    print("Socket Ready")

    demo_num = cfg.demo_num
    cam = Camera(0)
    save_name = 'data/{}/{}/'.format(cfg.task, cfg.alg)
    
    roi = cfg.roi

    if demo_num == 0 or cfg.get_img:
        input("Remove all objects of interest from the environment and press ENTER")
        img = cam.get_img()
        bg_img = img[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0]+ roi[2])]
        bg_img.resize(bg_img, (256, 256))
        cv2.imwrite(save_name + 'bg_img.png', bg_img)

        input("Place the objects back in the environment and press ENTER")
        img = cam.get_img()
        env_image = img[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0]+ roi[2])]
        env_image = cv2.resize(env_image, (256, 256))
        cv2.imwrite(save_name + 'env_0.png', env_image)
        cv2.imwrite(save_name + '0.png', img)

    cfg.img_path = save_name + 'env_0.png'
    cfg.bg_path = save_name + 'bg_img.png'
    sam = DETICSegment(cfg)
    
    
    while True:
        try:
            conn, addr = s.accept()
            conn.settimeout(1000.0)
            full_img = cv2.imread(save_name + '0.png')
            print(demo_num)
            if demo_num != 0:
                IMG = sam.segment_image(seed=demo_num)
                IMG = cv2.resize(IMG, tuple(cfg.roi_size), interpolation=cv2.INTER_CUBIC)
                IMG = np.array(IMG, dtype=np.uint8)
                IMG = cv2.cvtColor(IMG, cv2.COLOR_RGB2BGR)
                full_img[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0]+ roi[2])] = IMG
                cv2.imwrite(save_name + '{}.png'.format(demo_num), full_img)
            conn.sendall(pickle.dumps([0, full_img, cfg.task]))
            data = b''
            rec = conn.recv(int(10**8))
            conn.settimeout(1.0)
            while rec != b'':
                try:
                    data += rec
                    rec = conn.recv(int(1e8))
                except socket.timeout:
                    break
            data = pickle.loads(data)
            if data == []:
                continue
            if data == "done":
                print("Done")
                conn.close()
                exit()
            json.dump(data, open(save_name + '{}.json'.format(demo_num), 'w'))
            reconstruct_traj(cfg, str(demo_num), sam)
            demo_num += 1
        except socket.timeout:
            continue


def refine_demos(cfg):
    demo_dir = 'data/{}/{}/'.format(cfg.task, cfg.alg)
    digits = [str(i) for i in range(10)]

    cfg.img_path = demo_dir + 'env_0.png'
    sam = DETICSegment(cfg)

    num_demos = 0
    for filename in os.listdir(demo_dir):
        if ('.json' in filename) and (filename[0] in digits):
            num_demos += 1
    for idx in range(num_demos):
        reconstruct_traj(cfg, str(idx), sam)
