import numpy as np
import torch
import cv2
import json
import socket
import pickle
import time
from models import BC_Vis, Model
from utils import Joystick, Camera
from DETIC.Detic.manipulate_img import DETICSegment

    
def eval_imitation(cfg):
    # Initialize connection with the robot
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_address = ('172.16.0.3', 10982)
    s.bind(server_address)
    print('Starting up on {} port {}'.format(*server_address))

    s.listen()
    print("listening")
    conn, addr = s.accept()
    print("Connection Established")
    time.sleep(5.0)
    
    # Initialize camera and segmentation model
    camera = cv2.VideoCapture(0)
    time.sleep(0.01)
    camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    interface = Joystick()
    sam = DETICSegment(cfg)
    
    # Load the trainied models for evaluation
    num_ensembles = cfg.num_ensembles
    bc_models = []
    in_dim = 16 if cfg.task == 'long_horizon' else 16
    for idx in range(num_ensembles):
        bc_model = BC_Vis(in_dim)
        if cfg.fine_tune:
            bc_model.load_state_dict(torch.load('data/{}/{}/model_{}_ft.pt'.format(cfg.task, cfg.alg, idx)))
        else:
            bc_model.load_state_dict(torch.load('data/{}/{}/model_{}.pt'.format(cfg.task, cfg.alg, idx)))
        bc_model.eval()
        bc_models.append(bc_model)

    
    ## Load trained models for reconstruction
    models = []
    for idx in range(cfg.num_ensembles):
        model = Model()
        if not cfg.fine_tune:
            model.load_state_dict(torch.load('data/play/model_{}.pt'.format(idx), weights_only=True))
        else:
            model.load_state_dict(torch.load('data/{}/{}/recon_model_{}.pt'.format(cfg.task, cfg.alg, idx), weights_only=True))
             
        model.eval()
        models.append(model)
    
    # Get the initial image of environment and extract the initial object poses
    _, frame = camera.read()
    frame = cv2.flip(frame, 0)
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (1080, 810), interpolation=cv2.INTER_CUBIC)
    clone = frame.copy()
    roi = cfg.roi
    img = clone[int(roi[1]):int(roi[1] + roi[3]), \
                int(roi[0]):int(roi[0]+ roi[2])]
    
    img = cv2.resize(img, (256, 256))
    obj_pos = sam.get_boxes(img)
    obj_pos = obj_pos.flatten().numpy().astype(np.float64)
    focused_pos = []
    for obj in range(len(obj_pos)//4):
        focused_pos.append(2*(np.mean([obj_pos[4*obj], obj_pos[4*obj+2]]))/255 - 1)
        focused_pos.append(2*(np.mean([obj_pos[4*obj+1], obj_pos[4*obj+3]]))/255 - 1)
    
    img = cv2.resize(frame, (540, 405))
    cv2.imshow('img', img)
    obj_pos = sam.get_boxes(img)
    obj_pos = obj_pos.flatten().numpy().astype(np.float64)*2
    encoded_pos_2d = []

    for obj in range(len(obj_pos)//4):
        encoded_pos_2d.append(np.mean([obj_pos[4*obj], obj_pos[4*obj+2]]))
        encoded_pos_2d.append(np.mean([obj_pos[4*obj+1], obj_pos[4*obj+3]]))
    obj_pred = []
    for idx in range(cfg.num_ensembles):
        pred = models[idx](torch.FloatTensor(encoded_pos_2d).reshape((len(obj_pos)//4, 2)))
        obj_pred.append(pred.detach().numpy())
    encoded_pos_3d = np.mean(obj_pred, axis=0)

    action = [0.]*7
    if cfg.task == 'scooping':
        action = [0.] * 6 + [1]
    action = pickle.dumps(action, protocol=2)
    timestep = 1.
    print("Ready")
    play = False

    # Main Evaluation Loop
    while True:
            z, a_pressed, b_pressed, x_pressed, y_pressed, start = interface.input()
            U = []
            if timestep%1==0:
                # get image of the environment and extract environment state
                _, frame = camera.read()
                frame = cv2.flip(frame, 0)
                frame = cv2.flip(frame, 1)
                frame = cv2.resize(frame, (1080, 810), interpolation=cv2.INTER_CUBIC)
                print(timestep)

                img = cv2.resize(frame, (540, 405))
                obj_pos = sam.get_boxes(img)

                obj_pos = obj_pos.flatten().numpy().astype(np.float64)*2
                if len(obj_pos)>0:
                    encoded_pos_2d = []
                    for obj in range(len(obj_pos)//4):
                        encoded_pos_2d.append(np.mean([obj_pos[4*obj], obj_pos[4*obj+2]]))
                        encoded_pos_2d.append(np.mean([obj_pos[4*obj+1], obj_pos[4*obj+3]]))
                    obj_pred = []
                    for idx in range(cfg.num_ensembles):
                        pred = models[idx](torch.FloatTensor(encoded_pos_2d).reshape((len(obj_pos)//4, 2)))
                        obj_pred.append(pred.detach().numpy())
                    encoded_pos_3d = np.mean(obj_pred, axis=0)
                    encoded_pos_3d = np.concatenate((encoded_pos_3d.flatten(), np.array(encoded_pos_2d)*2/1080-1))
                    rect = (obj_pos/2).astype(np.int64)
                    cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (255, 0, 0))
                cv2.imshow('img', img)
                cv2.waitKey(1)
            
            # Start the evaluation
            if x_pressed:
                play = True
                time.sleep(0.1)
            
            # Pause the evaluation
            if a_pressed:
                action = [0.]*7
                action = pickle.dumps(action, protocol=2)
                play = False
                time.sleep(0.1)
            
            # Stop evaluation and send robot to home pose
            if start:
                conn.send(pickle.dumps('home', protocol=2))
                conn.send(pickle.dumps('stop', protocol=2))
                time.sleep(0.1)
                s.close()
                exit()
            
            # Get actions from trained models and send them to robot
            if play:
                timestep += 1
                state = torch.FloatTensor(state)
                for idx in range(len(bc_models)):
                    _, u_r, _ = bc_models[idx](state[:9].unsqueeze(0), torch.FloatTensor(np.concatenate((focused_pos, encoded_pos_3d))).unsqueeze(0))
                    U.append(u_r.squeeze().detach().numpy())
                action = (1.*np.mean(U, axis=0))
                action[:3] /= 1000.
                action[3:6] /= 100.
                action = action.tolist()
                if cfg.task != 'scooping' and state[2]< 0.21 and action[2] < 0.:
                     action[2] = 0.
                elif cfg.task == 'scooping' and state[2]< 0.3 and action[2] < 0.:
                    action[2] = 0.
                action[0] *= 3.
                action[1] *= 3.
                action[2] *= 3.
                action[3] *= 3.
                action[4] *= 3.
                action[5] *= 3.
                if cfg.task != 'scooping':
                    action[3:6] = [0.]*3
                action = pickle.dumps(action, protocol=2)

            conn.send(action)
            data = conn.recv(1024)
            state = pickle.loads(data)
