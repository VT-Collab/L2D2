import numpy as np
from utils import Joystick, Camera, Trajectory
import socket
import pickle
import time
import os, sys
import json
import cv2

def teleop(cfg):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_address = ('172.16.0.3', 10982)
    s.bind(server_address)
    print('Starting up on {} port {}'.format(*server_address))

    s.listen()
    print("listening")
    conn, addr = s.accept()
    print("Connection Established")

    interface = Joystick()
    camera = Camera(0)
	
    for demo_num in range(cfg.demo_num, 100):
        print(demo_num)
        traj = []
        img_traj = []
        save_name = 'data/{}/{}/'.format(cfg.task, demo_num)
        if not os.path.exists(save_name):
            os.makedirs(save_name)
		

        record = False
        print("[*] Press X to Start Recording Play Data")
        while True:
            z, a_pressed, b_pressed, x_pressed, y_pressed, start = interface.input()
            try:
                img_ee_pos = camera.get_ee_pos()
                if start:

                    json.dump(traj, open(save_name + 'traj.json', 'w'))
                    json.dump(img_traj, open(save_name +'img_traj.json', 'w'))
                    print("[*] Recorded Trajectory of length: ", len(traj))
                    camera.process_img(save_name=save_name)

                    print("[*] Press A to record another demonstration and B to exit")
                    time.sleep(0.5)


                if x_pressed:
                    camera.get_img(save_name=save_name)
                    record = True

                if a_pressed:
                    conn.send(pickle.dumps('home', protocol=2))
                    time.sleep(0.5)
                    break
				
                if b_pressed:
                    conn.send(pickle.dumps('stop', protocol=2))
                    time.sleep(0.1)
                    s.close()
                    exit()

				
                if record:
                    action = [0.15*z[1], -0.15*z[0], 0.15*z[2], 0., 0., 0., -1.]
                    action = pickle.dumps(action, protocol=2)
                    conn.send(action)
                    data = conn.recv(1024)
                    state = pickle.loads(data)
                    if img_ee_pos is None:
                        continue
                    traj.append(state)
                    img_traj.append(img_ee_pos)
            except KeyboardInterrupt:
                exit()