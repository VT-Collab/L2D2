import numpy as np
import torch
import cv2
import time
from utils import Camera, Joystick
import socket
import json
import pickle
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation
from DETIC.Detic.manipulate_img import DETICSegment
from models import Model
import os, sys
from tqdm import tqdm


def wrap_angular_actions(theta):
	if abs(theta[0]) > np.pi:
		theta[0] -= 2*np.sign(theta[0])*np.pi

	if abs(theta[1]) > np.pi:
		theta[1] -= 2*np.sign(theta[1])*np.pi

	if abs(theta[2]) > np.pi:
		theta[2] -= 2*np.sign(theta[2])*np.pi

	return theta


def process_demos(cfg):

	num_demos = cfg.demo_num
	sam = DETICSegment(cfg)

	save_name = 'data/{}/{}/'.format(cfg.task, cfg.alg)
	for idx in tqdm(range(num_demos)):
		demo_num=idx
		traj = json.load(open(save_name + 'c{}.json'.format(idx), 'r'))
		img_traj = json.load(open(save_name + 'c_0_img_{}.json'.format(idx), 'r'))
		img_dict = json.load(open(save_name + 'c_0_obj_{}.json'.format(idx), 'r'))

		obj_traj = []
		if cfg.get_obj:
			for idx in range(len(img_dict)):
				img = cv2.imread(img_dict[str(idx)])
				img = cv2.resize(img, (540, 405))
				obj_pos = sam.get_boxes(img)
				rect = obj_pos.flatten().numpy().astype(np.int64)
				
				if len(obj_pos)>0:
					obj_pos = obj_pos.flatten().numpy().astype(np.float64)*2
					obj_traj.append(obj_pos.tolist())
					prev_obj_pos = obj_pos.copy()
				else:
					obj_traj.append(prev_obj_pos.tolist())
			json.dump(obj_traj, open('data/{}/{}/obj0_traj_{}.json'.format(cfg.task, cfg.alg, demo_num), 'w'))
		else:
			obj_traj = json.load(open('data/{}/{}/obj0_traj_{}.json'.format(cfg.task, cfg.alg, demo_num), 'r'))

		if cfg.task != 'long_horizon':
			if (len(traj)//100) > 1:
				traj = traj[::len(traj)//100]
			if (len(img_traj)//100) > 1:
				img_traj = img_traj[::len(img_traj)//100]
			if (len(obj_traj)//100) > 1:
				obj_traj = obj_traj[::len(obj_traj)//100]
		
		if cfg.task == 'long_horizon':
			if (len(traj)//200) > 1:
				traj = traj[::len(traj)//200]
			if (len(img_traj)//200) > 1:
				img_traj = img_traj[::len(img_traj)//200]
			if (len(obj_traj)//200) > 1:
				obj_traj = obj_traj[::len(obj_traj)//200]
		
		traj = np.array(traj)[1:]

		
		img_traj = np.array(img_traj)[1:]
		obj_traj = np.array(obj_traj)[1:]

		gripper_traj = np.clip(traj[:, -1], 0, 1)
		traj = traj[:, :-1]
		window = 50 
		if cfg.filter:
			traj[:, 0] = savgol_filter(traj[:, 0], window, 2)
			traj[:, 1] = savgol_filter(traj[:, 1], window, 2)
			traj[:, 2] = savgol_filter(traj[:, 2], window, 2)

		theta_traj = traj[:, 3:7]
		models = []
		if cfg.alg == 'l2d2':
			for idx in range(cfg.num_ensembles):
				model = Model()
				model.load_state_dict(torch.load('data/play/model_{}.pt'.format(idx), weights_only=True))
				if cfg.fine_tune:
					model.load_state_dict(torch.load('data/{}/{}/recon_model_{}.pt'.format(cfg.task, cfg.alg, idx), weights_only=True))
				model.eval()
				models.append(model)

		img_og = cv2.imread('data/{}/{}/0_c{}.png'.format(cfg.task, cfg.alg, demo_num))
		
		clone = img_og.copy()
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

		sa_arr = []
		for idx in range(0, len(traj)-1):
			gripper_action = gripper_traj[idx]
			obj_pos = obj_traj[idx]
			embedded_pos = []

			for obj in range(len(obj_pos)//4):
				embedded_pos.append(np.mean([obj_pos[4*obj], obj_pos[4*obj+2]]))
				embedded_pos.append(np.mean([obj_pos[4*obj+1], obj_pos[4*obj+3]]))
			if cfg.alg == 'l2d2':
				embedded_pos = torch.FloatTensor(embedded_pos)
				obj_3d = []
				for idy in range(cfg.num_ensembles):
					pred = models[idy](embedded_pos.reshape((len(obj_pos)//4, 2)))
					obj_3d.append(pred.detach().numpy())
				obj_3d = np.mean(obj_3d, axis=0)
				obj_3d = np.concatenate((obj_3d.flatten(), embedded_pos.detach().numpy()/1080*2-1))
			else:
				obj_3d = 2*np.array(embedded_pos)/1080 - 1

			action_lin = traj[idx+1, :3] - traj[idx, :3]
			r = Rotation.from_quat(theta_traj[idx])
			theta1 = r.as_euler('xyz', degrees=False)
			r = Rotation.from_quat(theta_traj[idx+1])
			theta2 = r.as_euler('xyz', degrees=False)
			
			action_ang = wrap_angular_actions(theta2 - theta1)*100
			if not cfg.task == 'scooping':
				action_ang = (theta2 - theta1)*0

			action = np.concatenate((action_lin*1000., action_ang))

			if (action>0).any():
				sa_arr.append(traj[idx].tolist() + focused_pos + obj_3d.tolist() + action.tolist() + [gripper_action] + [1.0])

			if idx==len(traj)-2:
				for _ in range(10):
					sa_arr.append(traj[idx].tolist() + focused_pos + obj_3d.tolist() + [0.]*6 + [gripper_action] + [1.0])

		sa_arr = np.array(sa_arr)
		json.dump(sa_arr.tolist(), open('data/{}/{}/demo_c_{}.json'.format(cfg.task, cfg.alg, demo_num), 'w'))


		


def teleop(cfg):
	s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
	server_address = ('172.16.0.3', 10982)

	s.bind(server_address)
	print('Starting Port {} on Port {}'.format(*server_address))
	s.listen()
	print("Listening")
	conn, addr = s.accept()
	print("Connection Established")

	camera1 = Camera(0)
	time.sleep(0.5)
	corr_num = cfg.corr_num
	if cfg.alg=='l2d2':
			corr_num = cfg.user

	save_name = 'data/{}/{}/'.format(cfg.task, cfg.alg)
	if cfg.get_img:
		frame = camera1.get_img()
		frame = cv2.resize(frame, (256, 256))
		cv2.imwrite(save_name + 'env_0.png', frame)

	cfg.img_path = 'data/{}/{}/env_0.png'.format(cfg.task, cfg.alg)

	interface = Joystick()
	record = False
	gripper = False
	pause = False
	angular = False
	traj = []
	img_traj_1 = []
	img_dict_1 = {}
	print("[*] Press X to start recording corrections")
	timestep = 0
	action = [0.]*6 + [1] if cfg.task == 'scooping' else [0.]*7
	while True:
		z, a_pressed, b_pressed, x_pressed, y_pressed, start = interface.input()
		try: 
			img1 = camera1.get_img()
			img_ee_pos1 = camera1.get_ee_pos()
			if start:
				json.dump(traj, open(save_name + 'c{}.json'.format(corr_num), 'w'))
				json.dump(img_traj_1, open(save_name +'c_{}_img_{}.json'.format(0, corr_num), 'w'))
				json.dump(img_dict_1, open(save_name +'c_{}_obj_{}.json'.format(0, corr_num), 'w'))
				print("[*] Recorded Trajectory of length: ", len(traj))

				print("[*] Press X to record another demonstration and B to exit")
				pause = True
				time.sleep(0.5)

			if not record:
				clone = img1.copy()
				roi = cfg.roi
				frame = clone[int(roi[1]):int(roi[1] + roi[3]), \
				int(roi[0]):int(roi[0]+ roi[2])]
				cv2.imshow('focused_pos', frame)
				cv2.waitKey(1)
			
			if x_pressed and not record:
				if not os.path.exists(save_name + str(corr_num) + '/'):
					os.makedirs(save_name + str(corr_num) + '/')
				img_path = save_name + str(corr_num) + '/'
				camera1.get_img(save_name=save_name, demo_num='c{}'.format(corr_num))
				record = True
				print("start operating robot")
				time.sleep(0.5)
			
			if x_pressed and pause:
				record = False
				pause = False
				gripper = False
				angular = False
				conn.send(pickle.dumps([0]*6 + [0.], protocol=2))
				conn.send(pickle.dumps('home', protocol=2))
				traj  = []
				img_traj_1 = []
				img_dict_1 = {}
				timestep = 0
				corr_num += 1
				print(corr_num)
				print("[*] Press X to start recording corrections")
				time.sleep(0.5)
			
			if a_pressed and not gripper:
				gripper = True
				time.sleep(0.5)			
			elif a_pressed and gripper:
				gripper = False
				time.sleep(0.5)

			if y_pressed and not angular:
				angular = True
				time.sleep(0.2)
			elif y_pressed and angular:
				angular = False
				time.sleep(0.2)

			if b_pressed:
				conn.send(pickle.dumps('stop', protocol=2))
				time.sleep(0.1)
				s.close()
				return corr_num

			if record and not pause:
				if not angular:
					action = [0.1*z[1], -0.1*z[0], 0.05*z[2], 0., 0., 0., 
					0]
				else:
					action = [0., 0., 0., 0.1*z[1], -0.1*z[0], 0.1*z[2], 0]
				action[-1] = 0 if not gripper else 1
				if cfg.task == 'scooping':
					action[-1] = 1
				
				action_enc = pickle.dumps(action, protocol=2)
				conn.send(action_enc)
				data = conn.recv(1024)
				state = pickle.loads(data)
				state.append(action[-1])
				if img_ee_pos1 is None:
					continue
				cv2.imwrite(img_path + '{}_{}.png'.format(0, timestep), img1)
				img_dict_1[timestep] = img_path + '{}_{}.png'.format(0, timestep)
				traj.append(state)
				img_traj_1.append(img_ee_pos1)
				timestep += 1
			else:
				action = [0.]*6 + [1] if cfg.task == 'scooping' else [0.]*7
				conn.send(pickle.dumps(action, protocol=2))
				data = conn.recv(1024)
				state = pickle.loads(data)
				state.append(action[-1])


		except KeyboardInterrupt:
			conn.send(pickle.dumps('stop', protocol=2))
			time.sleep(0.1)
			s.close()
			exit()