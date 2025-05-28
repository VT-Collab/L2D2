import os, sys
import numpy as np
import pybullet as p
import pybullet_data
import pygame
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation
import pygame
import json
import time
import torch
from models import Model
import cv2
import pickle
from scipy.signal import savgol_filter


class Joystick(object):
	def __init__(self):
		pygame.init()
		self.gamepad = pygame.joystick.Joystick(0)
		self.gamepad.init()
		self.deadband = 0.1

	def input(self):
		pygame.event.get()
		z1 = self.gamepad.get_axis(0)
		z2 = self.gamepad.get_axis(1)
		z3 = self.gamepad.get_axis(4)
		if abs(z1) < self.deadband:
			z1 = 0.0
		if abs(z2) < self.deadband:
			z2 = 0.0
		if abs(z3) < self.deadband:
			z3 = 0.0
		A_pressed = self.gamepad.get_button(0)
		B_pressed = self.gamepad.get_button(1)
		X_pressed = self.gamepad.get_button(2)
		Y_pressed = self.gamepad.get_button(3)
		START_pressed = self.gamepad.get_button(7)
		return [z1, -z2, -z3], A_pressed, B_pressed, X_pressed, Y_pressed, START_pressed
	

"""Interpolating the generated trajectory for execution on robot"""
class Trajectory(object):

	def __init__(self, xi, T):
		""" create cublic interpolators between waypoints """
		self.xi = np.asarray(xi)
		self.T = T
		self.n_waypoints = xi.shape[0]
		timesteps = np.linspace(0, self.T, self.n_waypoints)
		self.f1 = interp1d(timesteps, self.xi[:,0], kind='cubic')
		self.f2 = interp1d(timesteps, self.xi[:,1], kind='cubic')
		self.f3 = interp1d(timesteps, self.xi[:,2], kind='cubic')
		self.f4 = interp1d(timesteps, self.xi[:,3], kind='cubic')
		self.f5 = interp1d(timesteps, self.xi[:,4], kind='cubic')
		self.f6 = interp1d(timesteps, self.xi[:,5], kind='cubic')

	def get(self, t):
		""" get interpolated position """
		if t < 0:
			q = [self.f1(0), self.f2(0), self.f3(0), self.f4(0), self.f5(0), self.f6(0)]
		elif t < self.T:
			q = [self.f1(t), self.f2(t), self.f3(t), self.f4(t), self.f5(t), self.f6(t)]
		else:
			q = [self.f1(self.T), self.f2(self.T), self.f3(self.T), self.f4(self.T
																   ), self.f5(self.T), self.f6(self.T)]
		return np.asarray(q)
	
class Camera(object):
	def __init__(self, camera_id=0):
		self.id = camera_id
		self.vs = cv2.VideoCapture(self.id)
		time.sleep(0.1)
		self.vs.set(cv2.CAP_PROP_AUTOFOCUS, 0)

		self.arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
		self.arucoParams = cv2.aruco.DetectorParameters()
		self.detector = cv2.aruco.ArucoDetector(self.arucoDict, self.arucoParams)

	def get_img(self, save_name=None, demo_num=None):
		for _ in range(2):
			_, frame = self.vs.retrieve(self.vs.grab())
		frame = cv2.flip(frame, 0)
		frame = cv2.flip(frame, 1)

		frame = cv2.resize(frame, (1080, 810), interpolation=cv2.INTER_CUBIC)
		if save_name is None:
			return frame
		if demo_num is None:
			cv2.imwrite(save_name + 'env_{}.png'.format(self.id), frame)
		else:
			cv2.imwrite(save_name + '{}_{}.png'.format(self.id, demo_num), frame)

	def get_ee_pos(self):
		_, frame = self.vs.read()
		frame = cv2.flip(frame, 0)
		frame = cv2.flip(frame, 1)
		frame = cv2.resize(frame, (1080, 810), interpolation=cv2.INTER_CUBIC)
		corners, ids, _ = self.detector.detectMarkers(frame)
		if len(corners) < 1:
			return None 
		for idx, corner in enumerate(corners):
			if ids[idx] == 75:
				cv2.aruco.drawDetectedMarkers(frame, np.array([corner]), np.array([ids[idx]]))
				corner = np.asarray(corner).squeeze()
				center_y = np.mean(corner[:, 0], dtype=np.float64)
				center_x = np.mean(corner[:, 1], dtype=np.float64)
				frame = cv2.circle(frame, (np.uint8(center_x),np.uint8(center_y)), radius=0, color=(0, 0, 255), thickness=5)
				cv2.imshow('Detected Markers {}'.format(self.id), frame)
				cv2.waitKey(1)
				return (center_y, center_x)

	def process_img(self, save_name):
		img = cv2.imread(save_name + 'env_{}.png'.format(self.id))
		traj_2d = np.asarray(json.load(open(save_name + 'img_traj.json'.format(self.id), 'r'))).astype(np.int32)
		img = cv2.polylines(img, [traj_2d], False, (0,0,255), 2)
		cv2.imwrite(save_name + '2d_traj{}.png'.format(self.id), img)

	def close(self):
		self.vs.release()


def reconstruct_traj(cfg, demo_num, sam):
	print(demo_num)
	start_theta = np.array([3.140, -0.0004, 0.005])

	## Load trajectory and image data
	data = torch.FloatTensor(json.load(open('data/{}/{}/{}.json'.format(cfg.task, cfg.alg, demo_num), 'r')))

	if (len(data)//100) > 1:
		data = data[::len(data)//100]
	traj_s = np.zeros_like(data)
	window_len = 50 
	traj_s[:, 0] = savgol_filter(data[:, 0], window_len, 2)
	traj_s[:, 1] = savgol_filter(data[:, 1], window_len, 2)
	traj_s[:, 2] = savgol_filter(data[:, 2], window_len, 2)
	traj_s[:, 3] = savgol_filter(data[:, 3], window_len, 2)
	traj_s[:, 4] = savgol_filter(data[:, 4], window_len, 2)
	traj_s[:, 5] = np.clip(data[:, 5], 0, 1)
	state_gripper = np.clip(data[:, 5], 0, 1)

	data = torch.FloatTensor(traj_s.copy())


	## Load trained models
	models = []
	for idx in range(cfg.num_ensembles):
		model = Model()
		if cfg.corr:
			model.load_state_dict(torch.load('data/{}/{}/recon_model_{}.pt'.format(cfg.task, cfg.alg, idx), weights_only=True))
		else:
			model.load_state_dict(torch.load('data/play/model_{}.pt'.format(idx), weights_only=True))
		model.eval()
		models.append(model)

	
	## Get object state information
	img_og = cv2.imread('data/{}/{}/{}.png'.format(cfg.task, cfg.alg, demo_num))
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
	focused_pos = np.array(focused_pos)

	
	img = cv2.resize(img_og, (540, 405))
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
	encoded_pos_3d = np.concatenate((encoded_pos_3d.flatten(), np.array(encoded_pos_2d)/1080*2-1))


	if encoded_pos_3d[2] < 0.:
		encoded_pos_3d[2] = 0.


	## Get reconstructed trajecroty
	traj_theta = data[:, 2:5].detach().numpy()
	traj_theta = traj_theta/180*np.pi
	traj_theta += start_theta
	traj_gripper = data[:, 5].detach().numpy().astype(np.float64)
	state_gripper = state_gripper.detach().numpy().astype(np.float64)
	pred_arr = []
	for idx in range(cfg.num_ensembles):
		pred = models[idx](data[:, :2])
		pred_arr.append(pred.detach().numpy())
	var = np.std(pred_arr, axis=0)
	var = (1/(np.mean(var, axis=-1) + 1.0)).tolist()
	pred = np.mean(pred_arr, axis=0)
	for idx in range(len(pred)):
		if cfg.task != 'scooping' and pred[idx, 2] < 0.24:
			pred[idx, 2] = 0.24
		elif cfg.task == 'scooping' and pred[idx, 2] < 0.3:
			pred[idx, 2] = 0.3
		
	pred[:, 0] = savgol_filter(pred[:, 0], window_len, 2)
	pred[:, 1] = savgol_filter(pred[:, 1], window_len, 2)
	pred[:, 2] = savgol_filter(pred[:, 2], window_len, 2)
	pred = np.concatenate((pred, traj_theta), axis=1, dtype=np.float64)

	sa_pairs = []
	for idx in range(0, len(pred)-1):
		action = (pred[idx+1] - pred[idx])
		action[:3] *= 1000
		action[3:6] = wrap_angular_actions(action[3:6])*100
		if cfg.task != 'scooping':
			action[3:6] == 0.
		action_2d = traj_s[idx+1,:2] - traj_s[idx, :2]
		state = pred[idx].tolist()
		r = Rotation.from_euler('xyz', state[3:6], degrees=False)
		quat = r.as_quat()
		if state_gripper[idx] == 1:
			gripper_state = [0, 1]
		else:
			gripper_state = [1, 0]
		gripper_action = traj_gripper[idx]
		if cfg.task == 'scooping':
			gripper_action = 1.
			gripper_state = [0, 1]


		noisy_pos = focused_pos

		obj_pos_3d = encoded_pos_3d
		if gripper_state == [0, 1] and cfg.task != 'scooping':
			obj_pos_3d[:3] += action[:3]/1000
			pos_2d = (obj_pos_3d[3*len(focused_pos)//2:3*len(focused_pos)//2+2] + 1)/2*1080
			obj_pos_3d[3*len(focused_pos)//2:3*len(focused_pos)//2+2] = (pos_2d + action_2d)*2/1080 - 1

		if (np.abs(np.array(action)) > 0.0).any():
			sa_pairs.append(state[:3] + quat.tolist() + gripper_state + noisy_pos.tolist() + obj_pos_3d.tolist() + action.tolist() + [gripper_action] + [var[idx]])
		else:
			print(action)
		
		if idx == len(pred)-2:
			for _ in range(10):
				sa_pairs.append(state[:3] + quat.tolist() + gripper_state + noisy_pos.tolist() + obj_pos_3d.tolist() + [0.]*6 + [gripper_action] + [1.0])

	print(np.shape(sa_pairs))
	json.dump(sa_pairs, open('data/{}/{}/demo_{}.json'.format(cfg.task, cfg.alg, demo_num), 'w'))

def wrap_angular_actions(theta):
	if abs(theta[0]) > np.pi:
		theta[0] -= 2*np.sign(theta[0])*np.pi

	if abs(theta[1]) > np.pi:
		theta[1] -= 2*np.sign(theta[1])*np.pi

	if abs(theta[2]) > np.pi:
		theta[2] -= 2*np.sign(theta[2])*np.pi

	return theta


def wrap_angles(theta):
	if abs(theta) > np.pi:
		theta -= 2*np.sign(theta)*np.pi
	return theta
