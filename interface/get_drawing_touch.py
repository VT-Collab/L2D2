import pygame
import sys
import time
import pickle
import json
import numpy as np
import os
import socket



def get_2d_demos(conn, camera, task): 

	os.environ['SDL_VIDEO_WINDOW_POS'] = "5,25"
	
	pygame.init()
	X = 1365
	Y = 810

	IMAGE_PARTITION = 1080
	BUTTON_PARTITION = 260
	SLIDER_PARTITION = 280

	SLIDER_START_X = IMAGE_PARTITION
	BUTTON_START_X = SLIDER_START_X + 10
	

	screen = pygame.display.set_mode((X, Y))
	pygame.display.set_caption('image')

	img = pygame.image.load('robot_img.png').convert()

	pygame.mouse.set_visible(False)

	if task != 'scooping':
		pointer = pygame.image.load('pointer_{}.png'.format(camera))
		if camera == 0:
			pointer = pygame.transform.scale(pointer, (120, 160)).convert_alpha()
		
		if camera == 2:
			pointer = pygame.transform.scale(pointer, (105, 160)).convert_alpha()
	else:
		pointer = pygame.image.load('scooping_{}.png'.format(camera))
		if camera == 0:
			pointer = pygame.transform.scale(pointer, (85, 230)).convert_alpha()
		
		if camera == 2:
			pointer = pygame.transform.scale(pointer, (75, 240)).convert_alpha()

	pointer_rect = pointer.get_rect()

	slider_width = 40
	slider_height = 350
	slider_color = (255, 255, 255)
	slider_button_radius = 20
	slider_button_color = (0, 0, 255)
	slider_spacing = (SLIDER_PARTITION - 2 * slider_width) // 4

	slider_x_x = SLIDER_START_X + slider_width //2
	slider_x_y = 450
	slider_x_value = 0.


	slider_y_x = slider_x_x + slider_width + slider_spacing
	slider_y_y = 450
	slider_y_value = 0.


	slider_z_x = slider_y_x + slider_width + slider_spacing
	slider_z_y = 450
	slider_z_value = 0.

	rx, ry, rz = 0., 0., 0.
	dx, dy, dz = 0., 0., 0.
	DX, DY, DZ = 0., 0., 0.

	# LISTS
	rot_starts = []
	rot_ends = []
	current_start_points = []
	current_end_points = []
	current_start_angles = []
	start_angles = []
	stop_angles = []
	gripper_open = []
	gripper_close = []
	current_gripper_open = []
	current_gripper_close = []
	traj = []
	
	# FLAGS
	set_start = False
	set_end = False
	current_start_idx = None
	set_gripper_open = False
	set_gripper_close = False
	draw = False

	# BUTTON DIMESIONS
	button_width = int(BUTTON_PARTITION * 0.45)
	button_height = 80
	button_spacing = 25
	big_button_width = int(BUTTON_PARTITION * 1.0067)
	big_button_height = 80

	# DRAW BUTTON
	draw_button_x = BUTTON_START_X + (BUTTON_PARTITION - big_button_width) //2
	draw_button_y = 10
	draw_button_color = (0, 255, 0)
	draw_button_text = "Start"

	# ANGLE START BUTTON
	angle_start_button_x = BUTTON_START_X + (BUTTON_PARTITION - 2*button_width - button_spacing) //2
	angle_start_button_y = draw_button_y + big_button_height + button_spacing
	angle_start_button_color = (42, 143, 189)
	angle_start_button_text = "Start Angle"

	# ANGLE END BUTTON
	angle_end_button_x =  angle_start_button_x + button_width + button_spacing
	angle_end_button_y = angle_start_button_y
	angle_end_button_color = (42, 143, 189)
	angle_end_button_text = "End Angle"

	# GRIPPER OPEN BUTTON
	gripper_open_button_x = BUTTON_START_X + (BUTTON_PARTITION - 2*button_width - button_spacing) //2
	gripper_open_button_y = angle_end_button_y + big_button_height + button_spacing
	gripper_open_button_color = (141, 95, 211)
	gripper_open_button_text = "Gripper Open"

	# GRIPPER CLOSE BUTTON
	gripper_close_button_x = gripper_open_button_x + button_width + button_spacing
	gripper_close_button_y = gripper_open_button_y
	gripper_close_button_color = (141, 95, 211)
	gripper_close_button_text = "Gripper Close"

	# SAVE BUTTON
	save_button_x = draw_button_x
	save_button_y = gripper_close_button_y + button_height + button_spacing
	save_button_color = (255, 153, 0)
	save_button_text = "Save"



	font = pygame.font.Font(None, 25)
	screen.blit(img, (0, 0))
	pygame.display.flip()
	touch_pos = (0, 0)

	while True:
		
		pointer_rect.center = touch_pos
		pointer_rect.centerx -= 10
		pointer_rect.centery += 45
		screen.fill([0, 0, 0])
		screen.blit(img, (0, 0))

		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				return("done")
				pygame.quit()
				sys.exit()

			
			if event.type == pygame.FINGERDOWN:
				touch_x = event.x * X
				touch_y = event.y * Y
				touch_pos = (touch_x, touch_y)

				# DRAW BUTTON EVENT
				if (draw_button_x <= touch_x <= draw_button_x + big_button_width and
					draw_button_y <= touch_y <= draw_button_y + button_height) and not draw:
					draw = True 
					draw_button_color = (255, 0, 0)
					draw_button_text = "Stop"
					time.sleep(0.2)
				elif (draw_button_x <= touch_x <= draw_button_x + big_button_width and
					draw_button_y <= touch_y <= draw_button_y + button_height) and draw:
					draw = False
					draw_button_color = (0, 255, 0)
					draw_button_text = "Start"
					time.sleep(0.2)

				# ANGEL START AND END BUTTON EVENT
				if (angle_start_button_x <= touch_x <= angle_start_button_x + button_width and
					angle_start_button_y <= touch_y <= angle_start_button_y + button_height and not draw and not set_start and not set_gripper_open and not set_gripper_close):
					set_start = True
					angle_start_button_color = (255, 255, 0)
					angle_start_button_text = "Selecting"
					print("[*] Select Start points for rotation")
					time.sleep(0.2)
				elif (angle_start_button_x <= touch_x <= angle_start_button_x + button_width and
					angle_start_button_y <= touch_y <= angle_start_button_y + button_height and not draw and set_start and not set_gripper_open and not set_gripper_close):
					set_start = False
					angle_start_button_color = (42, 143, 189)
					angle_start_button_text = "Set Start"
					if current_start_points:
						rot_starts.extend(current_start_points)
						start_angles.extend(current_start_angles)
					current_start_points = []
					current_start_angles = []
					slider_button_color = (0, 0, 255)
					print("[*] Start Points set for rotation")
					time.sleep(0.2)
				if (angle_end_button_x <= touch_x <= angle_end_button_x + button_width and angle_end_button_y <= touch_y <= angle_end_button_y + button_height and not draw and not set_start and not set_end and not set_gripper_open and not set_gripper_close and current_start_idx is not None):
					set_end = True
					angle_end_button_color = (255, 255, 0)
					angle_end_button_text = "Selecting"
					print("[*] Select End points for rotation")
					time.sleep(0.2)
				elif (angle_end_button_x <= touch_x <= angle_end_button_x + button_width and angle_end_button_y <= touch_y <= angle_end_button_y + button_height and not draw and not set_start and set_end and not set_gripper_open and not set_gripper_close):
					set_end = False
					angle_end_button_color = (42, 143, 189)
					angle_end_button_text = "Set End"
					if current_end_points:
						rot_ends.extend(current_end_points)
						stop_angles.extend([[start_angles[-1][0] + dx, start_angles[-1][1] + dy, start_angles[-1][2] + dz]])
						DX += dx
						DY += dy
						DZ += dz
						dx, dy, dz = 0, 0, 0
						slider_x_value = slider_y_value = slider_z_value = 0.
						current_start_idx = None
					current_end_points = []
					print("[*] End points set for rotation")
					time.sleep(0.2)
			
				# GRIPPER OPEN AND CLOSE BUTTON EVENT
				if (gripper_close_button_x <= touch_x <= gripper_close_button_x + button_width and gripper_close_button_y <= touch_y <= gripper_close_button_y + button_height and not draw and not set_start and not set_end and not set_gripper_open and not set_gripper_close):
					set_gripper_close = True
					gripper_close_button_color = (255, 255, 0)
					gripper_close_button_text = "Selecting"
					print("[*] Start close points for gripper")
					time.sleep(0.2)
				elif (gripper_close_button_x <= touch_x <= gripper_close_button_x + button_width and gripper_close_button_y <= touch_y <= gripper_close_button_y + button_height and not draw and not set_start and not set_end and not set_gripper_open and set_gripper_close):
					set_gripper_close = False
					gripper_close_button_color = (141, 95, 211)
					gripper_close_button_text = "Gripper Close"
					if current_gripper_close:
						gripper_close.extend(current_gripper_close)
					current_gripper_close = []
					print("[*] End point set for gripper")
					time.sleep(0.2)
				if (gripper_open_button_x <= touch_x <= gripper_open_button_x + button_width and gripper_open_button_y <= touch_y <= gripper_open_button_y + button_height and not draw and not set_start and not set_end and not set_gripper_open and not set_gripper_close):
					set_gripper_open = True
					gripper_open_button_color = (255, 255, 0)
					gripper_open_button_text = "Selecting"
					print("[*] Start open points for gripper")
					time.sleep(0.2)
				elif (gripper_open_button_x <= touch_x <= gripper_open_button_x + button_width and gripper_open_button_y <= touch_y <= gripper_open_button_y + button_height and not draw and not set_start and not set_end and set_gripper_open and not set_gripper_close):
					set_gripper_open = False
					gripper_open_button_color = (141, 95, 211)
					gripper_open_button_text = "Gripper Start"
					if current_gripper_open:
						gripper_open.extend(current_gripper_open)
					current_gripper_open = []
					print("[*] Open point set for gripper")
					time.sleep(0.2)

				# SAVE BUTTON EVENT
				if (save_button_x <= touch_x <= save_button_x + big_button_width and save_button_y <= touch_y <= save_button_y + button_height and not draw):

					if len(rot_ends) > 0 and len(rot_starts) > 0:
						traj_rotatations = np.zeros((len(traj), 3))
												
						for idx in range(len(traj)):
							traj[idx] = list(traj[idx]) + traj_rotatations[idx].tolist()
							
						for i in range(len(rot_starts)):
							start_idx = rot_starts[i]
							end_idx = rot_ends[i]
							start_angle = start_angles[i]
							stop_angle = stop_angles[i]
							rot_traj = np.linspace(start_angle, stop_angle, abs(25)).tolist()

							init_traj = traj [:end_idx]
							final_traj = traj[end_idx+1:]
							rot = [traj[end_idx]]

							for idx in range(len(rot_traj)):
								wp = rot[idx][:2] + rot_traj[idx]
								rot += [wp]

							for idx in range(len(init_traj)):
								init_traj[idx][2:] = start_angle
								
							for idx in range(len(final_traj)):
								final_traj[idx][2:] = stop_angle


						traj = init_traj + rot + final_traj

							# for idx in range(len(traj)):
							# 	if idx < end_idx:
							# 		traj_rotatations[idx] += start_angle
							# 	elif start_idx <= idx < end_idx:
							# 		traj_rotatations[idx] += rot_traj[idx - start_idx]
							# 	else:
							# 		traj_rotatations[idx] += stop_angle
						
						# for idx in range(len(traj)):
						# 	traj[idx] = list(traj[idx]) + traj_rotatations[idx].tolist()
					else:
						for idx in range(len(traj)):
							traj[idx] = list(traj[idx]) + [0.0, 0.0, 0.0]

					gripper_traj = -np.ones((len(traj), 1))
					for i in range(len(gripper_close)):
						gripper_traj[gripper_close[i]:] = 1
						for j in range(len(gripper_open)):
							if gripper_close[i]<gripper_open[j]:
								gripper_traj[gripper_open[j]:] = -1
								break
					for idx in range(len(traj)):
						traj[idx] = list(traj[idx]) + gripper_traj[idx].tolist()

					data = pickle.dumps((-DX, -DY, -DZ), protocol = 2)
					conn.send(data)
					
					print("Done")
					pygame.quit()
					time.sleep(0.5)
					return traj
					


			if event.type == pygame.FINGERMOTION:
				touch_x = event.x * X
				touch_y = event.y * Y
				touch_pos = (touch_x, touch_y)

		# ANGLE POINTS AND ANGLE CALCULATIONS
		if set_start or set_end:
			if touch_x <= 1080 and touch_y <= 810:
				if event.type == pygame.FINGERMOTION:
					if set_start:
						current_start_idx = np.linalg.norm(np.asarray(traj - np.array(touch_pos)), axis=1).argmin()
						current_start_points = [current_start_idx]
						current_start_angles = [[slider_x_value*360, slider_y_value*360, slider_z_value*360]]
						time.sleep(0.5)
					elif set_end and current_start_idx is not None:
						end_idx = np.linalg.norm(np.asarray(traj - np.array(touch_pos)), axis=1).argmin()
						if end_idx> current_start_idx:
							current_end_points = [end_idx]
							time.sleep(0.5)
			else: 
				data = pickle.dumps((0.0, 0.0, 0.0), protocol=2)
				if event.type == pygame.FINGERMOTION:
					if slider_x_x <= touch_pos[0] <= slider_x_x + slider_width and slider_x_y <= touch_pos[1] <= slider_x_y + slider_height:
						slider_x_value = (touch_pos[1] - slider_x_y) / slider_height
						rx = 5*(2*slider_x_value - 1)
						dx += rx
						# print(dx)
						data = pickle.dumps((np.round(rx, 2), 0.0, 0.0), protocol=2)

					if slider_y_x <= touch_pos[0] <= slider_y_x + slider_width and slider_y_y <= touch_pos[1] <= slider_y_y + slider_height:
						slider_y_value = (touch_pos[1] - slider_y_y) / slider_height
						ry = 5*(2*slider_y_value - 1)
						dy += ry
						data = pickle.dumps((0.0, np.round(ry, 2), 0.0), protocol=2)
					
					if slider_z_x <= touch_pos[0] <= slider_z_x + slider_width and slider_z_y <= touch_pos[1] <= slider_z_y + slider_height:
						slider_z_value = (touch_pos[1] - slider_z_y) / slider_height
						rz = 5*(2*slider_z_value - 1)
						dz += rz
						data = pickle.dumps((0.0, 0.0, np.round(rz, 2)), protocol=2)
				conn.send(data)
				time.sleep(0.1)

		# GRIPPER POINTS CALCULATIONS
		if set_gripper_open or set_gripper_close:
			if  touch_x <= 1080 and touch_y <= 810:
				if event.type == pygame.FINGERMOTION:
					if set_gripper_open:
						gripper_open_pts = np.linalg.norm(np.asarray(traj - np.array(touch_pos)), axis=1).argmin()
						current_gripper_open = [gripper_open_pts]
						time.sleep(0.5)

					else:
						gripper_close_pts = np.linalg.norm(np.asarray(traj - np.array(touch_pos)), axis=1).argmin()
						current_gripper_close = [gripper_close_pts]
						time.sleep(0.5)
			

		if draw and event.type == pygame.FINGERMOTION:
			if 0<= touch_pos[0] <= 1080 and 0 <= touch_pos[1] <= 810:
				traj.append(list(touch_pos))
				time.sleep(0.01)

		if len(traj) > 1:
			for idx in range(1, len(traj)):
				pygame.draw.line(screen, (255, 0, 0), traj[idx-1][:2], traj[idx][:2], 2)

		
		# DRAWING CODE
		pygame.draw.rect(screen, draw_button_color, (draw_button_x, draw_button_y, big_button_width, button_height))
		text = font.render(draw_button_text, True, (0, 0, 0))
		text_rect = text.get_rect(center = (draw_button_x + big_button_width/2, draw_button_y + button_height/2))
		screen.blit(text, text_rect)

		pygame.draw.rect(screen, save_button_color, (save_button_x, save_button_y, big_button_width, button_height))
		text = font.render(save_button_text, True, (0, 0, 0))
		text_rect = text.get_rect(center = (save_button_x + big_button_width/2, save_button_y + button_height/2))
		screen.blit(text, text_rect)

		pygame.draw.rect(screen, angle_start_button_color, (angle_start_button_x, angle_start_button_y, button_width, button_height))
		text = font.render(angle_start_button_text, True, (0, 0, 0))
		text_rect = text.get_rect(center = (angle_start_button_x + button_width/2, angle_start_button_y + button_height/2))
		screen.blit(text, text_rect)

		pygame.draw.rect(screen, angle_end_button_color, (angle_end_button_x, angle_end_button_y, button_width, button_height))
		text = font.render(angle_end_button_text, True, (0, 0, 0))
		text_rect = text.get_rect(center = (angle_end_button_x + button_width/2, angle_end_button_y + button_height/2))
		screen.blit(text, text_rect)

		pygame.draw.rect(screen, gripper_open_button_color, (gripper_open_button_x, gripper_open_button_y, button_width, button_height))
		text = font.render(gripper_open_button_text, True, (0, 0, 0))
		text_rect = text.get_rect(center = (gripper_open_button_x + button_width/2, gripper_open_button_y + button_height/2))
		screen.blit(text, text_rect)

		pygame.draw.rect(screen, gripper_close_button_color, (gripper_close_button_x, gripper_close_button_y, button_width, button_height))
		text = font.render(gripper_close_button_text, True, (0, 0, 0))
		text_rect = text.get_rect(center = (gripper_close_button_x + button_width/2, gripper_close_button_y + button_height/2))
		screen.blit(text, text_rect)
		

		pygame.draw.rect(screen, slider_color, (slider_x_x, slider_x_y, slider_width, slider_height))
		button_x_y = slider_x_y + int(slider_x_value * slider_height)
		pygame.draw.circle(screen, slider_button_color if set_start and set_end else (150, 150, 150), (slider_x_x + slider_width // 2, button_x_y), slider_button_radius)

		pygame.draw.rect(screen, slider_color, (slider_y_x, slider_y_y, slider_width, slider_height))
		button_y_y = slider_y_y + int(slider_y_value * slider_height)
		pygame.draw.circle(screen, slider_button_color if set_start and set_end else (150, 150, 150), (slider_y_x + slider_width // 2, button_y_y), slider_button_radius)

		pygame.draw.rect(screen, slider_color, (slider_z_x, slider_z_y, slider_width, slider_height))
		button_z_y = slider_z_y + int(slider_z_value * slider_height)
		pygame.draw.circle(screen, slider_button_color if set_start and set_end else (150, 150, 150), (slider_z_x + slider_width // 2, button_z_y), slider_button_radius)


		# DISPLAY CODE
		if touch_pos[0] < 1080:
			screen.blit(pointer, pointer_rect)

		for start_point in rot_starts:
			pygame.draw.circle(screen, (105, 185, 220), traj[start_point][:2], 5)
		for start_point in current_start_points:
			pygame.draw.circle(screen, (179, 179, 179), traj[start_point][:2], 5)

		for end_point in rot_ends:
			pygame.draw.circle(screen, (5, 75, 145), traj[end_point][:2], 5)
		for end_point in current_end_points:
			pygame.draw.circle(screen, (179, 179, 179), traj[end_point][:2], 5)

		for open_point in gripper_open:
			pygame.draw.circle(screen, (155, 105, 230), traj[open_point][:2], 5)
		for open_point in current_gripper_open:
			pygame.draw.circle(screen, (179, 179, 179), traj[open_point][:2], 5)
		
		for close_point in gripper_close:
			pygame.draw.circle(screen, (115, 40, 239), traj[close_point][:2], 5)
		for close_point in current_gripper_close:
			pygame.draw.circle(screen, (179, 179, 179), traj[close_point][:2], 5)

		pygame.display.update()



# get_2d_demos()