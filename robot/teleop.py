#!/usr/bin/env python

import rospy
import actionlib
import sys
import time
import numpy as np
import pygame
import socket
import pickle
from urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_kinematics import KDLKinematics
import copy
from collections import deque
from scipy.spatial.transform import Rotation

from std_msgs.msg import Float64MultiArray

from robotiq_2f_gripper_msgs.msg import (
    CommandRobotiqGripperFeedback, 
    CommandRobotiqGripperResult, 
    CommandRobotiqGripperAction, 
    CommandRobotiqGripperGoal
)

from robotiq_2f_gripper_control.robotiq_2f_gripper_driver import (
    Robotiq2FingerGripperDriver as Robotiq
)

from controller_manager_msgs.srv import (
    SwitchController, 
    SwitchControllerRequest, 
    SwitchControllerResponse
)

from control_msgs.msg import (
    FollowJointTrajectoryAction,
    FollowJointTrajectoryGoal,
    GripperCommandAction,
    GripperCommandGoal,
    GripperCommand
)
from trajectory_msgs.msg import (
    JointTrajectoryPoint
)
from sensor_msgs.msg import (
    JointState
)
from geometry_msgs.msg import(
    TwistStamped,
    Twist
)

# HOME = [-1.219569508229391, -0.9974325338946741, -2.3463192621814173, -1.3549531141864222, 1.5623568296432495, 0.363106667995452]
HOME = [-np.pi/2, -np.pi/4, -2*np.pi/3, -1.8326, np.pi/2, 0.0]

STEP_SIZE_L = 0.15
STEP_SIZE_A = 0.2 * np.pi / 4
STEP_TIME = 0.01
DEADBAND = 0.1 
MOVING_AVERAGE = 2

class JoystickControl(object):

    def __init__(self):
        pygame.init()
        self.gamepad = pygame.joystick.Joystick(0)
        self.gamepad.init()
        self.toggle = False
        self.action = None
        self.A_pressed = False
        self.B_pressed = False

    def getInput(self):
        pygame.event.get()
        toggle_angular = self.gamepad.get_button(4)
        toggle_linear = self.gamepad.get_button(5)
        self.A_pressed = self.gamepad.get_button(0)
        self.B_pressed = self.gamepad.get_button(1)
        if not self.toggle and toggle_angular:
            self.toggle = True
        elif self.toggle and toggle_linear:
            self.toggle = False
        return self.getEvent()

    def getEvent(self):
        z1 = self.gamepad.get_axis(0)
        z2 = self.gamepad.get_axis(1)
        z3 = self.gamepad.get_axis(4)
        z = [z1, z2, z3]
        for idx in range(len(z)):
            if abs(z[idx]) < DEADBAND:
                z[idx] = 0.0
        stop = self.gamepad.get_button(7)
        gripper_open = self.gamepad.get_button(1)
        gripper_close = self.gamepad.get_button(0)
        return tuple(z), (gripper_open, gripper_close), stop

    def getAction(self, z):
        if self.toggle:
            self.action = (0, 0, 0, STEP_SIZE_A * -z[1], STEP_SIZE_A * -z[0], STEP_SIZE_A * -z[2])
        else:
            self.action = (STEP_SIZE_L * -z[1], STEP_SIZE_L * -z[0], STEP_SIZE_L * -z[2], 0, 0, 0)

class TrajectoryClient(object):

    def __init__(self):
        # Action client for joint move commands
        self.client = actionlib.SimpleActionClient(
                '/scaled_pos_joint_traj_controller/follow_joint_trajectory',
                FollowJointTrajectoryAction)
        self.client.wait_for_server()
        # Velocity commands publisher
        self.vel_pub = rospy.Publisher('/joint_group_vel_controller/command',\
                 Float64MultiArray, queue_size=10)
        # Subscribers to update joint state
        self.joint_sub = rospy.Subscriber('/joint_states', JointState, self.joint_states_cb)
        # service call to switch controllers
        self.switch_controller_cli = rospy.ServiceProxy('/controller_manager/switch_controller',\
                 SwitchController)
        self.joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",\
                            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        self.base_link = "base_link"
        self.end_link = "wrist_3_link"
        self.joint_states = None
        self.robot_urdf = URDF.from_parameter_server()
        self.kdl_kin = KDLKinematics(self.robot_urdf, self.base_link, self.end_link)
        
        # Gripper action and client
        action_name = rospy.get_param('~action_name', 'command_robotiq_action')
        self.robotiq_client = actionlib.SimpleActionClient(action_name, \
                                CommandRobotiqGripperAction)
        self.robotiq_client.wait_for_server()
        # Initialize gripper
        goal = CommandRobotiqGripperGoal()
        goal.emergency_release = False
        goal.stop = False
        goal.position = 1.00
        goal.speed = 0.1
        goal.force = 5.0
        # Sends the goal to the gripper.
        self.robotiq_client.send_goal(goal)

        # store previous joint vels for moving avg
        self.qdots = deque(maxlen=MOVING_AVERAGE)
        for idx in range(MOVING_AVERAGE):
            self.qdots.append(np.asarray([0.0] * 6))

    def joint_states_cb(self, msg):
        try:
            if msg is not None:
                states = list(msg.position)
                states[2], states[0] = states[0], states[2]
                self.joint_states = tuple(states) 
        except:
            pass
    
    def switch_controller(self, mode=None):
        req = SwitchControllerRequest()
        res = SwitchControllerResponse()

        req.start_asap = False
        req.timeout = 0.0
        if mode == 'velocity':
            req.start_controllers = ['joint_group_vel_controller']
            req.stop_controllers = ['scaled_pos_joint_traj_controller']
            req.strictness = req.STRICT
        elif mode == 'position':
            req.start_controllers = ['scaled_pos_joint_traj_controller']
            req.stop_controllers = ['joint_group_vel_controller']
            req.strictness = req.STRICT
        else:
            rospy.logwarn('Unkown mode for the controller!')

        res = self.switch_controller_cli.call(req)

    def xdot2qdot(self, xdot):
        J = self.kdl_kin.jacobian(self.joint_states)
        J_inv = np.linalg.pinv(J)
        return J_inv.dot(xdot)

    def joint2pose(self, joint_state=None):
        if joint_state is None:
            joint_state = self.joint_states
        state = self.kdl_kin.forward(joint_state)
        xyz_lin = np.array(state[:,3][:3]).T
        xyz_lin = xyz_lin.tolist()
        R = state[:,:3][:3]
        r = Rotation.from_dcm(R)
        beta = -np.arcsin(R[2,0])
        alpha = np.arctan2(R[2,1]/np.cos(beta),R[2,2]/np.cos(beta))
        gamma = np.arctan2(R[1,0]/np.cos(beta),R[0,0]/np.cos(beta))
        xyz_ang = [alpha, beta, gamma]
        quat = r.as_quat() 
        xyz = np.asarray(xyz_lin[-1]).tolist() + np.asarray(quat).tolist()
        # print(r.as_euler('xyz', degrees=False))
        # print(xyz_ang)
        return xyz

    def send(self, xdot):
        qdot = self.xdot2qdot(xdot)
        # Moving avg over last N input velocities
        self.qdots = np.delete(self.qdots, 0, 0)
        qdot = np.array(qdot)
        self.qdots = np.vstack((self.qdots, qdot))
        # qdot = qdot.tolist()[0]
        qdot_mean = np.mean(self.qdots, axis=0).tolist()
        # state = self.joint2pose()
        cmd_vel = Float64MultiArray()
        cmd_vel.data = qdot_mean
        self.vel_pub.publish(cmd_vel)

    def send_joint(self, pos, time):
        waypoint = JointTrajectoryPoint()
        waypoint.positions = pos
        waypoint.time_from_start = rospy.Duration(time)
        goal = FollowJointTrajectoryGoal()
        goal.trajectory.joint_names = self.joint_names
        goal.trajectory.points.append(waypoint)
        goal.trajectory.header.stamp = rospy.Time.now()
        self.client.send_goal(goal)
        rospy.sleep(time)

    def actuate_gripper(self, pos, speed, force):
        Robotiq.goto(self.robotiq_client, pos=pos, speed=speed, force=force, block=True)
        return self.robotiq_client.get_result()

def main():
    rospy.init_node("teleop")
    
    print('connecting to robot')
    mover = TrajectoryClient()
    print('connected\n')
    
    # Initialize Socket for Robot Control
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = ('172.16.0.3', 10982)
    print("connecting to {}, {}".format(*server_address))
    sock.connect(server_address)
    sock.settimeout(1000)
    # print("Connection Established")

    start_time = time.time()
    rate = rospy.Rate(100)

    print("[*] Initialized, Moving Home")
    mover.switch_controller(mode='position')
    mover.send_joint(HOME, 4.0)
    mover.client.wait_for_result()
    mover.switch_controller(mode='velocity')
    print("[*] Ready for joystick inputs")
    gripper_open = True
    gripper_close = False
    while not rospy.is_shutdown():
        t_curr = time.time() - start_time
        z = sock.recv(1024)
        actions = pickle.loads(z)
        
        if actions == "stop":
            #pickle.dump(data, open(filename, "wb"))
            # sock.close()
            # print(mover.joint2pose())
            return False
        
        if actions == "home":
            print("[*] Initialized, Moving Home")
            mover.switch_controller(mode='position')
            mover.send_joint(HOME, 4.0)
            mover.client.wait_for_result()
            mover.switch_controller(mode='velocity')
            print("[*] Ready for joystick inputs")
            continue
        
        gripper = actions[6]
        action = actions[:6]
        # action[3:] = [0., 0., 0.]
        # print(action)
        state = mover.joint2pose()[:7]
        if gripper_open:
            state.append(1)
            state.append(0)
        else:
            state.append(0)
            state.append(1)
        # print(state)
        sock.send(pickle.dumps(state))

        
        print(actions)
        mover.send(action)
        if gripper<0.5 and gripper_close:
            mover.actuate_gripper(0.15, 0.1, 10)
            gripper_close = False
            gripper_open = True
        if gripper>0.5 and gripper_open and gripper!=0.0:
            mover.actuate_gripper(0., 0.1, 10)
            gripper_open = False
            gripper_close = True
            
        # if joystick.A_pressed:
        #     mover.actuate_gripper(0.05, 0.1, 10)
        # elif joystick.B_pressed:
            # mover.actuate_gripper(0., 0.1, 10)

        
        # print(mover.joint2pose(mover.joint_states))
        
        rate.sleep()

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass






















