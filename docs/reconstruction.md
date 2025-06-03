Below, we provide implementation details for collecting data for and training the reconstruction model that maps the users 2d drawings to robot's actions in 3d space.

## Training the reconstruction model $f_\phi$
We assume that the camera is placed in an approximately optimal location to minimize information loss.

### Data Collection
First, we collect the dataset $\mathcal{D}_{map}$. We collect this data physically to restrict the datapoints in the robot's workspace $\mathcal{W}$, where the tasks will be carried out. To track the 2D position of the robot's end-effector in the image, we attach an aruco marker (marker_id = $75$ and dimensions=$50\times50$) to the robot's end-effector. 

Follow the steps in sequence and follow instructions on the screen to collect $\mathcal{D}_{map}$ by teleoperating the robot using a joystick

- In one terminal window, navigate to `L2D2/` and run:
```
python3 main_3dmap.py task=play get_demo=True
```
If you already have a few demos, you can provide `demo_num=<demo_num>` argument to start the demo collection. 

- In another terminal window or computer connected to the robot (UR-10 in our case) initialize the robot controller, navigate to `robot/` and run:
```
python3 teleop.py
```

The demos collected will be saved in `L2D2/data/play/<demo_num>/`. Each demo folder will have an image of the environment `env.png`, an image with the 2d trjecory of the robot `2d_traj.png`, the 3d trajectory of the robot `traj.json` and the 2d trajectory `img_traj.json`. 

You can stop the data collection when you have around $\sim 5000 - 6000$ datapoints in the demonstrations. Note that these demos should not be task specific, and should involve random robot movements covering the robot's workspace.

### Training the Reconstruction Model

To train the reconstruction function, you can run the following command:
```
python3 main_3dmap.py task=play train=True
```
This command will train an ensemble of $5$ reconstruction models and save them in `L2D2/data/play/` with model name `model_<model_num>.pt`.
