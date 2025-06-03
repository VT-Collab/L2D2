Here we provide implementation details for collecting drawings from users and training an initial robot policy using the 3d demonstrations reconstructed from these drawings.


## Learning from Drawings

### Collecting Drawings
To set your object manipulation region in the image, set the `roi` and `roi_size` arguments in `L2D2/cfg/config.yaml` to the desired region of interest in your camera frame ($1080p \times 810p$).

To collect drawings, run the following set of commands in sequence

- In one terminal window, navigate to `L2D2/` and run:
```
python3 main_robot.py task=<task> get_demo=True get_img=True get_prompts=True alg=l2d2 demo_num=0
```
Follow the prompts on the screen to get the initial environment image and the objects of interest. Once you have provided these inputs, you can set `get_img` and `get_prompts` arguments to $False$ for the task that you are trying to teach.

- In a terminal window on your touch screen device, navigate to `interface/` and run:
```
./run.bat
```
You can now start providing drawings to convey your desired task to the robot using our drawing interface. 

Your drawings, and the processed 3d demos will be saved in `L2D2/data/<task>/l2d2/<demo_num>.json` and `L2D2/data/<task>/l2d2/demo_<demo_num>.json` respectively. 

### Training an initial policy
Once you are done collecting the demos run the following command to train an initial policy for the task
```
python3 main_robot.py task=<task> alg=l2d2 train_il=True
```
This will train an ensemble of $5$ policies and save them in `L2D2/data/<task>/l2d2/model_<index>.pt`.

### Evlauating the initial policy
To evaluate this model, you need to run the following scripts:

- In one terminal window, navigate to `L2D2/` and run:
```
python3 main_robot.py task=<task> alg=l2d2 eval=True
```

- In another terminal window or device connected to your robot navigate to `robot/`, initialize your robot driver and run
```
python3 teleop.py
```
This will roll-out the initial policy trained on only drawing data.

