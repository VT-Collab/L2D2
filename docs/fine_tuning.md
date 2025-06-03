This file provides implementatation details and instrutions to collect physical demonstrations on the robot and leveraging them to fine-tune the reconstruction model and the robot's initial policy.


## Fine-Tuning Robot Policy

### Collecting Physical Demonstrations
Run the following commands in sequence to teleoperate the robot and collect physical demonstrations

- In one terminal window navigate to `L2D2/` and run
```
python3 main_robot.py task=<task> alg=l2d2 corr=True corr_num=0
```

- In another terminal window or device connected to your robot navigate to `robot/`, initialize your robot driver and run
```
python3 teleop.py
```

These demonstrations will be saved in the form of a sequence of images and robot trajectories in `L2D2/data/<task>/l2d2/<corr_num>/` and `L2D2/data/<task>/l2d2/` respectively. 

### Processing the Demos and Updating the Reconstruction Model
Once you have collected a few physical demonstrations, run the following to process these demos to extract the dataset of state-action pairs that can be used to fine-tune the policy:
```
python3 main_robot.py task=<task> alg=l2d2 process=True corr=True get_obj=True
```
This command will fine-tune the reconstruction function based on the task-specific demonstrations, refine the drawings provided by the user usign the updated reconstruction function and will save the physical corrections in `L2D2/<task>/l2d2/demo_c_<corr_num>.json`.

### Fine-tuning the Robot Policy
Next, follow the two-setp process to fine-tune the robot's policy:
```
python3 main_robot.py task=<task> alg=l2d2 train_il=True
python3 main_robot.py task=<task> alg=l2d2 train_il=True fine_tune=True
```

This will train an ensemble of initial policies using the refined drawing demonstrations and fine-tune the ensemble using the physical demonstrations. The fine-tuned policies will be saved in `L2D2/data/<task>/l2d2/model_<index>_ft.pt`

### Evaluation of the Fine-tuned Policy
To evaluate the fine-tuned policy run the following sequence of commands:

- In one terminal window, navigate to `L2D2' and run 
```
python3 main_robot.py task=<task> alg=l2d2 eval=True fine_tune=True
```

- In another terminal window connected to the robot, navigate to `robot/` and run
```
python3 teleop.py
```
Follow the instructions on the screen to start evaluation of the fine-tuned robot policy for the task.
