import shutil
import subprocess
from roboflow import Roboflow
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

rf = Roboflow(api_key="I3Vbj9jDhiNN16f4T8PX")
project = rf.workspace("roboflow-jvuqo").project("football-players-detection-3zvbc")
version = project.version(1)
dataset = version.download("yolov8")

# def move_to_subfolder(dst):
#     shutil.move("football-players-detection-1/"+ dst, "football-players-detection-1/football-players-detection-1/"+ dst)

# move_to_subfolder("train")
# move_to_subfolder("test")
# move_to_subfolder("valid")

train_command = f"yolo task=detect model=yolov8m.pt data={dataset.location}/data.yaml epochs=25 imgsz=640 device=cuda"
subprocess.run(train_command, shell=True)