import shutil
import os
from roboflow import Roboflow

data_path = "data/licence-plate"

# Check if the target directory exists, if not, create it
if not os.path.exists(data_path):
    os.makedirs(data_path)
    
rf = Roboflow(api_key="dOBeh4ZsKyPLpCpPBk6V")
project = rf.workspace("mazen-alwaqdei").project("plate-license-dvt6r")
dataset = project.version(5).download("yolov8", data_path)


