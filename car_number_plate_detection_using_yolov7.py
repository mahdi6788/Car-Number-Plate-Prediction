import os
import shutil

"""## Kaggle settings"""

# Installing kaggle
!pip install kaggle

# Upload JSON file got from Kaggle
from google.colab import files
files.upload()

!mkdir ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Dataset API
!kaggle datasets download -d skhalili/iraniancarnumberplate

"""## Data Preprocessing"""

# Extract the Row data file to the saving destination (Google Drive)
import zipfile
# Specify the path to the downloaded zip file
zip_path = "/content/iraniancarnumberplate.zip"
# Extract the contents of the zip file
with zipfile.ZipFile(zip_path, "r") as zip_ref:
  zip_ref.extractall("/content/drive/MyDrive/Colab Notebooks/ComputerVision/Digit_Classification/Persian/CarNumberPlate/iraniancarnumberplate")

cars_path = "/content/drive/MyDrive/Colab Notebooks/ComputerVision/Digit_Classification/Persian/CarNumberPlate/iraniancarnumberplate/car"

train_images_path = "/content/drive/MyDrive/Colab Notebooks/ComputerVision/Digit_Classification/Persian/CarNumberPlate/train/images"
train_labels_path = "/content/drive/MyDrive/Colab Notebooks/ComputerVision/Digit_Classification/Persian/CarNumberPlate/train/labels"
val_images_path = "/content/drive/MyDrive/Colab Notebooks/ComputerVision/Digit_Classification/Persian/CarNumberPlate/val/images"
val_labels_path = "/content/drive/MyDrive/Colab Notebooks/ComputerVision/Digit_Classification/Persian/CarNumberPlate/val/labels"
test_images_path = "/content/drive/MyDrive/Colab Notebooks/ComputerVision/Digit_Classification/Persian/CarNumberPlate/test/images"
test_labels_path = "/content/drive/MyDrive/Colab Notebooks/ComputerVision/Digit_Classification/Persian/CarNumberPlate/test/labels"

# Creating the dirs if there are not already
if not os.path.exists(train_images_path):
  os.makedirs(train_images_path)
if not os.path.exists(train_labels_path):
  os.makedirs(train_labels_path)
if not os.path.exists(val_images_path):
  os.makedirs(val_images_path)
if not os.path.exists(val_labels_path):
  os.makedirs(val_labels_path)
if not os.path.exists(test_images_path):
  os.makedirs(test_images_path)
if not os.path.exists(test_labels_path):
  os.makedirs(test_labels_path)

# Converting the label format
import xml.etree.ElementTree as ET

def change_format(xml_path):
  tree = ET.parse(xml_path)
  root = tree.getroot()
  width = int(root.find('.//width').text)
  height = int(root.find('.//height').text)
  xmin = int(root.find('.//bndbox/xmin').text)
  ymin = int(root.find('.//bndbox/ymin').text)
  xmax = int(root.find('.//bndbox/xmax').text)
  ymax = int(root.find('.//bndbox/ymax').text)

  # Transform the bbox co-ordinates as per the format required by YOLO v7
  center_x = (xmin + xmax) / 2 
  center_y = (ymin + ymax) / 2
  b_width    = (xmax - xmin)
  b_height   = (ymax - ymin)
          
  # Normalise the co-ordinates by the dimensions of the image
  center_x /= width 
  center_y /= height 
  b_width    /= width 
  b_height   /= height

  class_label = 0

  return class_label, center_x, center_y, b_width, b_height


# print("Width: {}".format(width))
# print("Height: {}".format(height))
# print("XMin: {}, YMin: {}\nXMax: {}, YMax: {}".format(xmin, ymin, xmax, ymax))

# Reading data file and spliting it into train, validation and test sets both image and labels

for file_name in os.listdir(cars_path):
  if file_name.endswith(".jpg"):
      txt_name = file_name[:-4] + ".txt"
      xml_name = file_name[:-4] + ".xml"
      xml_path = os.path.join(cars_path, xml_name)

      img_path = os.path.join(cars_path, file_name)
      
      # train_set
      if random.random() <= 0.7:
          shutil.copy(os.path.join(cars_path, file_name) , os.path.join(train_images_path, file_name))
          txt_path = os.path.join(train_labels_path, txt_name)
          numbers = change_format(xml_path)
          with open(txt_path, "w") as file:
            for number in numbers:
              file.write(str(number) + " ")
      
      # test_set
      elif random.random() >= 0.9:
          shutil.copy(os.path.join(cars_path, file_name) , os.path.join(test_images_path, file_name))
          txt_path = os.path.join(test_labels_path, txt_name)
          numbers = change_format(xml_path)
          with open(txt_path, "w") as file:
            for number in numbers:
              file.write(str(number) + " ")
      
      # validation_set
      else:
          shutil.copy(os.path.join(cars_path, file_name) , os.path.join(val_images_path, file_name))
          txt_path = os.path.join(val_labels_path, txt_name)
          numbers = change_format(xml_path)
          with open(txt_path, "w") as file:
            for number in numbers:
              file.write(str(number) + " ")

"""## Fine tune the **Yolo7** over the dataset"""

!git clone https://github.com/augmentedstartups/yolov7.git

# Commented out IPython magic to ensure Python compatibility.
# %cd yolov7

!pip install -r requirements.txt

# Commented out IPython magic to ensure Python compatibility.
# ## Download yolov7 weights
# %%bash
# wget -P "/content/drive/MyDrive/Colab Notebooks/ComputerVision/Digit_Classification/Persian/CarNumberPlate" https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt

"""**NOTE:** If you want to run in **GPU**, before running: change the command line 742 in **loss.py** to the following command: 
**matching_matrix = torch.zeros_like(cost, device="cpu")**.
The difference is **device="cpu"**.
"""

!python train.py --weights "/content/drive/MyDrive/Colab Notebooks/ComputerVision/Digit_Classification/Persian/CarNumberPlate/yolov7.pt" --data "/content/drive/MyDrive/Colab Notebooks/ComputerVision/Digit_Classification/Persian/CarNumberPlate/yaml_yolov7.yml" --epochs 70 --batch 10 --device 0

!python detect.py --weight "/content/yolov7/runs/train/exp2/weights/best.pt" --source "/content/drive/MyDrive/Colab Notebooks/ComputerVision/Digit_Classification/Persian/CarNumberPlate/test/images"
