# 1. Installation Steps

## 1.1. Python 3

The project is developed in Python. Ensure you have **Python 3** installed (e.g., 3.11 or 3.12). On macOS:
`python3 --version`

## 1.2. Install OpenCV & Contribution Modules

We need the following contribution modules for CSRT tracking:
`pip3 install opencv-python opencv-contrib-python`

Where:

- **opencv-python**:  
  - This package contains the **core** functionalities of OpenCV.  
  - It includes many of the built-in modules (image I/O, transformations, basic image processing routines, etc.).

- **opencv-contrib-python**:  
  - This package includes all the extra (**contrib**) modules that are not part of the core OpenCV distribution.  
  - These “contrib” modules may be experimental, specialized, or just maintained separately from the core library.  
  - Object tracking algorithms such as **CSRT**, **KCF**, and **MOSSE** are within these contrib modules.

This gives us functions like `cv2.TrackerCSRT_create()`. 

## 1.3. Install PyTorch

Next, we need **PyTorch** because that’s the deep learning framework used by YOLOv5 to:
- Load pretrained weights (e.g., `yolov5m.pt`) and perform model inference on new images.
- Train or fine-tune models if you’re customizing YOLOv5 with your own dataset (as we did with the “character” class).


Adjust the exact command for your system, e.g., if you have CPU-only:
`pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu`

Or if you have a GPU on Linux:
`pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118`

## 1.4. Clone YOLOv5 / Install YOLOv5 Requirements

At this point, we can clone the **Ultralytics YOLOv5** repository and install it.
`git clone https://github.com/ultralytics/yolov5.git cd yolov5 pip3 install -r requirements.txt`

## 1.5. Install LabelImg

Next, we install **LabelImg** ([HumanSignal/labelImg](https://github.com/HumanSignal/labelImg)) to draw bounding boxes around the character in each screenshot and generate annotation files:
`pip3 install pyqt5 lxml git clone https://github.com/HumanSignal/labelImg.git
cd labelImg
make qt5py3
python3 labelImg.py`


## 1.6. Comet ML
We also want to log experiment metrics to **Comet ML** ([comet.com](https://www.comet.com/)). The first step is to head to Comet ML’s site, create a free account, and generate an API key as well as a new project:
`pip3 install comet-ml`

---

# 2. Create the Dataset

Now, we’re going to create the dataset we’ll later use to train our YOLOv5 model to detect the character.
`cd labelImg python3 labelImg.py`


Once opened, we need to open the directory containing the character screenshots we intend to use for training and validation. Ensure **YOLO** file format is selected. Then, create rectangular boxes around the character and assign the label (e.g., `"character"`). This will generate `.txt` annotations for each image.

---

# 3. Organise the Dataset

## 3.1. Train/Validation Split

- **80%** of the images go into `/character_screenshots/images/train`, with matching text files in `/character_screenshots/labels/train`.
- **20%** of the images go into `/character_screenshots/images/val`, with matching text files in `/character_screenshots/labels/val`.

## 3.2. Check Filenames

Next, we need to edit `data.yaml`. In YOLOv5, the `data.yaml` file is a configuration file that defines the structure and location of our dataset. It provides YOLO with all the information it needs to correctly load our training and validation data, as well as how to interpret the labels in our dataset.

By default, LabelImg might ship with multiple classes. Ensure `"character"` is the correct label ID (typically `0` if it’s the only class).

---

# 4. Train YOLOv5 (recommended Medium Model as a minimum)

At this stage, we’re going to train the YOLOv5 model to recognize the character ([Ultralytics YOLOv5 docs](https://docs.ultralytics.com)).

1. **Go to YOLOv5 Directory**  
`cd yolov5`


2. **Train Command** (using medium model `yolov5m.pt`):  
`python3 train.py --img 640 --batch 8 --epochs 50
--data data.yaml
--weights yolov5m.pt
--name character-detector`

- `--img 640`: each image is resized to 640×640.  
- `--batch 8`: adjust based on your hardware.  
- `--epochs 50`: you can increase if you want more training time.  
- `--data data.yaml`: the custom dataset config from above.  
- `--weights yolov5m.pt`: the medium YOLOv5 variant.

**Check Results**:  
YOLO logs training/val accuracy over epochs. The final weights go to `runs/train/character-detector/weights/best.pt`.

---

# 5. Offline Detection & Tracking (CSRT) with Comet ML

We develop a Python3 script (**detect_and_track.py**) that:
- Loads YOLO’s `best.pt` to detect the character.  
- Periodically uses YOLO detection or re-detects when the tracker fails.  
- Between detections, uses OpenCV’s **CSRT** tracker to follow the bounding box.  
- Logs metrics (IoU, bounding-box area, reacquisition counts, etc.) to both CSV and Comet ML.

## 5.1. Key Points in the Script
- **Periodic Detection**: by default every 30 frames.  
- **If Tracker Fails**: run detection immediately to reacquire.  
- **CSRT Tracking**: `tracker = cv2.TrackerCSRT_create()`
- **Comet ML**:  
  - Set your API key and project/workspace info at the top.  
  - `experiment.log_metric("metric_name", value, step=frame_index)` logs a metric.
