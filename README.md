# 1. Installation Steps

## 1.1. Python 3

The project is developed in Python. Ensure you have **Python 3** installed (e.g., 3.11 or 3.12). On macOS:
python3 --version

## 1.2. Install OpenCV & Contribution Modules

**OpenCV** is an open-source library that provides real-time computer vision functionalities.  
It’s written in C++ for efficiency and provides Python bindings so you can easily call those optimized routines from Python code. It’s a cornerstone for many computer vision projects, handling everything from basic image manipulation (blur, threshold, etc.) to more advanced tasks such as feature detection, camera calibration, object tracking, and more.

Typical reasons to use OpenCV in a project include:
- Reading and writing images or videos in formats like `.jpg`, `.png`, or `.mp4`.
- Pre-processing frames (e.g., resizing, cropping, color space conversion) before passing them to a deep learning model.
- Classical computer vision algorithms for tasks like keypoint matching (ORB, SIFT, etc.), face detection (Haar cascades), or object tracking (KCF, CSRT, etc.).
- Drawing bounding boxes, text, or other annotations over images or videos.

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
The **CSRT** (Channel and Spatial Reliability Tracking) tracker is a single-object tracking algorithm included in OpenCV’s contribution module. CSRT allows you to track a single region of interest (ROI) across frames in a video after we specify its initial bounding box. Once initialised, the tracker attempts to follow that object from frame to frame, handling moderate changes in scale and movement. It is known to be more accurate than some other built-in trackers (e.g., **KCF**, **MOSSE**), although it can be a bit slower. If an application demands higher accuracy rather than minimal computational cost, CSRT is typically a good choice.  

`cv2.TrackerCSRT_create()` is how we instantiate the tracker object in Python. Once created, we can initialise it by specifying the bounding box (x, y, width, height) on the first frame. Then we update it on each subsequent frame to get the new bounding box. The CSRT tracker follows exactly **one** object. If an application needs multi-object tracking, we either manage multiple trackers ourselves or use more sophisticated approaches (e.g., **SORT**, **DeepSORT**).

## 1.3. Install PyTorch

Next, we need **PyTorch** because that’s the deep learning framework used by YOLOv5 to:

- Load pretrained weights (e.g., `yolov5m.pt`) and perform model inference on new images.
- Train or fine-tune models if you’re customizing YOLOv5 with your own dataset (as we did with the “character” class).

Here’s what happens behind the scenes:
- **YOLOv5** is a collection of Python scripts and model definitions that rely on PyTorch for things like **tensor operations**, **GPU acceleration** (if we have a compatible GPU), **model loading/saving**, and the **training loop** (loss computation, backpropagation, etc.).
- When we run a command like `python3 train.py --weights yolov5m.pt`, PyTorch handles all neural network operations—loading the YOLO model architecture, setting up layers, optimizing parameters, and so on.
- When we do **inference** (detecting objects in a frame), YOLOv5 uses PyTorch to load the trained weights, process each frame as a tensor, and output bounding boxes, class labels, and confidences.

Essentially, **PyTorch** is the foundation that executes YOLOv5’s deep learning tasks. Without PyTorch, YOLOv5 wouldn’t be able to load or run its neural network, and thus no detection would occur.

Adjust the exact command for your system, e.g., if you have CPU-only:

`pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu`

Or if you have a GPU on Linux:

`pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118`


### 1.3.1. A Note on Tensors

In the context of PyTorch and YOLOv5, **tensors** are the fundamental data structures that store and process all the numerical values needed for deep learning tasks. You can think of tensors as multidimensional arrays similar to NumPy arrays, but with additional features like:

- **N-dimensional**: Tensors can represent data with any number of dimensions. For instance, a 2D tensor (matrix) for grayscale images, a 4D tensor for a batch of colored images `(batch_size, channels, height, width)`, etc.
- **GPU acceleration**: Tensors can reside on the CPU or GPU, and PyTorch automatically uses the correct device for accelerated linear algebra operations if a compatible GPU is available.
- **Automatic Differentiation**: Deep learning frameworks like PyTorch track operations performed on tensors, so they can compute gradients automatically. This is essential during training, where you update model weights via backpropagation.

How Tensors Fit into YOLOv5:

- **During Training**  
  Your images (and labels) are loaded from disk and turned into tensors.  
  These are fed into the YOLOv5 model, which also represents its weights as tensors.  
  PyTorch then performs forward passes (computing predictions) and backward passes (computing gradients), all as tensor operations, to update the model’s parameters.

- **During Inference**  
  When you pass a single image or a batch of images to YOLOv5, those images are converted to tensors (e.g., shape `(batch_size, 3, height, width)`).  
  The model processes the tensor, producing another tensor that contains the bounding box coordinates, class probabilities, etc.  
  Finally, PyTorch converts these predictions (still in tensor form) to more human-readable outputs (e.g., bounding box coordinates, confidence scores).

In short, **tensors** are the “containers” for your data and your model’s parameters. They allow PyTorch to efficiently perform large numbers of matrix and vector operations on the CPU or GPU, enabling YOLOv5 (and many other deep learning models) to learn and make predictions effectively.

## 1.4. Clone YOLOv5 / Install YOLOv5 Requirements

At this point, we can clone the **Ultralytics YOLOv5** repository and install it.
`git clone https://github.com/ultralytics/yolov5.git cd yolov5 pip3 install -r requirements.txt`

Alternatively, you can just do:
`pip3 install ultralytics`

to get the YOLO functionality, but the conversation’s steps use the YOLOv5 repo.

## 1.5. Install LabelImg

Next, we install **LabelImg** ([HumanSignal/labelImg](https://github.com/HumanSignal/labelImg)) to draw bounding boxes around the character in each screenshot and generate annotation files:
`pip3 install pyqt5 lxml git clone https://github.com/HumanSignal/labelImg.git cd labelImg make qt5py3 python3 labelImg.py`


## 1.6. Comet ML

We also want to log experiment metrics to **Comet ML** ([comet.com](https://www.comet.com/)). The first step is to head to Comet ML’s site, create a free account, and generate an API key as well as a new project:

*(Note: Steps on setting the COMET_API_KEY and installing the Comet library)*

`pip3 install comet-ml`


---

# 2. Create the Dataset

Now, we’re going to create the dataset we’ll later use to train our YOLOv5 model to detect the character.

`cd labelImg python3 labelImg.py`


Once opened, we need to open the directory containing the character screenshots we intend to use for training and validation. Ensure **YOLO** file format is selected. Then, create rectangular boxes around the character and assign the label (e.g., `"character"`). This will generate `.txt` annotations for each image.

---

# 3. Organise the Dataset

## 3.1. Train/Validation Split

- **80%** (40 images) go into `/Users/roger/Documents/SETracker/character_screenshots/images/train`, with matching text files in `/Users/roger/Documents/SETracker/character_screenshots/labels/train`.
- **20%** (10 images) go into `/Users/roger/Documents/SETracker/character_screenshots/images/val`, with matching text files in `/Users/Roger/Documents/SETracker/character_screenshots/labels/val`.

## 3.2. Check Filenames

Next, we need to edit `data.yaml`. In YOLOv5, the `data.yaml` file is a configuration file that defines the structure and location of our dataset. It provides YOLO with all the information it needs to correctly load our training and validation data, as well as how to interpret the labels in our dataset.

By default, LabelImg might ship with multiple classes. Ensure `"character"` is the correct label ID (typically `0` if it’s the only class).

---

# 4. Train YOLOv5 (Medium Model)

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
- **CSRT Tracking**: `tracker = cv2.TrackerCSRT_create()` (or `cv2.legacy.TrackerCSRT_create()`)  
- **Comet ML**:  
  - Set your API key and project/workspace info at the top.  
  - `experiment.log_metric("metric_name", value, step=frame_index)` logs a metric.
