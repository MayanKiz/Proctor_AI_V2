
# Proctoring-AI

Project to create an automated proctoring system where the user can be monitored automatically through the webcam and microphone. The project is divided into two parts: vision and audio based functionalities. An explanation of some functionalities of the project can be found on my [medium article](https://towardsdatascience.com/automating-online-proctoring-using-ai-e429086743c8?source=friends_link&sk=fbc385d1a8c55628a916dc714747f276).


## Run locally

This combined version includes the V1 app from [proctoring_ai_102](https://github.com/MayanKiz/proctoring_ai_102) and runs **V1 first, then the V2 visual checks**. Use **64-bit Python 3.11**. The requirements use MediaPipe plus the TensorFlow/OpenCV dependencies and pin scikit-learn 1.2.2 for the bundled older face-spoof model.

### Windows (PowerShell)

```powershell
git clone https://github.com/MayanKiz/Proctor_AI_V2.git
cd Proctor_AI_V2
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python run_camera.py
```

If the repo is already cloned, run `git pull origin main`, then start at `cd Proctor_AI_V2` (or open PowerShell in that folder) and continue from creating the virtual environment. If you already have a `.venv` from the earlier V2 setup, remove and recreate it before installing: the combined version replaces OpenCV with MediaPipe's compatible contrib package. In PowerShell, use `Remove-Item -Recurse -Force .venv` before `py -3.11 -m venv .venv`. If `py -3.11` is not recognized, install 64-bit Python 3.11 first. If `Activate.ps1` is not found, run `py -3.11 -m venv .venv` first, then activate it.

### macOS / Linux

```bash
git clone https://github.com/MayanKiz/Proctor_AI_V2.git
cd Proctor_AI_V2
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python run_camera.py
```

First, V1 starts in the MAYANK style and monitors the webcam. Look straight at the screen during its short calibration. Press `q` or `Esc` in the V1 window to finish V1; **V2 then starts automatically**. V1 asks whether to enable Telegram in an interactive terminal; choose Skip to keep the session local. Use `python run_camera.py --no-telegram` to skip the Telegram prompt and force local-only mode.

V2 then runs eye tracking, head pose, mouth opening, person/phone detection, and face-spoof checks in order. Press `q` in a V2 window to move to the next check. During mouth setup, face the camera and press `r` to calibrate; after calibration press `q` to continue. The first person/phone check downloads YOLOv3 weights (about 236 MB), so allow it time and ensure there is enough disk space.

If you add your exam question text to `paper.txt`, the launcher offers an optional V2 microphone check at the end. It records up to 30 seconds and sends the audio to Google Speech Recognition to transcribe it. Type exactly `YES` only if you agree to that transmission; pressing Enter skips it without recording. The transcript is saved locally as `test.txt`.

To run the FastAPI API instead, use `python -m uvicorn main:app --reload` and open `http://127.0.0.1:8000/docs`. Stop the combined run or API with `Ctrl+C` in the terminal.

**Hosting note:** `python run_camera.py` uses the computer's local webcam. Deploying this command to a cloud host would access the cloud machine's camera, not the visitor's. A live website that uses each visitor's webcam will need a browser camera interface and a secure frame-streaming/API design; that is a separate hosting step.

### Prerequisites
To run the programs in this repo, do the following:
- create a virtual environment using the command:
  - `python -m venv venv`
- activate the virtual environment
  - `./venv/Scripts/activate` (windows users)
  - `source ./venv/bin/activate` (mac and linux users)
- install the requirements
  - `pip install --upgrade pip` (to upgrade pip)
  - `pip install -r requirements.txt`

Once the requirements have been installed, The programs will run successfully.
Except for the `person_and_phone.py` script which requires a model to be downloaded.

More on that later.


For vision:
```
Tensorflow>2
OpenCV
sklearn=0.19.1 # for face spoofing. 
The model used was trained with this version and does not support recent ones.
```
For audio:
```
pyaudio
speech_recognition
nltk
```

## Vision

It has six vision based functionalities right now:
1. Track eyeballs and report if candidate is looking left, right or up.
2. Find if the candidate opens his mouth by recording the distance between lips at starting.
3. Instance segmentation to count number of people and report if no one or more than one person detected.
4. Find and report any instances of mobile phones.
5. Head pose estimation to find where the person is looking.
6. Face spoofing detection

### Face detection
Earlier, Dlib's frontal face HOG detector was used to find faces. However, it did not give very good results. In [face_detection](../../tree/master/face_detection) different face detection models are compared and OpenCV's DNN module provides best result and the results are present in [this article](https://towardsdatascience.com/face-detection-models-which-to-use-and-why-d263e82c302c?source=friends_link&sk=c9e2807cf216115d7bb5a9b827bb26f8).

It is implemented in `face_detector.py` and is used for tracking eyes, mouth opening detection, head pose estimation, and face spoofing.

An additional quantized model is also added for face detector as described in [Issue 14](https://github.com/vardanagarwal/Proctoring-AI/issues/14). This can be used by setting the parameter `quantized` as True when calling the `get_face_detector()`. On quick testing of face detector on my laptop the normal version gave ~17.5 FPS while the quantized version gave ~19.5 FPS. This would be especially useful when deploying on edge devices due to it being uint8 quantized.

### Facial Landmarks
Earlier, Dlib's facial landmarks model was used but it did not give good results when face was at an angle. Now, a model provided in this [repository](https://github.com/yinguobing/cnn-facial-landmark) is used. A comparison between them and the reason for choosing the new Tensorflow based model is shown in this [article](https://towardsdatascience.com/robust-facial-landmarks-for-occluded-angled-faces-925e465cbf2e?source=friends_link&sk=505eb1101576227f4c38474092dd4c22).

It is implemented in `face_landmarks.py` and is used for tracking eyes, mouth opening detection, and head pose estimation.

#### Note
If you want to use dlib models then checkout the [old-master branch](https://github.com/vardanagarwal/Proctoring-AI/tree/old_master).

### Eye tracking
`eye_tracker.py` is to track eyes. A detailed explanation is provided in this [article](https://towardsdatascience.com/real-time-eye-tracking-using-opencv-and-dlib-b504ca724ac6?source=friends_link&sk=d9db46e2f41258c6c23d18792775d2a5). However, it was written using dlib.

### Mouth Opening Detection
`mouth_opening_detector.py` is used to check if the candidate opens his/her mouth during the exam after recording it initially. It's explanation can be found in the main article, however, it is using dlib which can be easily changed to the new models.

### Person counting and mobile phone detection
`person_and_phone.py` is for counting persons and detecting mobile phones. YOLOv3 is used in Tensorflow 2 and it is explained in this [article](https://medium.com/analytics-vidhya/count-people-in-webcam-using-yolov3-tensorflow-f407679967d5?source=friends_link&sk=95ae7a010eeef429a407a7a2de2ff8ec) for more details.

### Head pose estimation
`head_pose_estimation.py` is used for finding where the head is facing. An explanation is provided in this [article](https://towardsdatascience.com/real-time-head-pose-estimation-in-python-e52db1bc606a?source=friends_link&sk=0bae01db2759930197bfd33777c9eaf4)

### Face spoofing
`face_spoofing.py` is used for finding whether the face is real or a photograph or image. An explanation is provided in this [article](https://medium.com/visionwizard/face-spoofing-detection-in-python-e46761fe5947). The model and working is taken from this Github [repo](https://github.com/ee09115/spoofing_detection).

### FPS obtained

Functionality | On Intel i5
--- | ---
Eye Tracking | 7.1
Mouth Detection | 7.2
Person and Phone Detection | 1.3
Head Pose Estimation | 8.5
Face Spoofing | 6.9

If you testing on a different processor a GPU consider making a pull request to add the FPS obtained on that processor.


## Audio
It is divided into two parts:
1. Audio from the microphone is recording and converted to text using Google's speech recognition API. A different thread is used to call the API such that the recording portion is not disturbed a lot, which processes the last one, appends its data to a text file and deletes it.
2. NLTK we remove the stopwods from that file. The question paper (in txt format) is taken whose stopwords are also removed and their contents are compared. Finally, the common words along with its number are presented to the proctor.

The code for this part is available in `audio_part.py`

## To do
1. ~~Replace the HOG based descriptor by OpenCV's DNN modules Caffe model and it will also solve the issues created by side faces and occlusion.~~
2. ~~Replace the dlib based facial landmarks with the CNN based facial landmarks as used in head_pose_detector.~~
3. Make a better face spoofing model as the accuracy is not good currently.
4. Use a smaller and faster model inplace of YOLOv3 that can give good FPS on a CPU.
5. Add a vision based functionality: face recognition such that no one else replaces the candidate and gives the exam midway.
6. Add a vision based functionality: id-card verification.
7. ~~Update README with videos of each functionality and the FPS obtained.~~
8. ~~Add documentation (docstring) in functions in codes.~~

### Problems
Speech to text conversion which might not work well for all dialects.

## Contributing

If you have any other ideas or do any step of to do consider making a pull request . Please update the README as well in the pull request.

## License

The V2 source is licensed under the MIT License in `LICENSE.md`; the imported V1 source retains its separate MIT license in `LICENSE-V1`. The facial-landmark model has its own dataset/model usage terms; check those terms before commercial use.

## Like what I am doing
<a href="https://www.buymeacoffee.com/rao.siyaaa
  " target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="41" width="174"></a>
