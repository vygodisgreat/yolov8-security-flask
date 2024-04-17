yolov8-security

`app.py` initializes a `Flask` application that runs on the local server and automatically connects to all the webcameras connected to the system.
This can be thought of as a scalable model for a server that handles multiple CCTV footages and needs automatic monitoring to detect any weapon if it is in frame.

The webapp then displays all active webcams on the system that are available.

Click on the desired camera icon to open a tab that displays live footage from the current cam and also when gun is detected in frame **(continuously for > 10 frames as well as with confidence > 0.7 by the model)** then the Frame is captured and stored in the local database and alert is sent to the User.

The model is `yolov8s.pt` that is custom traiend on a dataset of 3K images of pistols and people holding pistols in `colab`
The dataset is prepared from `Roboflow`. 

Dependencies are `opencv`, `Flask` and `ultralytics` for the model.