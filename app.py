from flask import Flask, render_template, Response
from ultralytics import YOLO
import supervision as sv
import cv2
import datetime
import os

model_path = 'yolov8sbest.pt'
# put the model path here!

num_cams = 0
# we need to calculate the numebr of webcams that are connected to the server!
for i in range(10):  
    # iterate through potential webcams
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if cap.isOpened():
        num_cams += 1
        cap.release()
    else:
        break
print("WEBCAMS FOUND:", num_cams)   

# creating captures directory
root_dir = os.getcwd()
print("pwd:", root_dir)
root_dir = f"{root_dir}\\CAPTURES"

try:
    os.mkdir(root_dir)
except:
    # directory already exists!
    pass

# create the image directories equal to number of webcams
for i in range(num_cams):
    try:
        os.mkdir(f"{root_dir}\\CAMERA{i}")
    # create the folder for ith camera images
    except:
        pass
    
app = Flask(__name__)
print("FLASK APP INITIALIZED")
        
def get_frames(cam_id):
    # continuously send processed frames to the application
    
    cam = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
    # start the camera of the given ID
    
    model = YOLO(model_path)
    # create a local model instance
    
    ctr = 0
    # counter for frames, only show when gun continuously for > 10 frames?
    rst = False
    # reset flag that helps reset the counter to zero if no gun detected!
    
    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    # labelling accessories
    
    images_directory = root_dir + "\\CAMERA" + str(cam_id) + "\\"
    # directory to store images
    
    print(f"STARTING CAMERA {cam_id}")
    
    while (True):    
        
        flag, frame = cam.read()
        if (not flag):
            print(f"CAMERA {cam_id} NOT WORKING!")
            break
        
        result = model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)
        
        # if gun is not detected just send in the raw frame
        if not detections.confidence.size:
            _, frame = cv2.imencode('.jpg', frame)
            frame = frame.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            continue
                
        rst = False
        
        # check if gun detection is valid and if it is, increment frame counter for threshold
        for val in detections.confidence:
            if (val>=0.6):
                ctr += 1
                rst = True
                break
        
        print(ctr)
        
        # if confidence of gun detection is not much then also return the raw frame   
        if (not rst):
            ctr = 0
        if (ctr<5):
            _, frame = cv2.imencode('.jpg', frame)
            frame = frame.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            continue
        
        # gun detected as well as valid
        labels = [f"{model.model.names[class_id]} {confidence:.2f}" for class_id, confidence in zip(detections.class_id, detections.confidence)]
        annotated_image = bounding_box_annotator.annotate(scene=frame, detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
        
        try:    
            flag, buffer = cv2.imencode('.jpg', annotated_image)
            buffer = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer + b'\r\n')
            # yield continuously returns the value like bytes in this case
            if (ctr%10==0):
                ts = datetime.datetime.now()
                ts = ts.strftime("%Y-%m-%d_%H-%M-%S")
                filepath = images_directory + ts + ".png"
                # creating the string timestamp for saving images
                print(filepath)
                status = cv2.imwrite(filepath, annotated_image)
                if status:
                    print("pic saved!")
                else:
                    print('unable to save!')
                    
        except Exception as e:
            print(e)
        
@app.route('/')
def INDEX():
    return render_template('index.html', cams=num_cams)
    # return "hello"

@app.route('/video/<int:cam_id>')
def VIDEO(cam_id):
    return Response(get_frames(cam_id), mimetype='multipart/x-mixed-replace; boundary=frame')

# let the show begin!
if (__name__=='__main__'):
    app.run()