from darkflow.darkflow.net.build import TFNet
import cv2
import tensorflow as tf
#options = {"model": "cfg/yolo-face.cfg", "load": "weight/yolo-face_final.weights", "threshold": 0.1, "gpu": 1.0}
options = {"model": "cfg/yolo-face.cfg", "load": "weight/yolo-face_4000.weights", "threshold": 0.1, "gpu": 1.0}
tfnet = TFNet(options)
count=0
tracker = cv2.TrackerMIL_create()
cam=cv2.VideoCapture(0)

while True:
    _, camcv = cam.read()
    camcv = cv2.resize(camcv, (448, 448))
    result= tfnet.return_predict(camcv)
    num_people=len(result)
    for i in range(num_people):
     #print(result[i]['topleft'])
        top_left_x=result[i]['topleft']['x']
        top_left_y=result[i]['topleft']['y']
        bottom_right_x=result[i]['bottomright']['x']
        bottom_right_y=result[i]['bottomright']['y']

        if count ==0 and top_left_x!=0:
            count +=1
            bbox=(top_left_x,top_left_y,bottom_right_x,bottom_right_y)
            ok = tracker.init(camcv, bbox)
        elif count==0:
            continue
        else:
            ok, bbox = tracker.update(camcv)
            person_num=str(i)

            cv2.putText(camcv, 'Person:'+person_num, (bbox[0],bbox[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            #cv2.putText(camcv, 'Person:' + 'ME', (top_left_x, top_left_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,

            cv2.rectangle(camcv,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(255,0,0))


    cv2.imshow('color', camcv)
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()


