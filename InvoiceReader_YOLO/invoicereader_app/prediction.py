#Importing cv2 for image preprocessing
import cv2 

def getPredictions(image):
    #Reading image through OpenCV

    img = image.copy()

    #load the COCO class labels on which pre-trained YOLO model was trained on
    with open(r"C:\Users\inass.elmadhoun\Documents\semester 7\Proof of concept\NLP cards\obj.names", 'r') as f: 
        classes = f.read().splitlines()

    #Reading YOLOv4 network configuration and weights 
    net = cv2.dnn.readNetFromDarknet(r"C:\Users\inass.elmadhoun\Documents\semester 7\Proof of concept\NLP cards\yolov4-obj.cfg", r"C:\Users\inass.elmadhoun\Documents\semester 7\Proof of concept\NLP cards\yolov4-obj_best.weights") 

    #Initializing detection model
    model = cv2.dnn_DetectionModel(net) 

    #Setting different parameters to preprocess the image
    model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True) 

    #Gives predicted ClassIds,Scores and bounding boxes co-ordinates
    classIds, scores, boxes = model.detect(img, confThreshold=0.6, nmsThreshold=0.4) 

    #Below code will draw the rectangle using co-ordinates of bounding boxes

    for (classId, score, box) in zip(classIds, scores, boxes):
        cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]),
                    color=(0, 255, 0), thickness=2) 
        
    #Below code will store class id and score of the same
        crop_image = img[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]  
        text = '%s: %.2f' % (classes[classId], score) 
        print(text)
        # plt.imshow(cropped_image)
        cv2.imwrite('contour1.png', crop_image)    
    #Below code will write text above bounding box which will display class names and scores
    cv2.putText(img, text, (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    color=(0, 255, 0), thickness=1) 
    
    
    cv2.imshow('Image', crop_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return img