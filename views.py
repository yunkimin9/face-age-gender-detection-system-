from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import cv2
import numpy as np
import base64
import json
import os
from datetime import datetime

# Initialize networks once when the module loads
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
faceModel = os.path.join(base_dir, 'opencv_face_detector_uint8.pb')
faceProto = os.path.join(base_dir, 'opencv_face_detector.pbtxt')
ageModel = os.path.join(base_dir, 'age_net.caffemodel')
ageProto = os.path.join(base_dir, 'age_deploy.prototxt')
genderModel = os.path.join(base_dir, 'gender_net.caffemodel')
genderProto = os.path.join(base_dir, 'gender_deploy.prototxt')

# Initialize networks
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# Constants
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# Age range mapping for more precise age estimation
age_ranges = {
    '(0-2)': (0, 2),
    '(4-6)': (4, 6),
    '(8-12)': (8, 12),
    '(15-20)': (15, 20),
    '(25-32)': (25, 32),
    '(38-43)': (38, 43),
    '(48-53)': (48, 53),
    '(60-100)': (60, 100)
}

# Store detection history
detection_history = []

def get_precise_age(age_preds):
    # Get the predicted age range
    age_range = ageList[age_preds[0].argmax()]
    min_age, max_age = age_ranges[age_range]
    
    # Calculate confidence for each age in the range
    confidences = age_preds[0]
    total_confidence = sum(confidences)
    
    # Calculate weighted average age
    weighted_age = 0
    for i, conf in enumerate(confidences):
        min_age, max_age = age_ranges[ageList[i]]
        avg_age = (min_age + max_age) / 2
        weighted_age += avg_age * (conf / total_confidence)
    
    return round(weighted_age)

@csrf_exempt
def detect_faces(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            image_data = data.get('image')
            
            # Remove the data URL prefix
            image_data = image_data.split(',')[1]
            
            # Decode base64 image
            nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Process frame
            frameHeight = frame.shape[0]
            frameWidth = frame.shape[1]
            
            # Resize frame for faster processing while maintaining aspect ratio
            max_dim = 640
            scale = min(max_dim / frameWidth, max_dim / frameHeight)
            if scale < 1:
                frame = cv2.resize(frame, None, fx=scale, fy=scale)
                frameHeight, frameWidth = frame.shape[:2]
            
            # Create blob and detect faces
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], swapRB=False)
            faceNet.setInput(blob)
            detection = faceNet.forward()
            
            faces = []
            current_time = datetime.now()
            
            for i in range(detection.shape[2]):
                confidence = detection[0, 0, i, 2]
                if confidence > 0.7:
                    x1 = int(detection[0, 0, i, 3] * frameWidth)
                    y1 = int(detection[0, 0, i, 4] * frameHeight)
                    x2 = int(detection[0, 0, i, 5] * frameWidth)
                    y2 = int(detection[0, 0, i, 6] * frameHeight)
                    
                    # Extract face ROI
                    face = frame[max(0, y1):min(y2, frameHeight-1), max(0, x1):min(x2, frameWidth-1)]
                    
                    if face.size == 0:
                        continue
                    
                    # Prepare blob for age and gender detection
                    face_blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                    
                    # Gender detection
                    genderNet.setInput(face_blob)
                    genderPreds = genderNet.forward()
                    gender = genderList[genderPreds[0].argmax()]
                    gender_confidence = float(genderPreds[0].max())
                    
                    # Age detection
                    ageNet.setInput(face_blob)
                    agePreds = ageNet.forward()
                    precise_age = get_precise_age(agePreds)
                    age_confidence = float(agePreds[0].max())
                    
                    # Scale coordinates back to original size if frame was resized
                    if scale < 1:
                        x1 = int(x1 / scale)
                        y1 = int(y1 / scale)
                        x2 = int(x2 / scale)
                        y2 = int(y2 / scale)
                    
                    face_data = {
                        'x1': x1,
                        'y1': y1,
                        'x2': x2,
                        'y2': y2,
                        'confidence': float(confidence),
                        'gender': gender,
                        'gender_confidence': gender_confidence,
                        'age': precise_age,
                        'age_confidence': age_confidence,
                        'timestamp': current_time.isoformat()
                    }
                    
                    faces.append(face_data)
                    detection_history.append(face_data)
            
            # Keep only the last 100 detections
            if len(detection_history) > 100:
                detection_history.pop(0)
            
            return JsonResponse({
                'success': True,
                'faces': faces
            })
            
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            })
    
    return JsonResponse({
        'success': False,
        'error': 'Invalid request method'
    })

def index(request):
    return render(request, 'detector/index.html') 