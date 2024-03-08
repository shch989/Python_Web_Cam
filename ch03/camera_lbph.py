import cv2

# 얼굴 감지기 모델을 불러옵니다.
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# 얼굴 인식기를 생성하고 미리 학습된 모델을 불러옵니다.
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("lbph_classifier.yml")

# 얼굴 이미지의 크기를 설정합니다.
width, height = 220, 220

# 텍스트 표시에 사용할 폰트를 설정합니다.
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

# 비디오 캡처 객체를 생성합니다. 웹캠을 사용하려면 인덱스 0을 전달합니다.
camera = cv2.VideoCapture(0)

while (True):
    # 비디오에서 프레임을 캡처합니다.
    connected, image = camera.read()

    # 프레임을 흑백으로 변환합니다.
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 얼굴을 감지합니다.
    detections = face_detector.detectMultiScale(image_gray, scaleFactor=1.5, minSize=(30,30))

    # 감지된 얼굴에 대해 반복합니다.
    for (x, y, w, h) in detections:
        # 얼굴 영역을 추출하고 크기를 조정합니다.
        image_face = cv2.resize(image_gray[y:y + w, x:x + h], (width, height))

        # 감지된 얼굴 주위에 사각형을 그립니다.
        cv2.rectangle(image, (x, y), (x + w, y + h), (0,0,255), 2)

        # 얼굴을 인식하고 해당하는 이름과 신뢰도를 얼굴 주위에 표시합니다.
        id, confidence = face_recognizer.predict(image_face)
        name = ""
        if id == 1:
            name = 'Jones'
        elif id == 2:
            name = 'Gabriel'
        cv2.putText(image, name, (x,y +(w+30)), font, 2, (0,0,255))
        cv2.putText(image, str(confidence), (x,y + (h+50)), font, 1, (0,0,255))

    # 결과 프레임을 화면에 표시합니다.
    cv2.imshow("Face", image)

    # 'q' 키를 누르면 종료합니다.
    if cv2.waitKey(1) == ord('q'):
        break

# 캡처 객체를 해제하고 윈도우를 닫습니다.
camera.release()
cv2.destroyAllWindows()