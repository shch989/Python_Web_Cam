import cv2

# OpenCV의 얼굴 감지기를 불러옵니다.
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# 비디오 캡처 객체를 생성합니다. 웹캠을 사용하려면 인덱스 0을 전달합니다.
video_capture = cv2.VideoCapture(0)

while True:
    # 프레임 단위로 캡처합니다.
    ret, frame = video_capture.read()

    # 이미지를 흑백으로 변환합니다.
    image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 얼굴을 감지합니다.
    detections = face_detector.detectMultiScale(image_gray, minSize=(100, 100),
                                                minNeighbors=5)

    # 감지된 얼굴 주위에 사각형을 그립니다.
    for (x, y, w, h) in detections:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # 결과 프레임을 화면에 표시합니다.
    cv2.imshow('Video', frame)

    # 'q' 키를 누르면 종료합니다.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 캡처 객체를 해제하고 윈도우를 닫습니다.
video_capture.release()
cv2.destroyAllWindows()