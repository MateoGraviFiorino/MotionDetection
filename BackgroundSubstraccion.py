import cv2 

video = r"C:\Users\mateo\OneDrive\Escritorio\Facultad\5to Cuatrimestre\Computer Vision\Practica1\Vídeo 1.mp4"
cap = cv2.VideoCapture(video)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

ret, prev = cap.read()
if not ret:
    print("Error: Failed to read the first frame.")
    exit()

prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video.")
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_diff = cv2.absdiff(gray, prev)
    ret, thres = cv2.threshold(frame_diff, 35, 255, cv2.THRESH_BINARY)
    prev = gray.copy()
    
    cv2.imshow('original', frame)
    cv2.imshow('foregroundMask', thres)
    
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
