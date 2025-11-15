import cv2

video_path = r'..\Videos\SpongeBob SquarePants - Writing Essay - Some of These - Meme Source.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Cannot open video file!")
    exit()

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("Total frames:", frame_count)

ret, frame = cap.read()
if ret:
    print("First frame shape:", frame.shape)
else:
    print("Error reading first frame")
cap.release()
