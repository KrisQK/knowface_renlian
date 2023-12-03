import cv2

# 创建一个 VideoCapture 对象，指定使用默认的摄像头（编号为 0）。
# 这个对象用于捕获摄像头视频流。
# 初始化视频捕获对象：cv2.VideoCapture(0) 
# 创建了一个捕获摄像头视频流的对象。

video = cv2.VideoCapture(0)

# 加载人脸检测器模型文件（XML 文件）。
# 注意：此处路径需要根据实际存储位置进行修改。
# 加载了一个已经训练好的人脸检测器模型。
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

#通过循环 while True 持续捕获摄像头的视频帧，
# 并在每一帧上执行以下步骤：
while True:
    # 读取摄像头的一帧。
    # 返回两个值：ret 是一个布尔值，表示是否成功读取帧；
    # frame 是图像帧的内容。
    ret, frame = video.read()

    # 将彩色图像转换为灰度图像，因为人脸检测通常在灰度图上进行。
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # 使用人脸检测器检测灰度图中的人脸。
    # facedetect.detectMultiScale() 在灰度图像中检测人脸，
    # 并返回矩形框的位置信息。
    # detectMultiScale 返回检测到的人脸的位置信息
    # （矩形框的坐标和尺寸）。
    faces = facedetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # 遍历检测到的人脸位置信息，并在每个位置绘制一个红色矩形框。
    # 这里 (x, y) 是矩形框的左上角坐标，w 和 h 分别是宽度和高度。
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
        # 绘制矩形框的参数依次为：图像、左上角坐标、右下角坐标、颜色（BGR 格式）、线条粗细。

    # 显示带有矩形框的原始视频帧。
    cv2.imshow("Face Detection", frame)

    # 等待用户按下键盘上的任意键。
    # 如果按下键盘上的 "q" 键，退出程序。
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

# 释放摄像头资源。
video.release()

# 关闭所有打开的窗口。
cv2.destroyAllWindows()
