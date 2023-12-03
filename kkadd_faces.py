import os
import pickle
import cv2
import numpy as np
# 创建一个 VideoCapture 对象，指定使用默认的摄像头（编号为 0）。
# 这个对象用于捕获摄像头视频流。
# 初始化视频捕获对象：cv2.VideoCapture(0) 
# 创建了一个捕获摄像头视频流的对象。

video = cv2.VideoCapture(0)

# 加载人脸检测器模型文件（XML 文件）
# 注意：此处路径需要根据实际存储位置进行修改。
# 加载了一个已经训练好的人脸检测器模型。
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# 用于存储捕获的人脸数据的空列表
faces_data = []

# 计数器，用于控制存储人脸数据的数量
i = 0
# 输入用户的姓名
name = input("Enter Your Name: ")

#通过循环 while True 持续捕获摄像头的视频帧，
# 并在每一帧上执行以下步骤：
while True:
    # 读取摄像头的一帧。
    # 返回两个值：ret 是一个布尔值，表示是否成功读取帧；
    # frame 是图像帧的内容。
    #frame里面是matlike,每个数字代表图像中对应位置的灰度值。这是一个简化的示例，实际图像的数组会更大、更复杂
    ret, frame = video.read()
    #print(frame)
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
        crop_img = frame[y:y+h, x:x+w, :]  # 截取人脸区域
        resized_img = cv2.resize(crop_img, (50, 50))  # 将截取的人脸图像调整为50x50像素大小
        # 如果人脸数据列表长度小于等于100并且满足每隔10帧存储一次的条件
        if len(faces_data) <= 100 and i % 10 == 0:
            faces_data.append(resized_img)  # 将人脸数据添加到列表中

        i += 1  # 帧计数器递增
        # 在图像上标注当前存储的人脸数量
        cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
        
        # 绘制矩形框的参数依次为：图像、左上角坐标、右下角坐标、颜色（BGR 格式）、线条粗细。
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 4)

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

# 将存储的人脸数据转换为NumPy数组，并重塑形状为100x（宽*高*通道数）
faces_data = np.asarray(faces_data)
faces_data = faces_data.reshape(100, -1)

# 如果 'names.pkl' 文件不存在
if 'names.pkl' not in os.listdir('data/'):
    names = [name] * 100  # 创建一个包含姓名的列表，重复100次
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)  # 将姓名列表保存为文件
else:
    # 如果 'names.pkl' 文件存在，则读取已有的姓名列表并添加新的姓名
    with open('data/names.pkl', 'rb') as f:
        names = pickle.load(f)
    names = names + [name] * 100  # 将新姓名添加到列表中，重复100次
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)  # 将更新后的姓名列表保存为文件

# 如果 'faces_data.pkl' 文件不存在
if 'faces_data.pkl' not in os.listdir('data/'):
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces_data, f)  # 将人脸数据保存为文件
else:
    # 如果 'faces_data.pkl' 文件存在，则读取已有的人脸数据并添加新的数据
    with open('data/faces_data.pkl', 'rb') as f:
        faces = pickle.load(f)
    faces = np.append(faces, faces_data, axis=0)  # 沿着数组的行方向拼接数据
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces, f)  # 将更新后的人脸数据保存为文件

        
        
# 视频捕获和人脸检测：

# 使用 OpenCV 的 cv2.VideoCapture() 创建了一个视频捕获对象，捕获来自默认摄像头的视频流。
# 使用 cv2.CascadeClassifier() 加载了一个训练好的人脸检测器模型。
# 在一个无限循环中，程序持续地读取视频流的每一帧，将其转换为灰度图像，然后使用人脸检测器检测图像中的人脸位置，并在检测到的人脸周围绘制红色矩形框。
# 数据收集：

# 每隔10帧，程序会截取人脸图像，并将其缩放为50x50像素大小，然后将这些图像数据存储到名为 faces_data 的列表中，最多存储100张图像。
# 同时，程序记录了存储的人脸数量，并在视频上显示这个数字和红色矩形框。
# 数据保存：

# 将收集到的人脸图像数据转换为 NumPy 数组，并将其重塑为100x（宽高通道数）的形状。
# 将姓名与这些人脸图像数据相关联。如果之前未保存过姓名和人脸数据的文件，则创建并保存名为 names.pkl 和 faces_data.pkl 的文件；如果文件已存在，则读取已有的数据并将新的数据添加进去。
# 这段代码的作用是从摄像头捕获人脸图像并将其与姓名相关联，存储为NumPy数组，并将其保存到文件中。