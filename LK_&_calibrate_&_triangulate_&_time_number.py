import cv2 as cv  
import glob  
import numpy as np  
import time
import pandas as pd
import os  

def calibrate_camera(name, rows, columns, world_scaling, images_folder):
    # rows = checkerboard的行數
    # columns = checkerboard的列數
    # world_scaling = 真實世界方格的大小(毫米)

    images_names = sorted(glob.glob(images_folder))  # 獲取文件夾中的圖像文件名並排序
    images = []
    for imname in images_names:
        im = cv.imread(imname, 1)  # 讀取圖像文件
        images.append(im)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)  # 設置校準準則

    objp = np.zeros((rows * columns, 3), np.float32)  # checkerboard世界空間中的方格座標
    objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
    objp = world_scaling * objp

    width = images[0].shape[1]  # 圖像的寬度
    height = images[0].shape[0]  # 圖像的高度

    imgpoints = []  # checkerboard的像素座標
    objpoints = []  # checkerboard在checkerboard世界空間中的座標

    for frame in images:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  # 將圖像轉換為灰度圖像

        ret, corners = cv.findChessboardCorners(gray, (rows, columns), None)  # 查找checkerboard

        if ret == True:
            conv_size = (11, 11)  # 用於改善角點檢測的卷積大小

            corners = cv.cornerSubPix(gray, corners, conv_size, (-1, -1), criteria)  # 改善checkerboard座標
            cv.drawChessboardCorners(frame, (rows, columns), corners, ret)
            cv.imshow(name, frame)
            k = cv.waitKey(500)

            objpoints.append(objp)
            imgpoints.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (width, height), None, None)  # 相機校準
    print(name, 'rmse:', ret)  # 打印校準結果
    print(name, '相機矩陣:\n', mtx)  # 打印相機矩陣
    print(name, '失真係數:\n', dist)  # 打印失真係數
    print(name, '旋轉向量:\n', rvecs)  # 打印旋轉向量
    print(name, '位移向量:\n', tvecs)  # 打印位移向量

    return mtx, dist

def stereo_calibrate(rows, columns, world_scaling, mtx1, dist1, mtx2, dist2, frames_folder1, frames_folder2):
    # rows = checkerboard的行數
    # columns = checkerboard的列數
    # world_scaling = 真實世界方格的大小(毫米)

    images_names1 = glob.glob(frames_folder1)  # 讀取相機1影格的檔案名稱
    images_names2 = glob.glob(frames_folder2)  # 讀取相機2影格的檔案名稱
    images_names1 = sorted(images_names1)
    images_names2 = sorted(images_names2)
    
    c1_images_names = images_names1
    c2_images_names = images_names2

    c1_images = []  # 存放相機1的影格
    c2_images = []  # 存放相機2的影格

    for im1, im2 in zip(c1_images_names, c2_images_names):
        _im = cv.imread(im1, 1)
        c1_images.append(_im)

        _im = cv.imread(im2, 1)
        c2_images.append(_im)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.0001)  # 校準條件

    objp = np.zeros((rows * columns, 3), np.float32)  # 棋盤格在世界空間中的3D座標
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling * objp

    width = c1_images[0].shape[1]  # 影格的寬度
    height = c1_images[0].shape[0]  # 影格的高度

    imgpoints_left = []  # 左相機的2D像素坐標
    imgpoints_right = []  # 右相機的2D像素坐標
    objpoints = []  # 真實世界空間中的3D坐標

    for frame1, frame2 in zip(c1_images, c2_images):
        gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        c_ret1, corners1 = cv.findChessboardCorners(gray1, (rows, columns), None)
        c_ret2, corners2 = cv.findChessboardCorners(gray2, (rows, columns), None)

        if c_ret1 == True and c_ret2 == True:
            corners1 = cv.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)

            cv.drawChessboardCorners(frame1, (rows, columns), corners1, c_ret1)
            cv.imshow('LEFT', frame1)

            cv.drawChessboardCorners(frame2, (rows, columns), corners2, c_ret2)
            cv.imshow('RIGHT', frame2)
            k = cv.waitKey(500)

            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)

    stereocalibration_flags = cv.CALIB_FIX_INTRINSIC
    ret, CM1, dist1, CM2, dist2, R, T, E, F = cv.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, mtx1, dist1,
                                                                 mtx2, dist2, (width, height), criteria=criteria, flags=stereocalibration_flags)

    print(f"平均重投影誤差 : {ret}")  # 打印校準結果
    return R, T

def triangulate(mtx1, mtx2, R, T, xL, yL, xR, yR):
    # 構造相機矩陣P1和P2
    RT1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    P1 = mtx1 @ RT1

    RT2 = np.hstack((R, T.reshape(-1, 1)))
    P2 = mtx2 @ RT2

    # 將點座標轉換為二維齊次座標形式
    points1 = np.array([xL, yL, 1]).reshape(-1, 1)
    points2 = np.array([xR, yR, 1]).reshape(-1, 1)

    # 使用OpenCV的triangulatePoints函数
    points4D = cv.triangulatePoints(P1, P2, points1[:2], points2[:2])

    # 將齊次座標轉換為3D座標
    points3D = points4D / points4D[3]
    
    return points3D[:3].ravel()

def pickROI(winName, img, interpolation=cv.INTER_LINEAR):
    # 初始影像的預設寬度和高度
    maxW, maxH = img.shape[1], img.shape[0]
    # 初始化影像的顯示區域
    x0, y0, x1, y1 = 0, 0, maxW, maxH
    
    while True:
        # 計算顯示比例
        scale_x = maxW / (x1 - x0)
        scale_y = maxH / (y1 - y0)
        scale = min(scale_x, scale_y)

        # 根據比例調整影像大小
        imgResized = cv.resize(img[y0:y1,x0:x1], None, fx=scale, fy=scale, interpolation=interpolation)
        
        # 使用cv.selectROI選擇區域
        roi = cv.selectROI(winName, imgResized, fromCenter=False)
        if roi[2] > 0 and roi[3] > 0:  # 確保選擇的ROI是有效的
            # 計算ROI在原圖中的座標
            roi_x0 = int((roi[0] / scale) + x0)
            roi_y0 = int((roi[1] / scale) + y0)
            roi_x1 = int((roi[0] + roi[2]) / scale) + x0
            roi_y1 = int((roi[1] + roi[3]) / scale) + y0
            
            # 更新坐標以進一步放大或確認選擇
            x0, y0, x1, y1 = roi_x0, roi_y0, roi_x1, roi_y1
            w = x1 - x0
            h = y1 - y0
            
            # 如果用戶滿意選擇，按ESC鍵退出循環
            key = cv.waitKey(0) & 0xFF
            if key == 27:  # ESC key
                cv.destroyAllWindows()
                break
        else:
            # 如果使用者沒有選擇有效區域，直接退出
            cv.destroyAllWindows()
            return None

    return (x0, y0, w, h)

# 計算相機校準參數
mtx1, dist1 = calibrate_camera("LEFT", 8, 5, 29.0115, 'D:/Code/school_paper/LK_&_triangulate/stereo_images/stereo_imagesL/*')
mtx2, dist2 = calibrate_camera("RIGHT", 8, 5, 29.0115, 'D:/Code/school_paper/LK_&_triangulate/stereo_images/stereo_imagesR/*')
 
# 進行立體校準
R, T = stereo_calibrate(8, 5, 29.0115, mtx1, dist1, mtx2, dist2, 'D:/Code/school_paper/LK_&_triangulate/stereo_images/stereo_imagesL/*', 'D:/Code/school_paper/LK_&_triangulate/stereo_images/stereo_imagesR/*')

# 將光流法檢測的參數放入一個字典中
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

cam0 = cv.VideoCapture(0)  # 初始化攝像頭(左)
cam1 = cv.VideoCapture(1)  # 初始化攝像頭(右)

# 獲取幀速率
fps0 = cam0.get(cv.CAP_PROP_FPS)  # 獲取攝像頭的幀速率
fps1 = cam1.get(cv.CAP_PROP_FPS)  # 獲取攝像頭的幀速率
print(f"左相機的預設FPS：{fps0} \n右相機的預設FPS：{fps1}")  # 打印幀速率信息

frame_count = 0  # 初始化帧數計數器

while True:
    retval0, img0 = cam0.read()  # 讀取圖像
    retval1, img1 = cam1.read()  # 讀取圖像
    cv.imshow("Cam0_First_image", img0)  # 顯示圖像
    cv.imshow("Cam1_First_image", img1)  # 顯示圖像
    ikey = cv.waitKey(1)  # 等待按鍵事件
    if ikey == 32:  # 空格鍵退出
        break

# 選擇多個特徵區塊
rois0 = []  # 存儲選擇的ROI
rois1 = []  # 存儲選擇的ROI
features_num = 1 # 允許選擇幾個ROI -------------------------------------------這個可以調整幾個特徵點區塊-------------------------------------------　
for i in range(features_num):  
    roi0 = pickROI(f"Cam0_Pick_image_{i+1}", img0)  # 選擇ROI
    roi1 = pickROI(f"Cam1_Pick_image_{i+1}", img1)  # 選擇ROI
    if roi0[2] > 0 and roi0[3] > 0:  # 確保選擇的ROI是有效的
        rois0.append(roi0)
    if roi1[2] > 0 and roi1[3] > 0:  # 確保選擇的ROI是有效的
        rois1.append(roi1)
    if len(rois0) < i + 1:  # 如果用戶選擇取消，則停止選擇
        break
    if len(rois1) < i + 1:  # 如果用戶選擇取消，則停止選擇
        break

# 計算每個ROI中心的二維座標，並進行三角化以獲得初始的三維座標
initial_p3ds = []  # 儲存初始三維座標

for i in range(len(rois0)):
    # 計算ROI中心點座標
    x0, y0, w0, h0 = rois0[i]
    x1, y1, w1, h1 = rois1[i]
    centerX0 = x0 + w0 / 2
    centerY0 = y0 + h0 / 2
    centerX1 = x1 + w1 / 2
    centerY1 = y1 + h1 / 2

    # 使用中心點座標進行三角化
    p3d = triangulate(mtx1, mtx2, R, T, centerX0, centerY0, centerX1, centerY1)
    initial_p3ds.append(p3d)
    
# 初始化用於儲存資料的列表
distance_data = []

cv.destroyAllWindows()  # 關閉所有OpenCV窗口

# 開始時間
start_all = time.time()  # 記錄開始時間

# 添加限制時間
end_time = start_all + 20

while True:

    os.system('cls' if os.name == 'nt' else 'clear')  # 清空屏幕

    if time.time() > end_time:  # 檢查是否超過10秒
        break  # 如果超過10秒，則跳出循環

    retval00, img00 = cam0.read()  # 讀取下一幀圖像
    retval01, img01 = cam1.read()  # 讀取下一幀圖像

    roi_info_0 = []  # 用來收集每個ROI的信息
    roi_info_1 = []  # 用來收集每個ROI的信息

    for i, roi in enumerate(rois0):
        x0, y0, w, h = roi  # 解析ROI
        prevPts = np.array([(x0 + w / 2), (y0 + h / 2)], dtype=np.float32).reshape(-1, 1, 2)  # 計算ROI中心點
        nextPts, st, err = cv.calcOpticalFlowPyrLK(img0, img00, prevPts, None, **lk_params)  # 計算光流以跟蹤ROI

        p1_x = nextPts[0][0][0]  # 獲得新中心點x座標
        p1_y = nextPts[0][0][1]  # 獲得新中心點y座標
        rec_x0 = int(p1_x - w / 2)  # 計算新的ROI左上角x座標
        rec_y0 = int(p1_y - h / 2)  # 計算新的ROI左上角y座標
        rec_x1 = rec_x0 + w  # 計算新的ROI右下角x座標
        rec_y1 = rec_y0 + h  # 計算新的ROI右下角y座標
        cv.rectangle(img00, (rec_x0, rec_y0), (rec_x1, rec_y1), (0, 255, 0), 2)  # 畫出新的ROI矩形

        # 收集每個ROI的信息
        roi_info_0.append((p1_x, p1_y))

    for i, roi in enumerate(rois1):
        x0, y0, w, h = roi  # 解析ROI
        prevPts = np.array([(x0 + w / 2), (y0 + h / 2)], dtype=np.float32).reshape(-1, 1, 2)  # 計算ROI中心點
        nextPts, st, err = cv.calcOpticalFlowPyrLK(img1, img01, prevPts, None, **lk_params)  # 計算光流以跟蹤ROI

        p1_x = nextPts[0][0][0]  # 獲得新中心點x座標
        p1_y = nextPts[0][0][1]  # 獲得新中心點y座標
        rec_x0 = int(p1_x - w / 2)  # 計算新的ROI左上角x座標
        rec_y0 = int(p1_y - h / 2)  # 計算新的ROI左上角y座標
        rec_x1 = rec_x0 + w  # 計算新的ROI右下角x座標
        rec_y1 = rec_y0 + h  # 計算新的ROI右下角y座標
        cv.rectangle(img01, (rec_x0, rec_y0), (rec_x1, rec_y1), (0, 255, 0), 2)  # 畫出新的ROI矩形

        # 收集每個ROI的信息
        roi_info_1.append((p1_x, p1_y))

    img00 = cv.resize(img00, (0 , 0), fx = 1, fy = 1)
    img01 = cv.resize(img01, (0 , 0), fx = 1, fy = 1)

    cv.imshow("Cam0_Image_tracking", img00)  # 顯示跟蹤後的圖像
    cv.imshow("Cam1_Image_tracking", img01)  # 顯示跟蹤後的圖像

    # 使用所得的座標進行三角化計算，並計算與初始三維座標的距離
    for i in range(min(len(roi_info_0), len(roi_info_1))):
        xL, yL = roi_info_0[i]
        xR, yR = roi_info_1[i]
        # 假設triangulate函數已正確定義，且mtx1, mtx2, R, T等參數已準備好
        p3d = triangulate(mtx1, mtx2, R, T, xL, yL, xR, yR)

        # 計算與初始三維座標的距離
        initial_p3d = initial_p3ds[i]  # 取得初始三維座標
        distance = p3d - initial_p3d  # 計算距離
        physical_distance = np.linalg.norm(distance)

        # 即時輸出每個ROI的目前三維座標和與初始位置的距離
        print(f"ROI {i+1} 的 X 方向距离: {distance[0]:>20.10f}, Y 方向距离: {distance[1]:>20.10f}, Z 方向距离: {distance[2]:>20.10f}, 与初始位置的距离: {np.linalg.norm(distance)}")

        # 累積每個ROI的距離數據
        distance_data.append((i+1, distance[0], distance[1], distance[2], physical_distance))

    frame_count += 1  # 增加幀數計數器
    
    ikey = cv.waitKey(1)  # 等待按鍵事件
    if ikey == 27:  # ESC鍵退出
        break

# 結束時間
end_all = time.time()  # 記錄結束時間

seconds = end_all - start_all  # 計算時間間隔，即捕獲所有幀所花費的時間
fps_all = frame_count / seconds  # 計算幀速率

print(f"\n實驗總花費時間：{seconds:.10f} 秒\n實驗總平均 FPS: {fps_all:.10f}")  # 打印估計的幀速率信息
 
# 將累積的資料轉換為pandas DataFrame
df = pd.DataFrame(distance_data, columns=['ROI', 'X方向距離', 'Y方向距離', 'Z方向距離', '與初始位置的總距離'])

# 指定Excel檔案路徑
excel_path = "C:/Users/allen/OneDrive/桌面/All_distance_data.xlsx" 

# 保存DataFrame到Excel文件
df.to_excel(excel_path, index=False)

# 釋放資源
cam0.release()
cam1.release()
cv.destroyAllWindows()  # 關閉所有OpenCV窗口
