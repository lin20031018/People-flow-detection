import cv2
import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

# 顯式允許 DetectionModel 作為安全全局類
torch.serialization.add_safe_globals([DetectionModel])

# 初始化 YOLOv8 模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO('yolov8n.pt').to(device)

# 動態定義 ROI (可調整為其他方法來動態更新 ROI)
def is_inside_roi(bbox, roi):
    x1, y1, x2, y2 = bbox
    rx1, ry1, rx2, ry2 = roi
    return rx1 <= x1 <= rx2 and ry1 <= y1 <= ry2

# 繪製警示資訊
def draw_info(frame, people_count):
    if people_count == 1:
        color = (0, 255, 0)  # 綠色
        status = "Sparse"
    elif people_count == 2:
        color = (0, 255, 255)  # 黃色
        status = "Moderate"
    elif people_count >= 3:
        color = (0, 0, 255)  # 紅色
        status = "Crowded"
    else:
        color = (255, 255, 255)
        status = "No queue"

    cv2.putText(frame, f'People in ROI: {people_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, status, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

# 定義 ROI 區域
# 根據您提供的圖片位置進行調整
roi = (500, 100, 850, 550)  # 黃框的位置 (x1, y1, x2, y2)

# 讀取影片
video_path = "C:/Users/mray3/桌面/video.mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 調整畫面大小為 1280x720
    frame = cv2.resize(frame, (1280, 720))

    # YOLOv8 偵測
    results = model(frame)
    detections = []

    people_in_roi = 0

    for result in results:  # 每幀的結果
        for box in result.boxes:  # 遍歷每個檢測框
            x1, y1, x2, y2 = box.xyxy[0].tolist()  # 提取邊界框座標
            conf = box.conf[0].item()  # 提取信心度
            cls = box.cls[0].item()  # 提取類別
            if cls == 0 and conf > 0.2:  # 0 是人類的標籤
                # 確保 YOLO 偵測框映射到原圖尺寸
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                detections.append((x1, y1, x2, y2))

                # 在畫面中繪製 YOLO 偵測框 (藍色)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # YOLO 偵測框 (藍色)
                cv2.putText(frame, "YOLO Detection", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # 檢查是否進入 ROI
                if is_inside_roi((x1, y1, x2, y2), roi):
                    people_in_roi += 1
                    cv2.putText(frame, "Inside ROI", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 繪製 ROI 區域
    cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), (0, 0, 255), 2)

    # 顯示警示資訊
    draw_info(frame, people_in_roi)

    # 顯示畫面
    cv2.imshow("YOLOv8 Monitoring", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放資源
cap.release()
cv2.destroyAllWindows()
