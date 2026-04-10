import cv2
import os
# 解決 OpenMP 衝突問題
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from ultralytics import YOLO

def main():
    # 1. 指向你訓練好的模型路徑 (請確認 orange_project2 是你最後產出的資料夾)
    model_path = r"C:\Users\user\runs\detect\Fridge_project4\weights\best.pt"
    
    if not os.path.exists(model_path):
        print(f"❌ 找不到模型檔案：{model_path}")
        print("請確認訓練是否已完成，或路徑名稱是否正確。")
        return

    # 2. 載入訓練好的模型
    model = YOLO(model_path)

    # 3. 開啟攝影機 (0 通常是內建鏡頭)
    cap = cv2.VideoCapture(0)

    print("--- 系統啟動中，按下 'q' 鍵可結束程式 ---")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("無法讀取攝影機畫面")
            break

        # 4. 執行辨識 (使用 stream=True 效能較好)
        results = model(frame, stream=True)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                # 取得座標、類別索引、信心值
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls]

                # 5. 設定顏色：如果是 rotten orange 就用紅色 (0,0,255)，否則用綠色 (0,255,0)
                color = (0, 0, 255) if label == "rotten orange" else (0, 255, 0)
                thickness = 3 if label == "rotten orange" else 2

                # 6. 畫出矩形框
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

                # 7. 顯示文字標籤
                text = f"{label} {conf:.2f}"
                cv2.putText(frame, text, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 顯示畫面
        cv2.imshow("Orange Detection Real-time", frame)

        # 按下 'q' 離開
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 釋放資源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()