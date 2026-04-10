import os

# --- 加入這行解決 OMP: Error #15 ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# ----------------------------------

import sys
from ultralytics import YOLO

def run_training():
    # 2. 自動鎖定程式碼所在路徑
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # 3. 定義資料集名稱 (請確認你的資料夾真的叫 my_dataset)
    dataset_folder = "my_dataset"
    yaml_name = "data.yaml"
    
    full_yaml_path = os.path.join(BASE_DIR, dataset_folder, yaml_name)

    # --- 診斷測試：列出路徑內容 ---
    print(f"--- 路徑檢查中 ---")
    print(f"目前執行位置: {BASE_DIR}")
    
    if os.path.exists(os.path.join(BASE_DIR, dataset_folder)):
        print(f"✅ 找到資料夾: {dataset_folder}")
        files = os.listdir(os.path.join(BASE_DIR, dataset_folder))
        print(f"📁 資料夾內包含: {files}")
    else:
        print(f"❌ 找不到資料夾: {dataset_folder}")
        print(f"👉 請確保 {dataset_folder} 資料夾放在 {BASE_DIR} 裡面")
        return

    # 4. 載入模型 (若無 pt 檔會自動下載)
    model = YOLO("yolo11m.pt") 

    # 5. 開始訓練
    try:
        model.train(
            data=full_yaml_path,
            epochs=300,
            imgsz=640,
            batch=8,
            patience=30,

            box=7.5,
            cls=2.0,
            dfl=1.5,

            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,

            mixup=0.2,
            cos_lr=True,

            plots=True,
            name="Fridge_project"
        )
    except Exception as e:
        print(f"❌ 訓練發生錯誤: {e}")

if __name__ == "__main__":
    run_training()