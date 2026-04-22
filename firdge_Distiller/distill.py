import sys
import os
import torch
    

# 將本地的 ultralytics 加入路徑，優先使用修改過的版本
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, r"C:\Users\user\Downloads\Firdge_Distiller\Firdge_Distiller\yolo-distiller-main")
#sys.path.insert(0, os.path.join(project_root, 'yolo-distiller-main'))
print(sys.path[:3])

from ultralytics import YOLO

def main():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # 載入指定的教師模型
    teacher_path = os.path.join(project_root, 'models', 'best.pt')
    if not os.path.exists(teacher_path):
        print(f"錯誤：找不到教師模型 {teacher_path}")
        return

    print(f"正在載入教師模型：{teacher_path}")
    teacher_model = YOLO(teacher_path).model 
    teacher_model.eval()

    for p in teacher_model.parameters():
        p.requires_grad = False

    # 選擇學生模型（這裡使用 YOLOv8n 作為小型學生模型）
    student_model_name = "yolo11n.pt"
    print(f"正在初始化學生模型：{student_model_name}")
    student_model = YOLO("yolo11n.yaml") 
    student_model.model.to(device)

    # 開始蒸餾訓練
    data_yaml = os.path.join(project_root, 'data.yaml')
    
    print(f"\n" + "="*50)
    print(f"訓練資訊摘要:")
    print(f"  教師模型: {teacher_path}")
    print(f"  學生模型: {student_model_name}")
    print(f"  蒸餾損失: mgd")
    print(f"  資料配置: {data_yaml}")
    print("="*50 + "\n")

    print("開始蒸餾訓練...")
    
    print(f"環境診斷:")
    print(f"  PyTorch 版本: {torch.__version__}")
    print(f"  CUDA 是否可用: {torch.cuda.is_available()}")
    print(f"  Python 執行路徑: {sys.executable}")
    print(f"  使用設備: {device}")
    print("Teacher type:", type(teacher_model))
    print("Student type:", type(student_model))
    print("KD config check:")
    print("teacher loaded")
    print("distill:", "cwd")
    print("Teacher requires_grad check:",
      any(p.requires_grad for p in teacher_model.model.parameters()))

    student_model.train(
        data=data_yaml,
        teacher=teacher_model,
        distillation_loss="cwd", # "mgd" (預設)：適合從頭學習特徵圖。"cwd"：適合強化類別判斷能力。
        epochs=50,
        batch=2,          # 11n 模型較小，建議使用較大的 batch 以加速收斂
        imgsz=640,
        workers=0,         # Windows 系統下設為 0 最穩定，避免多進程引發的 Hook 錯誤
        device=device,
        exist_ok=True,
        hsv_s=0.5,
        hsv_v=0.3,

        project=os.path.join(project_root, "results"),
        name="kd_yolo11n"
    )

if __name__ == "__main__":
    main()
