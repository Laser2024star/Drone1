from ultralytics import YOLO
import torch

def main():
    # 检查GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 加载融合模型作为预训练
    model_path = "models/fusion_best.pt"
    print(f"加载预训练模型: {model_path}")
    model = YOLO(model_path)
    
    # 开始微调
    results = model.train(
        data="data/visdrone/visdrone.yaml",
        epochs=30,
        imgsz=1280,
        batch=16,
        lr0=0.0001,
        device=device,
        name="visdrone_finetune",
        exist_ok=True,
        val=True,
        save=True,
        save_period=5,
        verbose=True
    )
    
    print("\n" + "="*50)
    print("✅ VisDrone 微调完成！")
    print(f"最佳模型保存在: {results.save_dir}/weights/best.pt")
    print("="*50)

if __name__ == "__main__":
    main()