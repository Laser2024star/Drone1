from ultralytics import YOLO
import torch
import os

def main():
    # 检查GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 检查模型文件是否存在
    model_path = "models/best.pt"
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        print("请确认 models/best.pt 是否存在")
        return
    
    # 检查数据集配置文件
    data_yaml = "data/uavdb.yaml"
    if not os.path.exists(data_yaml):
        print(f"❌ 数据集配置文件不存在: {data_yaml}")
        print("请确认 data/uavdb.yaml 是否创建正确")
        return
    
    print(f"加载模型: {model_path}")
    print(f"数据集配置: {data_yaml}")
    
    # 加载你们训练好的 best.pt
    model = YOLO(model_path)
    
    # 在 UAVDB 上微调
    results = model.train(
        data=data_yaml,
        epochs=50,
        imgsz=1280,
        batch=16,
        device=device,
        lr0=0.0001,
        warmup_epochs=3,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        scale=0.5,
        fliplr=0.5,
        project='runs/train',
        name='uavdb_finetune',
        exist_ok=True,
        val=True,
        save=True,
        save_period=10,
        verbose=True,
    )
    
    print("\n" + "="*50)
    print("✅ 微调完成！")
    print(f"最佳模型保存在: {results.save_dir}/weights/best.pt")
    print("="*50)

if __name__ == "__main__":
    main()
    