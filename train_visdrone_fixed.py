import torch
import os
import sys
from ultralytics import YOLO

# 设置环境变量避免多进程问题
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['PYTHONHASHSEED'] = '0'

# 清理所有缓存文件
import shutil

print("="*60)
print("清理所有缓存文件...")
print("="*60)

cache_patterns = [
    'data/visdrone/train/*.cache',
    'data/visdrone/val/*.cache',
    'data/visdrone/test/*.cache',
    'runs/**/*.cache'
]

import glob
for pattern in cache_patterns:
    for cache_file in glob.glob(pattern, recursive=True):
        try:
            os.remove(cache_file)
            print(f"删除: {cache_file}")
        except:
            pass

print("\n" + "="*60)
print("开始训练配置")
print("="*60)

# 检查GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用设备: {device}")

if device == 'cuda':
    torch.cuda.empty_cache()
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU显存: {gpu_mem:.1f} GB")
    
    # 最保守配置
    batch = 1
    imgsz = 512
    print(f"使用保守配置: batch={batch}, imgsz={imgsz}")

# 加载模型 - 使用最小的模型
print("\n加载模型: yolov8n.pt")
model = YOLO('yolov8n.pt')

# 训练配置 - 完全禁用所有可能引起内存问题的选项
print("\n开始训练...")
print("="*60)

try:
    results = model.train(
        data='data/visdrone/visdrone.yaml',
        epochs=50,                    # 先用50轮测试
        imgsz=imgsz,
        batch=batch,
        device=device,
        
        # 禁用所有缓存
        cache=False,                  # 禁用图像缓存
        workers=0,                    # 禁用多进程
        rect=False,                   # 禁用矩形训练
        
        # 基础优化器
        optimizer='SGD',              # 使用SGD代替AdamW，更稳定
        lr0=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        
        # 简化数据增强
        mosaic=1.0,
        mixup=0.0,                    # 禁用mixup
        copy_paste=0.0,               # 禁用copy_paste
        erasing=0.0,                  # 禁用erasing
        
        # 基本几何增强
        scale=0.3,
        degrees=5.0,
        translate=0.05,
        hsv_h=0.01,
        hsv_s=0.3,
        hsv_v=0.3,
        flipud=0.1,
        fliplr=0.5,
        
        # 训练控制
        close_mosaic=10,
        patience=20,
        save=True,
        save_period=10,
        plots=True,
        
        # 输出
        project='runs/train',
        name='visdrone_fixed',
        exist_ok=True,
        verbose=True
    )
    
    print("\n" + "="*60)
    print("✅ 训练成功开始！")
    print(f"最佳模型: {results.save_dir}/weights/best.pt")
    print("="*60)
    
except Exception as e:
    print(f"\n❌ 训练失败: {e}")
    print("\n尝试解决方案:")
    print("1. 重启电脑释放内存")
    print("2. 使用CPU训练: device='cpu'")
    print("3. 进一步降低 imgsz=320")