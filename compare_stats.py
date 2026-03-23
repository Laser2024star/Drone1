from ultralytics import YOLO
import cv2
import numpy as np
import time
import pandas as pd

def analyze_video(model_path, video_path, model_name):
    """
    分析视频的检测效果，统计各项指标
    """
    print(f"\n正在分析 {model_name}...")
    
    # 加载模型
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    
    # 获取视频信息
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"视频信息: {width}x{height}, {fps}FPS, 总帧数:{total_frames}")
    
    # 统计指标
    inference_times = []
    detections_per_frame = []
    confidence_scores = []
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 计时推理
        start = time.time()
        results = model(frame, imgsz=1280, verbose=False)[0]
        inference_time = (time.time() - start) * 1000  # 毫秒
        inference_times.append(inference_time)
        
        # 统计检测结果
        if results.boxes is not None:
            num_detections = len(results.boxes)
            confs = results.boxes.conf.cpu().numpy()
            
            detections_per_frame.append(num_detections)
            confidence_scores.extend(confs)
        
        frame_count += 1
        if frame_count % 50 == 0:
            print(f"已处理 {frame_count}/{total_frames} 帧...")
    
    cap.release()
    
    # 计算统计结果
    stats = {
        "模型名称": model_name,
        "平均推理时间(ms)": round(np.mean(inference_times), 2),
        "最慢推理时间(ms)": round(np.max(inference_times), 2),
        "最快推理时间(ms)": round(np.min(inference_times), 2),
        "理论最大FPS": round(1000 / np.mean(inference_times), 1),
        "平均每帧检测数": round(np.mean(detections_per_frame), 2) if detections_per_frame else 0,
        "最多检测数(单帧)": int(np.max(detections_per_frame)) if detections_per_frame else 0,
        "平均置信度": round(np.mean(confidence_scores), 3) if confidence_scores else 0,
        "总检测目标数": len(confidence_scores),
    }
    
    return stats

def compare_models():
    """对比原始模型和优化模型"""
    
    # 两个模型路径
    models = [
        ("原始模型 (yolov8m)", "yolov8m.pt"),
        ("优化模型 (best.pt)", "models/best.pt")
    ]
    
    # 测试视频路径
    video_path = "data/input/sample4.mp4"
    
    results = []
    
    for name, path in models:
        try:
            stats = analyze_video(path, video_path, name)
            results.append(stats)
        except Exception as e:
            print(f"分析 {name} 失败: {e}")
    
    # 创建对比表格
    df = pd.DataFrame(results)
    
    # 计算提升百分比
    if len(results) == 2:
        orig = results[0]
        opt = results[1]
        
        print("\n" + "="*60)
        print("📊 效果对比分析")
        print("="*60)
        
        for key in orig.keys():
            if key == "模型名称":
                continue
            
            orig_val = orig[key]
            opt_val = opt[key]
            
            if isinstance(orig_val, (int, float)):
                change = opt_val - orig_val
                change_percent = (change / orig_val) * 100 if orig_val != 0 else 0
                
                # 判断指标好坏方向
                if key in ["平均推理时间(ms)", "最慢推理时间(ms)"]:
                    # 时间越小越好
                    arrow = "↓" if change < 0 else "↑" if change > 0 else "→"
                    better = "✅ 提升" if change < 0 else "⚠️ 变慢" if change > 0 else "➡️ 持平"
                else:
                    # 其他指标越大越好
                    arrow = "↑" if change > 0 else "↓" if change < 0 else "→"
                    better = "✅ 提升" if change > 0 else "⚠️ 下降" if change < 0 else "➡️ 持平"
                
                print(f"{key:<20}: {orig_val:<10} → {opt_val:<10} {arrow} {change_percent:+.1f}% {better}")
    
    # 保存到CSV
    df.to_csv("model_comparison_stats.csv", index=False)
    print("\n✅ 详细数据已保存到 model_comparison_stats.csv")
    
    return results

if __name__ == "__main__":
    compare_models()