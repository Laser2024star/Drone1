from ultralytics import YOLO
import cv2
import numpy as np

def quick_frame_compare(frame_path="test_frame.jpg"):
    """
    快速对比单帧图像的效果
    """
    # 如果没有测试图片，从视频中提取一帧
    if not cv2.imread(frame_path) is not None:
        cap = cv2.VideoCapture("data/input/sample4.mp4")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 100)  # 取第100帧
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(frame_path, frame)
        cap.release()
    
    # 读取图片
    frame = cv2.imread(frame_path)
    
    # 加载两个模型
    model_orig = YOLO("yolov8m.pt")
    model_opt = YOLO("models/best.pt")
    
    # 分别推理
    print("原始模型推理中...")
    results_orig = model_orig(frame)[0]
    
    print("优化模型推理中...")
    results_opt = model_opt(frame)[0]
    
    # 获取检测结果
    num_orig = len(results_orig.boxes) if results_orig.boxes else 0
    num_opt = len(results_opt.boxes) if results_opt.boxes else 0
    
    print(f"\n📊 单帧对比结果:")
    print(f"原始模型检测到: {num_orig} 个目标")
    print(f"优化模型检测到: {num_opt} 个目标")
    
    if num_orig > 0:
        avg_conf_orig = results_orig.boxes.conf.mean().item()
        print(f"原始模型平均置信度: {avg_conf_orig:.3f}")
    
    if num_opt > 0:
        avg_conf_opt = results_opt.boxes.conf.mean().item()
        print(f"优化模型平均置信度: {avg_conf_opt:.3f}")
    
    # 生成对比图
    img_orig = results_orig.plot()
    img_opt = results_opt.plot()
    
    # 左右拼接
    cv2.putText(img_orig, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(img_opt, "Optimized", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
    
    combined = np.hstack((img_orig, img_opt))
    cv2.imwrite("frame_comparison.jpg", combined)
    
    print("\n✅ 对比图已保存: frame_comparison.jpg")
    
    return results_orig, results_opt

if __name__ == "__main__":
    quick_frame_compare()