import cv2
import numpy as np
import os

def create_comparison_video(original_video, optimized_video, output_file="comparison_result.mp4"):
    """
    左右分屏显示两个视频的对比
    """
    print(f"\n正在生成对比视频...")
    
    # 打开两个视频
    cap1 = cv2.VideoCapture(original_video)
    cap2 = cv2.VideoCapture(optimized_video)
    
    # 检查视频是否成功打开
    if not cap1.isOpened():
        print(f"❌ 无法打开视频: {original_video}")
        return False
    if not cap2.isOpened():
        print(f"❌ 无法打开视频: {optimized_video}")
        return False
    
    # 获取视频属性
    fps = int(cap1.get(cv2.CAP_PROP_FPS))
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = min(int(cap1.get(cv2.CAP_PROP_FRAME_COUNT)), 
                       int(cap2.get(cv2.CAP_PROP_FRAME_COUNT)))
    
    print(f"原始视频: {width}x{height}, {fps}FPS, {total_frames}帧")
    
    # 创建输出视频（宽度×2）
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width*2, height))
    
    frame_count = 0
    
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 or not ret2:
            break
        
        # 在画面上添加标签
        # 左侧：原始模型
        cv2.putText(frame1, "Original Model", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame1, f"Frame: {frame_count}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        # 右侧：优化模型
        cv2.putText(frame2, "Optimized Model", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame2, f"Frame: {frame_count}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        # 添加中间分割线
        combined = np.hstack((frame1, frame2))
        cv2.line(combined, (width, 0), (width, height), (255, 255, 255), 2)
        
        # 写入输出视频
        out.write(combined)
        
        frame_count += 1
        if frame_count % 50 == 0:
            print(f"已处理 {frame_count}/{total_frames} 帧...")
    
    # 释放资源
    cap1.release()
    cap2.release()
    out.release()
    
    print(f"\n✅ 对比视频已生成: {output_file}")
    print(f"文件大小: {os.path.getsize(output_file) / 1024 / 1024:.1f} MB")
    
    return True

def extract_key_frames(video_path, output_folder="key_frames", num_frames=5):
    """
    从视频中提取关键帧，用于详细对比
    """
    os.makedirs(output_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 均匀选取几帧
    frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
    
    extracted = []
    
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            filename = f"{output_folder}/frame_{frame_idx:06d}.jpg"
            cv2.imwrite(filename, frame)
            extracted.append(filename)
    
    cap.release()
    return extracted

if __name__ == "__main__":
    # 定义视频路径
    original = "data/output/videos/final_result.mp4"  # 原来的结果
    optimized = "data/output/videos/test_optimized.mp4"  # 刚跑的结果
    
    # 1. 生成左右对比视频
    create_comparison_video(original, optimized, "comparison_full.mp4")
    
    # 2. 提取关键帧做详细对比
    print("\n正在提取关键帧...")
    frames1 = extract_key_frames(original, "frames_original", 10)
    frames2 = extract_key_frames(optimized, "frames_optimized", 10)
    
    print(f"✅ 关键帧已保存到 frames_original/ 和 frames_optimized/ 文件夹")