import cv2
import time
import sys
from ultralytics import YOLO
import os

class EdgeDetector:
    def __init__(self, model_path, use_trt=False):
        """
        边缘设备部署
        use_trt: 是否使用 TensorRT 加速
        """
        self.use_trt = use_trt
        
        if use_trt:
            # 尝试加载 TensorRT 引擎
            engine_path = model_path.replace('.pt', '.engine')
            if os.path.exists(engine_path):
                self.model = YOLO(engine_path)
                print("✅ 使用 TensorRT 引擎")
            else:
                print("⚠️ TensorRT 引擎不存在，使用 PyTorch 模型")
                self.model = YOLO(model_path)
        else:
            self.model = YOLO(model_path)
        
        # 检测参数（针对边缘设备优化）
        self.conf = 0.25
        self.iou = 0.45
        self.imgsz = 640  # 降低分辨率提升速度
        
    def benchmark(self, test_video, iterations=100):
        """
        性能基准测试
        """
        print(f"开始性能测试，迭代次数: {iterations}")
        
        cap = cv2.VideoCapture(test_video)
        inference_times = []
        
        for i in range(iterations):
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            
            start = time.time()
            results = self.model(frame, conf=self.conf, iou=self.iou, imgsz=self.imgsz, verbose=False)
            inference_time = (time.time() - start) * 1000
            inference_times.append(inference_time)
            
            if (i + 1) % 20 == 0:
                print(f"已完成 {i+1}/{iterations} 次")
        
        cap.release()
        
        # 统计结果
        avg_time = np.mean(inference_times)
        std_time = np.std(inference_times)
        
        print(f"\n📊 边缘设备性能报告:")
        print(f"   平均推理时间: {avg_time:.2f} ms")
        print(f"   标准差: {std_time:.2f} ms")
        print(f"   理论最大FPS: {1000/avg_time:.1f}")
        print(f"   最低FPS: {1000/max(inference_times):.1f}")
        print(f"   最高FPS: {1000/min(inference_times):.1f}")
        
        return inference_times

def export_tensorrt(model_path):
    """
    导出 TensorRT 引擎
    """
    print("正在导出 TensorRT 引擎...")
    model = YOLO(model_path)
    
    # 导出 FP16 引擎
    model.export(format="engine", imgsz=640, half=True, device=0)
    print("✅ FP16 TensorRT 引擎导出完成")
    
    # 导出 INT8 引擎（需要校准数据）
    # model.export(format="engine", imgsz=640, int8=True, device=0)

if __name__ == "__main__":
    import numpy as np
    
    # 检查是否使用 TensorRT
    use_trt = "--trt" in sys.argv
    
    # 选择模型（融合或单个）
    model_path = "models/best.pt"  # 或融合模型
    
    if use_trt:
        # 先导出 TensorRT
        export_tensorrt(model_path)
        model_path = model_path.replace('.pt', '.engine')
    
    # 测试边缘部署性能
    detector = EdgeDetector(model_path, use_trt=use_trt)
    detector.benchmark("data/input/sample4.mp4", iterations=100)