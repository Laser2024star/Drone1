import cv2
import numpy as np
from ultralytics import YOLO
import time

class EnsembleDetector:
    def __init__(self, model1_path, model2_path, weights=[0.5, 0.5]):
        """
        模型融合检测器
        model1: UAVDB 无人机模型
        model2: 交通标志通用模型
        weights: 融合权重 [w1, w2]
        """
        print("加载模型...")
        self.model1 = YOLO(model1_path)
        self.model2 = YOLO(model2_path)
        self.weights = weights
        
        # 无人机检测专用置信度（从F1曲线得到）
        self.drone_conf = 0.7
        
        # 通用检测置信度
        self.general_conf = 0.25
        
        print("✅ 模型加载完成")
    
    def detect(self, frame, use_ensemble=True):
        """
        执行检测，支持融合模式
        """
        results = []
        
        # 1. 无人机专用检测（高精度）
        result1 = self.model1(frame, conf=self.drone_conf, verbose=False)[0]
        
        # 2. 通用检测
        result2 = self.model2(frame, conf=self.general_conf, verbose=False)[0]
        
        if not use_ensemble:
            return result2  # 只用通用模型
        
        # 3. 融合处理
        return self._merge_results(result1, result2)
    
    def _merge_results(self, result1, result2):
        """
        融合两个模型的检测结果
        """
        merged_boxes = []
        merged_scores = []
        merged_labels = []
        
        # 处理无人机模型结果（类别ID 0 表示无人机）
        if result1.boxes is not None:
            boxes1 = result1.boxes.xyxy.cpu().numpy()
            scores1 = result1.boxes.conf.cpu().numpy()
            labels1 = result1.boxes.cls.cpu().numpy()
            
            for i, (box, score, label) in enumerate(zip(boxes1, scores1, labels1)):
                merged_boxes.append(box)
                merged_scores.append(score * self.weights[0])
                merged_labels.append(int(label))  # 无人机类别
        
        # 处理通用模型结果
        if result2.boxes is not None:
            boxes2 = result2.boxes.xyxy.cpu().numpy()
            scores2 = result2.boxes.conf.cpu().numpy()
            labels2 = result2.boxes.cls.cpu().numpy()
            
            for i, (box, score, label) in enumerate(zip(boxes2, scores2, labels2)):
                merged_boxes.append(box)
                merged_scores.append(score * self.weights[1])
                merged_labels.append(int(label))
        
        # 非极大值抑制（NMS）去重
        if merged_boxes:
            keep = self._nms(merged_boxes, merged_scores, iou_threshold=0.5)
            merged_boxes = [merged_boxes[i] for i in keep]
            merged_scores = [merged_scores[i] for i in keep]
            merged_labels = [merged_labels[i] for i in keep]
        
        return merged_boxes, merged_scores, merged_labels
    
    def _nms(self, boxes, scores, iou_threshold=0.5):
        """简单NMS实现"""
        if not boxes:
            return []
        
        boxes = np.array(boxes)
        scores = np.array(scores)
        
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        
        order = scores.argsort()[::-1]
        keep = []
        
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(ovr <= iou_threshold)[0]
            order = order[inds + 1]
        
        return keep
    
    def draw_detections(self, frame, boxes, scores, labels):
        """绘制检测结果，区分无人机和通用目标"""
        annotated = frame.copy()
        
        # 无人机类别ID（根据训练数据确定）
        DRONE_CLASS_ID = 0
        
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = map(int, box)
            
            # 区分颜色：无人机用红色，其他用绿色
            if label == DRONE_CLASS_ID:
                color = (0, 0, 255)  # 红色
                label_text = f"DRONE {score:.2f}"
            else:
                color = (0, 255, 0)  # 绿色
                label_text = f"Obj {label} {score:.2f}"
            
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, label_text, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return annotated

def test_ensemble():
    """测试融合检测"""
    # 模型路径
    model1_path = "models/best_old.pt"  # UAVDB 无人机模型
    model2_path = "models/best.pt"  # 交通标志模型
    
    # 初始化融合检测器
    detector = EnsembleDetector(model1_path, model2_path, weights=[0.6, 0.4])
    
    # 测试图片
    cap = cv2.VideoCapture("data/input/sample4.mp4")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("ensemble_result.mp4", fourcc, fps, (width, height))
    
    frame_count = 0
    inference_times = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        start = time.time()
        
        # 融合检测
        boxes, scores, labels = detector.detect(frame, use_ensemble=True)
        
        inference_time = (time.time() - start) * 1000
        inference_times.append(inference_time)
        
        # 绘制结果
        annotated = detector.draw_detections(frame, boxes, scores, labels)
        
        # 显示FPS
        cv2.putText(annotated, f"FPS: {1000/inference_time:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        out.write(annotated)
        
        frame_count += 1
        if frame_count % 50 == 0:
            print(f"处理进度: {frame_count} 帧, 平均延迟: {np.mean(inference_times):.1f}ms")
    
    cap.release()
    out.release()
    
    avg_time = np.mean(inference_times)
    print(f"\n📊 融合检测性能:")
    print(f"   平均推理时间: {avg_time:.1f} ms")
    print(f"   理论最大FPS: {1000/avg_time:.1f}")
    print(f"   处理总帧数: {frame_count}")
    print(f"✅ 结果保存到: ensemble_result.mp4")

if __name__ == "__main__":
    import numpy as np
    test_ensemble()