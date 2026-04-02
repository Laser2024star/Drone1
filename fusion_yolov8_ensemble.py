"""
fusion (无人机+交通标志) + yolov8m (通用目标) 融合检测器
符合赛题要求：多场景（路况+无人机）、多目标
"""

import cv2
import numpy as np
import time
import os
from ultralytics import YOLO

class FusionYOLOEnsemble:
    def __init__(self, 
                 fusion_model_path="models/fusion_best.pt",
                 general_model_path="yolov8m.pt",
                 fusion_weight=0.6,
                 general_weight=0.4):
        
        print("🚀 初始化 fusion+yolov8 融合检测器...")
        print(f"   fusion模型: {fusion_model_path} (无人机+交通标志)")
        print(f"   通用模型: {general_model_path} (人员/车辆)")
        
        self.fusion_model = YOLO(fusion_model_path)
        self.general_model = YOLO(general_model_path)
        self.fusion_weight = fusion_weight
        self.general_weight = general_weight
        
        # 检测参数
        self.conf_threshold = 0.15
        self.iou_nms = 0.45
        self.imgsz = 416  # 速度优化
        
        # fusion模型类别（无人机+交通标志）
        self.fusion_classes = {
            0: 'drone',           # 无人机
            1: 'traffic_sign'     # 交通标志
        }
        
        # yolov8m COCO类别（路况相关）
        self.general_classes = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle',
            5: 'bus', 7: 'truck', 9: 'traffic_light', 11: 'stop_sign'
        }
        
        print("✅ 融合检测器初始化完成")
    
    def detect_frame(self, frame, verbose=False):
        """单帧检测"""
        start_time = time.time()
        
        # fusion模型检测（无人机+交通标志）
        fusion_results = self.fusion_model(
            frame, conf=self.conf_threshold, iou=self.iou_nms,
            imgsz=self.imgsz, verbose=False
        )[0]
        
        # 通用模型检测（人员+车辆）
        general_results = self.general_model(
            frame, conf=self.conf_threshold, iou=self.iou_nms,
            imgsz=self.imgsz, verbose=False
        )[0]
        
        # 融合结果
        boxes, scores, labels, names = self._merge_results(fusion_results, general_results)
        
        inference_time = (time.time() - start_time) * 1000
        
        if verbose:
            print(f"检测到 {len(boxes)} 个目标, 耗时: {inference_time:.1f}ms")
            for name, score in zip(names, scores):
                print(f"  - {name}: {score:.2f}")
        
        return boxes, scores, labels, names, inference_time
    
    def _merge_results(self, fusion_results, general_results):
        """融合两个模型的结果"""
        boxes = []
        scores = []
        labels = []
        names = []
        
        # 处理fusion模型结果
        if fusion_results.boxes is not None:
            f_boxes = fusion_results.boxes.xyxy.cpu().numpy()
            f_scores = fusion_results.boxes.conf.cpu().numpy()
            f_labels = fusion_results.boxes.cls.cpu().numpy()
            
            for box, score, label in zip(f_boxes, f_scores, f_labels):
                class_id = int(label)
                class_name = self.fusion_classes.get(class_id, f'class_{class_id}')
                boxes.append(box)
                scores.append(score * self.fusion_weight)
                labels.append(class_id)
                names.append(class_name)
        
        # 处理通用模型结果
        if general_results.boxes is not None:
            g_boxes = general_results.boxes.xyxy.cpu().numpy()
            g_scores = general_results.boxes.conf.cpu().numpy()
            g_labels = general_results.boxes.cls.cpu().numpy()
            
            for box, score, label in zip(g_boxes, g_scores, g_labels):
                class_id = int(label)
                if class_id in self.general_classes:
                    class_name = self.general_classes[class_id]
                    boxes.append(box)
                    scores.append(score * self.general_weight)
                    labels.append(class_id)
                    names.append(class_name)
        
        # NMS去重
        if boxes:
            keep = self._nms(boxes, scores, 0.5)
            boxes = [boxes[i] for i in keep]
            scores = [scores[i] for i in keep]
            labels = [labels[i] for i in keep]
            names = [names[i] for i in keep]
        
        return boxes, scores, labels, names
    
    def _nms(self, boxes, scores, iou_threshold):
        """非极大值抑制"""
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
    
    def draw_detections(self, frame, boxes, scores, names):
        """绘制检测结果"""
        annotated = frame.copy()
        
        color_map = {
            'drone': (0, 0, 255),           # 红色
            'traffic_sign': (255, 0, 255),  # 紫色
            'traffic_light': (255, 0, 255), # 紫色
            'stop_sign': (255, 0, 255),     # 紫色
            'person': (0, 255, 0),          # 绿色
            'car': (255, 0, 0),             # 蓝色
            'motorcycle': (255, 255, 0),    # 青色
            'bus': (0, 255, 255),           # 黄色
            'truck': (255, 165, 0),         # 橙色
            'bicycle': (255, 255, 0)        # 青色
        }
        default_color = (128, 128, 128)
        
        for box, score, name in zip(boxes, scores, names):
            x1, y1, x2, y2 = map(int, box)
            color = color_map.get(name, default_color)
            
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            label = f"{name}: {score:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - 20), (x1 + w, y1), color, -1)
            cv2.putText(annotated, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return annotated
    
    def process_video(self, input_path, output_path=None):
        """处理视频"""
        cap = cv2.VideoCapture(input_path)
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        inference_times = []
        
        print(f"开始处理: {input_path}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            boxes, scores, labels, names, infer_time = self.detect_frame(frame)
            inference_times.append(infer_time)
            
            annotated = self.draw_detections(frame, boxes, scores, names)
            
            # 显示FPS
            current_fps = 1000 / infer_time
            cv2.putText(annotated, f"FPS: {current_fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if output_path:
                out.write(annotated)
            
            frame_count += 1
            if frame_count % 50 == 0:
                avg_time = np.mean(inference_times[-50:])
                print(f"进度: {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%), FPS: {1000/avg_time:.1f}")
        
        cap.release()
        if output_path:
            out.release()
        
        avg_time = np.mean(inference_times)
        print(f"\n✅ 完成！平均FPS: {1000/avg_time:.1f}")
        
        return inference_times


def main():
    detector = FusionYOLOEnsemble()
    
    input_video = "data/input/sample4.mp4"
    output_video = "data/output/videos/fusion_yolo_result.mp4"
    
    if os.path.exists(input_video):
        detector.process_video(input_video, output_video)
        print(f"✅ 结果: {output_video}")
    else:
        print(f"⚠️ 视频不存在: {input_video}")

if __name__ == "__main__":
    main()