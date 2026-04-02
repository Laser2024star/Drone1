"""
best_old (94.8% 无人机专用) + yolov8m (通用目标) 融合检测器
修复版：正确区分无人机和人员
"""

import cv2
import numpy as np
import time
import os
from ultralytics import YOLO

class DroneGeneralEnsemble:
    def __init__(self, 
                 drone_model_path="models/best_old.pt",
                 general_model_path="yolov8m.pt",
                 drone_weight=0.6,
                 general_weight=0.4,
                 iou_threshold=0.5):
        
        print("🚀 初始化融合检测器...")
        print(f"   无人机模型: {drone_model_path}")
        print(f"   通用模型: {general_model_path}")
        
        self.drone_model = YOLO(drone_model_path)
        self.general_model = YOLO(general_model_path)
        self.drone_weight = drone_weight
        self.general_weight = general_weight
        self.iou_threshold = iou_threshold
        
        self.conf_threshold = 0.25
        self.iou_nms = 0.45
        self.imgsz = 640
        
        # yolov8m COCO类别映射（正确映射）
        self.class_names = {
            0: 'person',
            1: 'bicycle',
            2: 'car',
            3: 'motorcycle',
            4: 'airplane',
            5: 'bus',
            6: 'train',
            7: 'truck',
            8: 'boat',
            9: 'traffic light',
            10: 'fire hydrant',
            11: 'stop sign',
            12: 'parking meter',
            13: 'bench',
            14: 'bird',
            15: 'cat',
            16: 'dog',
            17: 'horse',
            18: 'sheep',
            19: 'cow',
            20: 'elephant',
            21: 'bear',
            22: 'zebra',
            23: 'giraffe',
            24: 'backpack',
            25: 'umbrella',
            26: 'handbag',
            27: 'tie',
            28: 'suitcase',
            29: 'frisbee',
            30: 'skis',
            31: 'snowboard',
            32: 'sports ball',
            33: 'kite',
            34: 'baseball bat',
            35: 'baseball glove',
            36: 'skateboard',
            37: 'surfboard',
            38: 'tennis racket',
            39: 'bottle',
            40: 'wine glass',
            41: 'cup',
            42: 'fork',
            43: 'knife',
            44: 'spoon',
            45: 'bowl',
            46: 'banana',
            47: 'apple',
            48: 'sandwich',
            49: 'orange',
            50: 'broccoli',
            51: 'carrot',
            52: 'hot dog',
            53: 'pizza',
            54: 'donut',
            55: 'cake',
            56: 'chair',
            57: 'couch',
            58: 'potted plant',
            59: 'bed',
            60: 'dining table',
            61: 'toilet',
            62: 'tv',
            63: 'laptop',
            64: 'mouse',
            65: 'remote',
            66: 'keyboard',
            67: 'cell phone',
            68: 'microwave',
            69: 'oven',
            70: 'toaster',
            71: 'sink',
            72: 'refrigerator',
            73: 'book',
            74: 'clock',
            75: 'vase',
            76: 'scissors',
            77: 'teddy bear',
            78: 'hair drier',
            79: 'toothbrush'
        }
        
        print("✅ 融合检测器初始化完成")
    
    def detect_frame(self, frame, verbose=False):
        start_time = time.time()
        
        # 无人机专用检测
        drone_results = self.drone_model(
            frame, conf=self.conf_threshold, iou=self.iou_nms, 
            imgsz=self.imgsz, verbose=False
        )[0]
        
        # 通用目标检测
        general_results = self.general_model(
            frame, conf=self.conf_threshold, iou=self.iou_nms,
            imgsz=self.imgsz, verbose=False
        )[0]
        
        # 融合结果
        merged_boxes, merged_scores, merged_labels, merged_names = self._merge_results(
            drone_results, general_results
        )
        
        inference_time = (time.time() - start_time) * 1000
        
        if verbose:
            print(f"检测到 {len(merged_boxes)} 个目标, 耗时: {inference_time:.1f}ms")
            for name, score in zip(merged_names, merged_scores):
                print(f"  - {name}: {score:.2f}")
        
        return merged_boxes, merged_scores, merged_labels, merged_names, inference_time
    
    def _merge_results(self, drone_results, general_results):
        """融合两个模型的检测结果 - 修复类别冲突"""
        merged_boxes = []
        merged_scores = []
        merged_labels = []
        merged_names = []
        
        # 处理无人机模型结果（只有无人机）
        if drone_results.boxes is not None:
            boxes = drone_results.boxes.xyxy.cpu().numpy()
            scores = drone_results.boxes.conf.cpu().numpy()
            
            for box, score in zip(boxes, scores):
                merged_boxes.append(box)
                merged_scores.append(score * self.drone_weight)
                merged_labels.append(999)  # 特殊ID标记无人机
                merged_names.append('drone')
        
        # 处理通用模型结果（正确映射类别名称）
        if general_results.boxes is not None:
            boxes = general_results.boxes.xyxy.cpu().numpy()
            scores = general_results.boxes.conf.cpu().numpy()
            labels = general_results.boxes.cls.cpu().numpy()
            
            for box, score, label in zip(boxes, scores, labels):
                class_id = int(label)
                class_name = self.class_names.get(class_id, f'class_{class_id}')
                
                merged_boxes.append(box)
                merged_scores.append(score * self.general_weight)
                merged_labels.append(class_id)
                merged_names.append(class_name)
        
        # NMS去重
        if merged_boxes:
            keep = self._nms(merged_boxes, merged_scores, self.iou_threshold)
            merged_boxes = [merged_boxes[i] for i in keep]
            merged_scores = [merged_scores[i] for i in keep]
            merged_labels = [merged_labels[i] for i in keep]
            merged_names = [merged_names[i] for i in keep]
        
        return merged_boxes, merged_scores, merged_labels, merged_names
    
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
            'drone': (0, 0, 255),      # 红色
            'person': (0, 255, 0),     # 绿色
            'car': (255, 0, 0),        # 蓝色
            'motorcycle': (255, 255, 0), # 青色
            'bicycle': (255, 255, 0),
            'bus': (0, 255, 255),      # 黄色
            'truck': (255, 165, 0)     # 橙色
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
    
    def process_video(self, input_path, output_path=None, show_fps=True):
        """处理视频文件"""
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
        
        print(f"开始处理视频: {input_path}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            boxes, scores, labels, names, infer_time = self.detect_frame(frame)
            inference_times.append(infer_time)
            
            annotated = self.draw_detections(frame, boxes, scores, names)
            
            if show_fps:
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
        print(f"\n✅ 处理完成！平均FPS: {1000/avg_time:.1f}")
        
        return inference_times


def main():
    detector = DroneGeneralEnsemble()
    
    input_video = "data/input/sample4.mp4"
    output_video = "data/output/videos/ensemble_fixed.mp4"
    
    if os.path.exists(input_video):
        detector.process_video(input_video, output_video)
        print(f"✅ 结果保存到: {output_video}")
    else:
        print(f"⚠️ 视频不存在: {input_video}")

if __name__ == "__main__":
    main()