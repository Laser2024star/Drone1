from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import cv2
import os
import uuid
import json
from ultralytics import YOLO
import time

# 初始化
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# 加载模型 (全局加载一次)
print("✅ 正在加载 YOLO 模型...")
model = YOLO# 相对路径（更通用）
model = YOLO("models/best.pt")
print("✅ 模型加载完成")

# 配置上传文件夹
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

# --- 新增：视频上传与处理接口 ---
@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': '没有发现视频文件'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': '未选择文件'}), 400

    # 生成唯一文件名防止冲突
    unique_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1]
    input_filename = f"{unique_id}{ext}"
    output_filename = f"{unique_id}_result.mp4"
    
    input_path = os.path.join(UPLOAD_FOLDER, input_filename)
    output_path = os.path.join(PROCESSED_FOLDER, output_filename)

    try:
        # 1. 保存上传的视频
        file.save(input_path)
        print(f"📂 视频已保存: {input_path}")

        # 2. 开始处理视频
        print("🎬 开始处理视频...")
        cap = cv2.VideoCapture(input_path)
        
        # 获取视频属性
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 定义编码器 (注意：不同系统支持的编码器不同，mp4v 兼容性较好)
        fourcc = cv2.VideoWriter_fourcc(*'avc1') 
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # YOLO 推理
            results = model(frame, stream=False, verbose=False) # stream=False 适合文件处理
            
            # 绘制结果
            for result in results:
                # plot() 方法可以直接在图像上画出框和标签，并返回图像
                # 或者手动遍历 boxes 绘制，这里使用 plot() 最简单
                plotted_frame = result.plot()
                out.write(plotted_frame)
            
            frame_count += 1
            # 可选：打印进度
            if frame_count % 30 == 0:
                print(f"⏳ 处理进度: {frame_count}/{total_frames}")

        cap.release()
        out.release()
        print(f"✅ 视频处理完成: {output_path}")

        # 3. 返回结果链接
        # 前端可以通过 /download/<filename> 访问处理后的视频
        return jsonify({
            'success': True,
            'video_url': f'/download/{output_filename}',
            'message': '处理成功'
        })

    except Exception as e:
        print(f"❌ 处理出错: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        # 可选：清理上传的原始文件，保留结果文件一段时间
        if os.path.exists(input_path):
            os.remove(input_path)

@app.route('/download/<filename>')
def download_video(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

# 保持原有的 WebSocket 逻辑用于摄像头（如果需要同时支持）
@socketio.on('connect')
def connect():
    print('📡 客户端已连接 (WebSocket)')

@socketio.on('message')
def handle_message(data):
    # 这里处理摄像头的实时数据逻辑，保持不变
    pass

if __name__ == '__main__':
    # 启动服务
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)