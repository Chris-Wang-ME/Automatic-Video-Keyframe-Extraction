import os
import cv2
from flask import Flask, render_template, request, send_from_directory

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'frames'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def extract_keyframes(video_path, output_dir, threshold=0.3):
    cap = cv2.VideoCapture(video_path)
    prev_frame = None
    count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 将帧转为灰度并缩小，计算差异
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if prev_frame is not None:
            # 计算当前帧与上一帧的差异
            frame_diff = cv2.absdiff(prev_frame, gray)
            diff_score = frame_diff.mean() / 255
            
            # 如果差异大于阈值，判定为关键帧（场景切换）
            if diff_score > threshold:
                frame_path = os.path.join(output_dir, f"keyframe_{saved_count}.jpg")
                cv2.imwrite(frame_path, frame)
                saved_count += 1
        else:
            # 第一帧强制保存
            cv2.imwrite(os.path.join(output_dir, "keyframe_0.jpg"), frame)
            saved_count += 1

        prev_frame = gray
        count += 1
    
    cap.release()
    return saved_count

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['video']
        if file:
            video_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(video_path)
            
            # 清空之前的截图
            for f in os.listdir(OUTPUT_FOLDER):
                os.remove(os.path.join(OUTPUT_FOLDER, f))
            
            extract_keyframes(video_path, OUTPUT_FOLDER)
            frames = os.listdir(OUTPUT_FOLDER)
            return render_template('index.html', frames=frames)
    
    return render_template('index.html', frames=[])

@app.route('/frames/<filename>')
def display_frame(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == '__main__':
    app.run(port=5001, debug=True)