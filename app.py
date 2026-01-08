import streamlit as st
import cv2
import numpy as np
import tempfile
import zipfile
import os
from io import BytesIO

# 页面基础配置
st.set_page_config(page_title="镜头切换自动截帧工具", layout="wide")
st.title("🎬 视频镜头自动识别与截帧")
st.markdown("上传视频后，系统会自动分析画面，**每当镜头切换时**提取一张清晰的关键帧。")

def get_blur_score(image):
    """计算清晰度得分，过滤模糊帧"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

# --- 文件上传 ---
uploaded_file = st.file_uploader("📂 选择视频文件 (MP4, MOV, AVI)", type=["mp4", "mov", "avi"])

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if st.button("🚀 开始自动分析镜头"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        frames_to_save = []
        last_hist = None
        
        # 预设镜头检测阈值
        SENSITIVITY = 0.85  # 相似度低于 0.85 判定为新镜头
        MIN_BLUR = 70.0     # 清晰度过滤
        
        # 为了网页端性能，每 3 帧扫描一次（不影响镜头切换捕捉）
        step = 3 
        
        cols = st.columns(4)
        img_count = 0

        for i in range(0, total_frames, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret: break
            
            # 计算直方图特征
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            
            is_shot_change = False
            if last_hist is None:
                is_shot_change = True
            else:
                # 比较当前帧与上一镜头的相似度
                correlation = cv2.compareHist(last_hist, hist, cv2.HISTCMP_CORREL)
                if correlation < SENSITIVITY:
                    is_shot_change = True
            
            if is_shot_change:
                # 只有画面清晰才保存
                if get_blur_score(frame) > MIN_BLUR:
                    frames_to_save.append(frame)
                    # 实时显示预览
                    with cols[img_count % 4]:
                        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=f"镜头 {img_count+1}")
                    img_count += 1
                    # 更新参考帧，用于检测下一个镜头
                    last_hist = hist
            
            progress_bar.progress(min(i / total_frames, 1.0))
            status_text.text(f"分析进度: {int((i/total_frames)*100)}%")

        cap.release()
        os.unlink(tfile.name)
        st.success(f"处理完成！共识别到 {len(frames_to_save)} 个镜头。")

        # --- 打包下载 ---
        if frames_to_save:
            zip_buf = BytesIO()
            with zipfile.ZipFile(zip_buf, "a", zipfile.ZIP_DEFLATED) as zf:
                for idx, f in enumerate(frames_to_save):
                    _, buf = cv2.imencode(".jpg", f)
                    zf.writestr(f"shot_{idx+1}.jpg", buf.tobytes())
            
            st.download_button("📥 下载镜头截图 (ZIP)", zip_buf.getvalue(), "shots.zip", "application/zip")