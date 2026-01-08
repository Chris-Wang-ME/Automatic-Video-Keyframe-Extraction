import streamlit as st
import cv2
import numpy as np
import tempfile
import zipfile
import os
from io import BytesIO

# --- 页面设置 ---
st.set_page_config(page_title="Mac 视频关键帧提取器", layout="wide")
st.title("🎬 视频关键帧智能提取 (一键版)")
st.markdown("上传视频后，系统将自动识别关键画面并过滤模糊帧。")

def get_blur_score(image):
    """计算图像清晰度得分"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

# --- 文件上传 ---
uploaded_file = st.file_uploader("📂 请上传视频文件 (MP4, MOV, AVI)", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # 临时保存上传文件
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if st.button("🚀 开始提取关键帧"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        frames_to_save = []
        last_hist = None
        
        # --- 后台预设参数 (不再显示滑块) ---
        interval_secs = 1.0  # 每 1 秒扫描一次
        blur_limit = 80.0    # 基础清晰度过滤
        sensitivity = 0.95   # 场景切换灵敏度
        
        step = int(fps * interval_secs)
        if step < 1: step = 1
        
        cols = st.columns(4) # 每行显示4张预览图
        img_count = 0

        for i in range(0, total_frames, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret: break
            
            # 1. 画面变化检测
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            
            is_new_scene = True
            if last_hist is not None:
                diff = cv2.compareHist(last_hist, hist, cv2.HISTCMP_CORREL)
                if diff > sensitivity: 
                    is_new_scene = False
            
            # 2. 模糊过滤并保存
            if is_new_scene:
                if get_blur_score(frame) >= blur_limit:
                    frames_to_save.append(frame)
                    
                    # 实时预览
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    with cols[img_count % 4]:
                        st.image(frame_rgb, caption=f"时间: {i/fps:.1f}s")
                    img_count += 1
                
                last_hist = hist
            
            # 更新进度
            progress_bar.progress(min(i / total_frames, 1.0))
            status_text.text(f"已处理: {int((i/total_frames)*100)}%")

        cap.release()
        os.unlink(tfile.name) # 删除临时文件
        st.success(f"处理完成！提取了 {len(frames_to_save)} 张关键帧。")

        # --- 打包下载 ---
        if frames_to_save:
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zf:
                for idx, f in enumerate(frames_to_save):
                    is_success, buffer = cv2.imencode(".jpg", f)
                    if is_success:
                        zf.writestr(f"keyframe_{idx}.jpg", buffer.tobytes())
            
            st.download_button(
                label="📥 点击下载所有关键帧 (ZIP)",
                data=zip_buffer.getvalue(),
                file_name="keyframes.zip",
                mime="application/zip"
            )