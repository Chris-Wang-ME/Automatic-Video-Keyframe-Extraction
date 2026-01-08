import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import zipfile
from io import BytesIO

# 页面配置
st.set_page_config(page_title="Mac 视频关键帧提取专家", layout="wide")

def get_blur_score(image):
    """计算清晰度得分（拉普拉斯方差）"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

st.title("🎬 视频关键帧智能提取工具")
st.info("适配 Mac 环境：支持模糊过滤、自定义提取频率及 ZIP 批量下载")

# --- 侧边栏：参数自定义 ---
st.sidebar.header("⚙️ 提取参数设置")

# 1. 提取频率：每隔多少秒提取一次
interval = st.sidebar.slider("提取间隔 (秒)", 0.1, 10.0, 1.0, step=0.1, help="每隔多少秒扫描一次视频帧")

# 2. 清晰度过滤：低于该值将被舍弃
blur_threshold = st.sidebar.slider("清晰度阈值", 0, 500, 100, help="数值越大，过滤掉的模糊图片越多。建议范围: 80-150")

# 3. 画面变化灵敏度
sensitivity = st.sidebar.slider("画面变化灵敏度", 0.0, 1.0, 0.95, step=0.01, help="值越低，对画面变化的捕捉越敏锐")

# --- 文件上传 ---
uploaded_file = st.file_uploader("📂 请上传视频文件 (MP4, MOV, AVI)", type=["mp4", "mov", "avi"])

if uploaded_file:
    # 暂存上传的视频
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    st.text(f"视频时长: {duration:.2f} 秒 | 帧率: {fps:.2f}")

    if st.button("🚀 开始智能提取"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        extracted_images = []
        last_hist = None
        
        # 计算帧跳跃步长 (基于用户设置的间隔秒数)
        frame_step = int(fps * interval)
        if frame_step < 1: frame_step = 1

        curr_frame_idx = 0
        grid = st.columns(4) # 每行显示4张图
        
        while curr_frame_idx < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, curr_frame_idx)
            ret, frame = cap.read()
            if not ret: break
            
            # --- 步骤 A: 画面变化检测 (防止重复画面) ---
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            
            should_save = False
            if last_hist is None:
                should_save = True
            else:
                diff = cv2.compareHist(last_hist, hist, cv2.HISTCMP_CORREL)
                if diff < sensitivity: # 画面发生了显著变化
                    should_save = True
            
            # --- 步骤 B: 模糊过滤 ---
            if should_save:
                blur_score = get_blur_score(frame)
                if blur_score >= blur_threshold:
                    # 存储图片 (RGB格式用于显示)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    extracted_images.append(frame) # 存原图BGR用于下载
                    
                    with grid[len(extracted_images) % 4]:
                        st.image(frame_rgb, caption=f"时间: {curr_frame_idx/fps:.1f}s | 得分: {int(blur_score)}")
                
                last_hist = hist

            # 更新进度
            curr_frame_idx += frame_step
            progress_bar.progress(min(curr_frame_idx / total_frames, 1.0))
            status_text.text(f"正在处理: {int((curr_frame_idx/total_frames)*100)}%")

        cap.release()
        st.success(f"提取完成！共获得 {len(extracted_images)} 张高清关键帧。")

        # --- 下载部分 ---
        if extracted_images:
            buf = BytesIO()
            with zipfile.ZipFile(buf, "a", zipfile.ZIP_DEFLATED) as z:
                for idx, img in enumerate(extracted_images):
                    _, img_encoded = cv2.imencode(".jpg", img)
                    z.writestr(f"frame_{idx}.jpg", img_encoded.tobytes())
            
            st.download_button(
                label="📥 点击下载所有关键帧 (ZIP)",
                data=buf.getvalue(),
                file_name="keyframes_output.zip",
                mime="application/zip"
            )

    # 清理临时文件
    os.unlink(tfile.name)