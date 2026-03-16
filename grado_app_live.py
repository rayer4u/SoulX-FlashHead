import gradio as gr
import os
import torch
import numpy as np
import threading
import time
import librosa
import queue
import subprocess
from datetime import datetime
from collections import deque
from loguru import logger

# Import internal modules
from flash_head.inference import get_pipeline, get_base_data, get_infer_params, get_audio_embedding, run_pipeline

# Global variable to store the loaded pipeline
pipeline = None
loaded_ckpt_dir = None
loaded_wav2vec_dir = None
loaded_model_type = None
live_audio_queue = queue.Queue()
global_sample_rate = 16000
global_slice_len_samples = None 

class LiveFFmpegPipeline:
    """工业级 FFmpeg 管道推流器：常驻内存，吃裸数据，吐标准直播流"""
    def __init__(self, output_dir, width=512, height=512, fps=25, sample_rate=16000):
        self.output_dir = output_dir
        self.width = width
        self.height = height
        self.fps = fps
        self.sample_rate = sample_rate
        
        os.makedirs(output_dir, exist_ok=True)
        self.playlist_path = os.path.join(output_dir, "playlist.m3u8")
        
        self.audio_fifo = os.path.join(output_dir, "audio_pipe.fifo")
        if os.path.exists(self.audio_fifo):
            os.remove(self.audio_fifo)
        os.mkfifo(self.audio_fifo)
        
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            # -- 视频输入流 (来自 stdin, 接收 Raw RGB24 字节流) --
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}',
            '-pix_fmt', 'rgb24',  
            '-r', str(fps),
            '-thread_queue_size', '512', # <--- 加在这里！扩容视频接收队列
            '-probesize', '32',        # <--- 强制关闭视频探测
            '-analyzeduration', '0',   # <--- 强制关闭视频分析
            '-i', 'pipe:0',
            
            # -- 音频输入流 (来自命名管道, 接收 Raw Float32 字节流) --
            '-f', 'f32le',        
            '-ar', str(sample_rate),
            '-ac', '1',
            '-probesize', '32',        # <--- 强制关闭音频探测（救命参数！）
            '-analyzeduration', '0',   # <--- 强制关闭音频分析（救命参数！）
            '-i', self.audio_fifo,
            
            # -- 输出编码设置 --
            '-c:v', 'libx264',
            '-profile:v', 'baseline',
            '-preset', 'ultrafast',   # 直播级极速编码
            '-g', '48',               # <--- 【新增】强制每 48 帧生成一个关键帧！
            '-keyint_min', '48',      # <--- 【新增】最小关键帧间隔也是 48
            '-sc_threshold', '0',     # <--- 【新增】禁止 FFmpeg 瞎猜场景切换
            '-pix_fmt', 'yuv420p',
            '-c:a', 'aac',
            '-ar', '48000',           # 强制升频到 48kHz，完美迎合浏览器
            '-b:a', '128k',
            
            # -- HLS 直播滑动窗口配置 --
            '-f', 'hls',
            '-hls_time', '1.92',      # 切片长度 (48帧完美对齐)
            '-hls_list_size', '20',    # 滑动窗口只保留最新的 20 个切片
            '-hls_flags', 'delete_segments', # 自动删除旧切片，防止硬盘爆炸
            '-hls_segment_filename', os.path.join(output_dir, 'chunk_%04d.ts'),
            self.playlist_path
        ]
        
        # 3. 启动 FFmpeg 守护进程
        logger.info("Starting FFmpeg Live Pipeline...")
        self.process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
        
        # 4. 打开音频管道准备写入 (必须在 Popen 之后)
        fd = os.open(self.audio_fifo, os.O_RDWR)
        self.audio_fd = open(fd, 'wb')
        logger.info("Pipeline is ready and listening for data.")

    def push_data(self, video_frames_np, audio_data_np):
        try:
            # 先把矩阵转成字节流
            video_bytes = video_frames_np.tobytes()
            audio_bytes = audio_data_np.astype(np.float32).tobytes()

            # 定义喂视频的任务
            def write_video():
                try:
                    self.process.stdin.write(video_bytes)
                    self.process.stdin.flush()
                except Exception as e:
                    logger.error(f"Video write blocked/failed: {e}")

            # 定义喂音频的任务
            def write_audio():
                try:
                    self.audio_fd.write(audio_bytes)
                    self.audio_fd.flush()
                except Exception as e:
                    logger.error(f"Audio write blocked/failed: {e}")

            # 开启两个线程，左手右手同时喂！
            tv = threading.Thread(target=write_video)
            ta = threading.Thread(target=write_audio)

            tv.start()
            ta.start()

            # 等待这一批次的数据全部被 FFmpeg 吃完
            tv.join()
            ta.join()

        except Exception as e:
            logger.error(f"Pipeline push_data error: {e}")

    def close(self):
        logger.info("Initiating pipeline shutdown...")
        
        # 1. 【极度关键】：必须先同时关闭视频和音频的输入！
        # 这样 FFmpeg 才会收到完整的 EOF (End of File) 信号，知道直播结束了。
        if hasattr(self, 'process') and self.process and self.process.stdin:
            try:
                self.process.stdin.close()
            except Exception:
                pass
                
        if hasattr(self, 'audio_fd') and self.audio_fd:
            try:
                self.audio_fd.close()
            except Exception:
                pass
                
        # 2. 现在 FFmpeg 知道没数据了，它会把最后一点数据切片，然后正常退出
        if hasattr(self, 'process') and self.process:
            try:
                # 等待它正常打包收尾，最多给它 5 秒钟
                self.process.wait(timeout=5)
                logger.info("FFmpeg exited gracefully.")
            except subprocess.TimeoutExpired:
                # 如果 5 秒后还没死，说明遇到了僵尸进程，直接拔电源！
                logger.warning("FFmpeg graceful exit timeout. Killing process...")
                self.process.kill()
                self.process.wait()
                
        # 3. 拆除“墙上的窗户”（删除命名管道）
        if hasattr(self, 'audio_fifo') and os.path.exists(self.audio_fifo):
            try:
                os.remove(self.audio_fifo)
            except Exception as e:
                logger.error(f"Failed to remove audio fifo: {e}")
                
        logger.info("Pipeline closed cleanly. Ready for next stream.")


def insert_dynamic_audio(new_audio_path):
    """动态插播音频处理函数"""
    global global_sample_rate, global_slice_len_samples
    
    if new_audio_path is None:
        gr.Warning("请先上传音频文件！")
        return "⚠️ 未检测到音频。"
        
    if global_slice_len_samples is None:
        gr.Warning("直播尚未开始，请先启动 Start Live Stream！")
        return "⚠️ 直播未启动，插播失败。"

    try:
        # 加载新音频
        new_audio_array, _ = librosa.load(new_audio_path, sr=global_sample_rate, mono=True)
        
        # 补齐末尾，防止切片报错
        remainder = len(new_audio_array) % global_slice_len_samples
        if remainder > 0:
            new_audio_array = np.concatenate([
                new_audio_array, 
                np.zeros(global_slice_len_samples - remainder, dtype=new_audio_array.dtype)
            ])
            
        # 切片并送入队列
        slices = new_audio_array.reshape(-1, global_slice_len_samples)
        for s in slices:
            live_audio_queue.put(s)
            
        logger.info(f"Successfully inserted {len(slices)} chunks of new audio into the stream!")
        gr.Info("插播成功！数字人即将开始播报。")
        return f"✅ 成功插入 {len(slices)} 个音频切片，正在排队播放..."
        
    except Exception as e:
        logger.error(f"Insert audio failed: {e}")
        return f"❌ 插播失败: {str(e)}"


def run_inference(
    ckpt_dir,
    wav2vec_dir,
    model_type,
    cond_image,
    audio_path,
    seed,
    use_face_crop,
    progress=gr.Progress()
):
    global pipeline, loaded_ckpt_dir, loaded_wav2vec_dir, loaded_model_type
    global global_sample_rate, global_slice_len_samples

    # 1. Load Model if needed
    if pipeline is None or loaded_ckpt_dir != ckpt_dir or loaded_wav2vec_dir != wav2vec_dir or loaded_model_type != model_type:
        progress(0, desc="Loading Model...")
        logger.info(f"Loading pipeline with ckpt_dir={ckpt_dir}, wav2vec_dir={wav2vec_dir}")
        try:
            pipeline = get_pipeline(world_size=1, ckpt_dir=ckpt_dir, model_type=model_type, wav2vec_dir=wav2vec_dir)
            loaded_ckpt_dir = ckpt_dir
            loaded_wav2vec_dir = wav2vec_dir
            loaded_model_type = model_type
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise gr.Error(f"Failed to load model: {e}")

    # 2. Prepare Data
    progress(0.1, desc="Preparing Data...")
    
    base_seed = int(seed) if seed >= 0 else 9999

    try:
        get_base_data(pipeline, cond_image_path_or_dir=cond_image, base_seed=base_seed, use_face_crop=use_face_crop)
    except Exception as e:
        logger.error(f"Error in get_base_data: {e}")
        raise gr.Error(f"Error processing inputs: {e}")

    infer_params = get_infer_params()

    sample_rate = infer_params['sample_rate']
    tgt_fps = infer_params['tgt_fps']
    cached_audio_duration = infer_params['cached_audio_duration']
    frame_num = infer_params['frame_num']
    motion_frames_num = infer_params['motion_frames_num']
    slice_len = frame_num - motion_frames_num

    try:
        human_speech_array_all, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    except Exception as e:
        raise gr.Error(f"Failed to load audio file: {e}")

    human_speech_array_slice_len = slice_len * sample_rate // tgt_fps
    
    global_sample_rate = sample_rate
    global_slice_len_samples = human_speech_array_slice_len
    output_dir = 'gradio_results_live'
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
    stream_session_dir = os.path.join(output_dir, f"session_{timestamp}")

    # 3. 初始化实时直播管道
    live_pipeline = LiveFFmpegPipeline(
        output_dir=stream_session_dir,
        width=512, height=512, fps=tgt_fps, sample_rate=sample_rate
    )

    playlist_yielded = False
    cached_audio_length_sum = sample_rate * cached_audio_duration
    audio_end_idx = cached_audio_duration * tgt_fps
    audio_start_idx = audio_end_idx - frame_num
    audio_dq = deque([0.0] * cached_audio_length_sum, maxlen=cached_audio_length_sum)

    # 用静音填充音频末尾，防止最后的分块被截断
    remainder = len(human_speech_array_all) % human_speech_array_slice_len
    if remainder > 0:
        human_speech_array_all = np.concatenate([
            human_speech_array_all, 
            np.zeros(human_speech_array_slice_len - remainder, dtype=human_speech_array_all.dtype)
        ])
    human_speech_array_slices = human_speech_array_all.reshape(-1, human_speech_array_slice_len)

    # 1. 启动前清空残留队列
    while not live_audio_queue.empty():
        try: live_audio_queue.get_nowait()
        except queue.Empty: break

    # 2. 将初始音频放入队列
    for chunk in human_speech_array_slices:
        live_audio_queue.put(chunk)

    # 3. 准备一段标准静音数据用于填充无音频的时刻
    silent_speech_array = np.zeros(human_speech_array_slice_len, dtype=np.float32)

    # --- 【新增】时间轴对齐参数 ---
    stream_start_time = time.time()
    chunk_duration = human_speech_array_slice_len / sample_rate
    MAX_AHEAD_SECONDS = 2.5  # 限制 GPU 最多只能超前生成 2.5 秒的缓冲，保证插播极速响应

    try:
        chunk_idx = 0
        while True:
            # 尝试从队列拿音频，拿不到就用静音
            try:
                current_speech_array = live_audio_queue.get_nowait()
                is_silent_mode = False
            except queue.Empty:
                current_speech_array = silent_speech_array
                is_silent_mode = True

            # --- AI 推理 ---
            audio_dq.extend(current_speech_array.tolist())
            audio_array = np.array(audio_dq)
            
            audio_embedding = get_audio_embedding(pipeline, audio_array, audio_start_idx, audio_end_idx)
            video = run_pipeline(pipeline, audio_embedding)
            video = video[motion_frames_num:] # 截取有效动作帧 (通常是 24 帧)

            # --- 转换并推流 ---
            video_np = video.cpu().numpy().astype(np.uint8)
            live_pipeline.push_data(video_np, current_speech_array)

            if is_silent_mode:
                logger.info(f"Idle Streaming... chunk_idx: {chunk_idx}")
            else:
                logger.info(f"Speech Streaming... chunk_idx: {chunk_idx} (Queue size: {live_audio_queue.qsize()})")

            # --- 唤醒前端 ---
            if not playlist_yielded and os.path.exists(live_pipeline.playlist_path):
                logger.info(f"Live Playlist generated: {live_pipeline.playlist_path}")

                html_content = f"""<!DOCTYPE html>
                <html>
                <head>
                    <script src="https://cdn.jsdelivr.net/npm/hls.js@latest"></script>
                    <style>
                        body {{ margin: 0; background: black; overflow: hidden; display: flex; align-items: center; justify-content: center; height: 100vh; }}
                        video {{ width: 100%; max-height: 100%; object-fit: contain; outline: none; }}
                    </style>
                </head>
                <body>
                    <video id="live_video" controls autoplay muted></video>
                    <script>
                        var video = document.getElementById('live_video');
                        // 动态拼装 7900 端口的推流地址
                        var url = 'playlist.m3u8';

                        if (Hls.isSupported()) {{
                            var hls = new Hls({{
                                maxBufferLength: 30,         // 允许在内存里最大缓冲 30 秒
                                liveSyncDurationCount: 3,
                                // liveMaxLatencyDurationCount: 6,
                                maxLiveSyncPlaybackRate: 1,
                                debug: true,
                            }});
                            hls.loadSource(url);
                            hls.attachMedia(video);
                            hls.on(Hls.Events.MANIFEST_PARSED, function() {{
                                video.play();
                            }});
                        }} else if (video.canPlayType('application/vnd.apple.mpegurl')) {{
                            video.src = url;
                            video.addEventListener('loadedmetadata', function() {{
                                video.play();
                            }});
                        }}
                    </script>
                </body>
                </html>"""

                # 2. 将 HTML 真实写入硬盘 (和 m3u8 同级目录)
                player_html_path = os.path.join(live_pipeline.output_dir, "player.html")
                with open(player_html_path, "w", encoding="utf-8") as f:
                    f.write(html_content)
                
                # 3. 计算通过 FastAPI 静态挂载路由访问该 html 的 URL
                # 例如：/stream/session_20260312-074434-830/player.html
                rel_player_path = os.path.relpath(player_html_path, "gradio_results_live")
                player_url = f"/stream/{rel_player_path}"
                
                # 4. 用 iframe 将这个纯净的 URL 嵌入 Gradio
                clean_iframe = f'<iframe src="{player_url}" style="width: 100%; height: 500px; border: none; border-radius: 8px; background: black;"></iframe>'
                
                yield {
                    html_output: clean_iframe,
                    download_output: gr.update(visible=False)
                }
                playlist_yielded = True
            else:
                # 配合 Gradio 机制进行跳过更新，以持续监听中止事件
                yield {
                    html_output: gr.skip(),
                    download_output: gr.skip()
                }

            # --- 【核心新增】防过度生成限速锁 ---
            generated_time = (chunk_idx + 1) * chunk_duration
            real_elapsed_time = time.time() - stream_start_time
            
            ahead_time = generated_time - real_elapsed_time
            # 如果 GPU 生成的视频时间比现实时间超前了 0.5 秒以上，强制挂起等待
            if ahead_time > MAX_AHEAD_SECONDS:
                time.sleep(ahead_time - MAX_AHEAD_SECONDS)

            chunk_idx += 1

    finally:
        # 当音频播放完毕，关闭流并安全释放 FFmpeg 进程
        live_pipeline.close()
        logger.info("Live streaming session finished.")


with gr.Blocks(title="SoulX-FlashHead Live Pipeline", theme=gr.themes.Soft()) as app:
    gr.Markdown("# ⚡ SoulX-FlashHead Live Video Pipeline")
    gr.Markdown("工业级流媒体架构：零存盘、零重叠、纯内存管道推流")

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### 🎬 Generation Inputs")
                with gr.Row():
                    cond_image_input = gr.Image(
                        label="Condition Image", 
                        type="filepath", 
                        value="examples/girl.png",
                        height=300
                    )
                    audio_path_input = gr.Audio(
                        label="Audio Input (Start / Insert)", 
                        type="filepath", 
                        value="examples/podcast_sichuan_16k.wav"
                    )

            generate_btn = gr.Button("🚀 Start Live Stream", variant="primary", size="lg")
            stop_btn = gr.Button("🛑 Stop Broadcast", variant="stop", size="lg")

            # --- 【精简版】插播控制台 ---
            with gr.Group():
                gr.Markdown("### 🎤 Live Interaction")
                gr.Markdown("直播中想换词？直接在上方 **Audio Input** 清空并上传新音频，然后点击下方按钮插播！")
                insert_btn = gr.Button("⚡ Insert Current Audio into Stream")
                insert_status = gr.Textbox(label="Status", interactive=False)

            with gr.Accordion("⚙️ Advanced Settings & Model Configuration", open=False):
                with gr.Tabs():
                    with gr.TabItem("Model Paths"):
                        model_type_input = gr.Dropdown(
                            label="FlashHead Model Type", 
                            choices=["pro", "lite"],
                            value="lite"
                        )
                        ckpt_dir_input = gr.Textbox(
                            label="FlashHead Checkpoint Directory", 
                            value="models/SoulX-FlashHead-1_3B",
                        )
                        wav2vec_dir_input = gr.Textbox(
                            label="Wav2Vec Directory", 
                            value="models/wav2vec2-base-960h",
                        )

                    with gr.TabItem("Inference Params"):
                        use_face_crop_input = gr.Checkbox(
                            label="Use Face Crop", 
                            value=False
                        )
                        seed_input = gr.Number(
                            label="Random Seed", 
                            value=9999, 
                            precision=0
                        )

        with gr.Column(scale=1):
            gr.Markdown("### 📺 Live Broadcast Stream")
            html_output = gr.HTML(
                label="Live Stream Viewer", 
                value="<div style='width: 100%; height: 500px; background: black; display: flex; align-items: center; justify-content: center; color: white;'>Waiting for live stream to start...</div>"
            )

            download_output = gr.File(
                label="📥 Download Component (Hidden)", 
                visible=False
            )

    stream_event = generate_btn.click(
        fn=run_inference,
        inputs=[
            ckpt_dir_input,
            wav2vec_dir_input,
            model_type_input,
            cond_image_input,
            audio_path_input,
            seed_input,
            use_face_crop_input
        ],
        outputs=[html_output, download_output],
        queue=True
    )

    stop_btn.click(fn=None, inputs=None, outputs=None, cancels=[stream_event])

    # --- 【修改这里】直接提取 audio_path_input 里的新文件 ---
    insert_btn.click(
        fn=insert_dynamic_audio,
        inputs=[audio_path_input],  # <--- 直接复用它！
        outputs=[insert_status]
    )

if __name__ == "__main__":
    import uvicorn
    from fastapi import FastAPI
    from fastapi.staticfiles import StaticFiles
    
    # 1. 获取推流目录的绝对路径，确保 FastAPI 绝对能找到它
    stream_dir = os.path.abspath("gradio_results_live")
    os.makedirs(stream_dir, exist_ok=True)
    
    # 2. 创建一个纯血的 FastAPI 原生应用
    fastapi_app = FastAPI()
    
    # 3. 稳稳当当地挂载我们的视频切片目录
    # 现在的 /stream 路由是写死在 FastAPI 核心里的，神仙也覆盖不掉！
    fastapi_app.mount("/stream", StaticFiles(directory=stream_dir), name="stream")
    
    # 4. 把我们的 Gradio 应用 (app) 作为一个子应用，挂载到根目录 "/"
    fastapi_app = gr.mount_gradio_app(fastapi_app, app, path="/")
    
    # 5. 使用工业级网关 Uvicorn 启动服务
    logger.info("🚀 Starting Production Server with Uvicorn on port 7860...")
    uvicorn.run(fastapi_app, host="0.0.0.0", port=7860)