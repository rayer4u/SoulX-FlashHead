import gradio as gr
import os
import torch
import numpy as np
import time
import imageio
import librosa
import subprocess
import soundfile as sf
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

def run_multi_gpu_inference(
    gpu_ids,
    ckpt_dir,
    wav2vec_dir,
    model_type,
    cond_image,
    audio_path,
    audio_encode_mode,
    use_face_crop,
    seed,
    progress=gr.Progress()
):
    """
    Executes the inference using torchrun for Multi-GPU support.
    """
    gpu_list = [x.strip() for x in gpu_ids.split(',') if x.strip()]
    num_gpus = len(gpu_list)
    if num_gpus == 0:
        raise gr.Error("Please specify at least one GPU ID (e.g., '0,1,2,3').")

    cuda_visible_devices = ",".join(gpu_list)
    
    # Define output path beforehand to know where to look
    output_dir = 'gradio_results_multigpu'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
    # Note: generate_video.py generates its own filename, so we pass --save_file to control it
    filename = f"res_{timestamp}.mp4"
    save_path = os.path.abspath(os.path.join(output_dir, filename))

    # Construct the command
    # CUDA_VISIBLE_DEVICES=... torchrun --nproc_per_node=... generate_video.py ...
    cmd = [
        "torchrun",
        f"--nproc_per_node={num_gpus}",
        "generate_video.py",
        "--ckpt_dir", ckpt_dir,
        "--wav2vec_dir", wav2vec_dir,
        "--model_type", model_type,
        "--cond_image", cond_image,
        "--audio_path", audio_path,
        "--audio_encode_mode", audio_encode_mode,
        "--use_face_crop", str(use_face_crop),
        "--base_seed", str(int(seed)),
        "--save_file", save_path,
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    
    logger.info(f"Starting Multi-GPU inference with command: {' '.join(cmd)}")
    logger.info(f"Visible Devices: {cuda_visible_devices}")
    
    progress(0, desc="Starting Multi-GPU process... (Check terminal for logs)")
    
    try:
        # Run the command
        process = subprocess.Popen(
            cmd, 
            env=env, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True, 
            bufsize=1, 
            universal_newlines=True
        )
        
        # Read output line by line to update progress (simple heuristic)
        for line in process.stdout:
            print(line, end='') # Print to console for debugging
            if "Generate video chunk" in line:
                progress(0.5, desc="Generating chunks...")
            elif "Saving generated video" in line:
                progress(0.9, desc="Saving video...")
        
        process.wait()
        
        if process.returncode != 0:
            raise gr.Error(f"Multi-GPU process failed with return code {process.returncode}")
            
    except Exception as e:
        logger.error(f"Error during multi-gpu execution: {e}")
        raise gr.Error(f"Multi-GPU execution failed: {e}")

    if os.path.exists(save_path):
        return save_path
    else:
        raise gr.Error("Output video file was not found. Check terminal logs for errors.")

def save_video_to_file(frames_list, video_path, audio_path, fps):
    """
    Helper function to save the video, similar to generate_video.py but adapted for function usage.
    """
    temp_video_path = video_path.replace('.mp4', '_temp.mp4')
    
    # Make sure directory exists
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    
    try:
        with imageio.get_writer(temp_video_path, format='mp4', mode='I',
                                fps=fps , codec='h264', ffmpeg_params=['-bf', '0']) as writer:
            for frames in frames_list:
                frames = frames.numpy().astype(np.uint8)
                for i in range(frames.shape[0]):
                    frame = frames[i, :, :, :]
                    writer.append_data(frame)
        
        # merge video and audio
        # Use aac audio codec for better compatibility instead of copy
        # This handles cases where input audio (like PCM wav) is not supported in MP4 container
        cmd = ['ffmpeg', '-i', temp_video_path, '-i', audio_path, '-c:v', 'copy', '-c:a', 'aac', '-shortest', video_path, '-y']
        subprocess.run(cmd, check=True)
    except Exception as e:
        logger.error(f"Error saving video: {e}")
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        raise e
    finally:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
    
    return video_path

def save_video_fragment(frames_list, output_path, fps, audio_path=None):
    """【纯净 TS 生成器】：确保输出的 TS 切片时间戳永远是从 0.0000 严格开始"""
    if not output_path.endswith('.ts'):
        output_path = output_path.rsplit('.', 1)[0] + '.ts'

    temp_video_path = output_path.replace('.ts', '_temp.mp4')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        # 强制指定浏览器唯一支持的 yuv420p
        with imageio.get_writer(temp_video_path, format='mp4', mode='I',
                                fps=fps, codec='h264') as writer:
            for frames in frames_list:
                frames = frames.numpy().astype(np.uint8)
                for i in range(frames.shape[0]):
                    writer.append_data(frames[i, :, :, :])

        if audio_path:
            # 引入魔法参数：-avoid_negative_ts make_zero，强制打平时间戳！
            final_cmd = [
                'ffmpeg', '-y', '-i', temp_video_path, '-i', audio_path,
                '-c:v', 'copy', '-c:a', 'aac', '-ar', '44100',
                '-avoid_negative_ts', 'make_zero', '-f', 'mpegts', output_path
            ]
            subprocess.run(final_cmd, check=True, capture_output=True)
        else:
            final_cmd = [
                'ffmpeg', '-y', '-i', temp_video_path, '-c', 'copy',
                '-avoid_negative_ts', 'make_zero', '-f', 'mpegts', output_path
            ]
            subprocess.run(final_cmd, check=True, capture_output=True)

        return output_path
    finally:
        if os.path.exists(temp_video_path): os.remove(temp_video_path)
        if audio_path and os.path.exists(audio_path): os.remove(audio_path) # 清理临时 wav


def run_inference(
    ckpt_dir,
    wav2vec_dir,
    model_type,
    cond_image,
    audio_path,
    audio_encode_mode,
    seed,
    use_face_crop,
    progress=gr.Progress()
):
    global pipeline, loaded_ckpt_dir, loaded_wav2vec_dir, loaded_model_type

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
    
    # Handle seed
    base_seed = int(seed) if seed >= 0 else 9999

    # Prepare base data (prompt, image)
    try:
        get_base_data(pipeline, cond_image_path_or_dir=cond_image, base_seed=base_seed, use_face_crop=use_face_crop)
    except Exception as e:
        logger.error(f"Error in get_base_data: {e}")
        raise gr.Error(f"Error processing inputs: {e}")

    # Get parameters from global config (infer_params)
    infer_params = get_infer_params()

    sample_rate = infer_params['sample_rate']
    tgt_fps = infer_params['tgt_fps']
    cached_audio_duration = infer_params['cached_audio_duration']
    frame_num = infer_params['frame_num']
    motion_frames_num = infer_params['motion_frames_num']
    slice_len = frame_num - motion_frames_num
    
    generated_list = []

    # Load Audio
    try:
        human_speech_array_all, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    except Exception as e:
        raise gr.Error(f"Failed to load audio file: {e}")

    human_speech_array_slice_len = slice_len * sample_rate // tgt_fps
    human_speech_array_frame_num = frame_num * sample_rate // tgt_fps

    logger.info("Data preparation done. Start to generate video...")
    output_dir = 'gradio_results'
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
    base_filename = f"res_{timestamp}"

    # 3. Generation Loop
    if audio_encode_mode == 'once':
        # pad audio with silence to avoid truncating the last chunk
        remainder = (len(human_speech_array_all) - human_speech_array_frame_num) % human_speech_array_slice_len
        if remainder > 0:
            pad_length = human_speech_array_slice_len - remainder
            human_speech_array_all = np.concatenate([human_speech_array_all, np.zeros(pad_length, dtype=human_speech_array_all.dtype)])

        audio_embedding_all = get_audio_embedding(pipeline, human_speech_array_all)
        audio_embedding_chunks_list = [audio_embedding_all[:, i * slice_len: i * slice_len + frame_num].contiguous() for i in range((audio_embedding_all.shape[1]-frame_num) // slice_len)]
        
        total_chunks = len(audio_embedding_chunks_list)
        for chunk_idx, audio_embedding_chunk in enumerate(audio_embedding_chunks_list):
            progress(0.2 + 0.7 * (chunk_idx / total_chunks), desc=f"Generating chunk {chunk_idx+1}/{total_chunks}")
            
            torch.cuda.synchronize()
            start_time = time.time()

            # inference
            video = run_pipeline(pipeline, audio_embedding_chunk)

            if chunk_idx != 0:
                video = video[motion_frames_num:]

            torch.cuda.synchronize()
            end_time = time.time()
            logger.info(f"Generate video chunk-{chunk_idx} done, cost time: {(end_time - start_time):.2f}s")
            
            generated_list.append(video.cpu())

        # 4. Save Video
        progress(0.95, desc="Saving Video...")

        final_save_path = os.path.join(output_dir, f"{base_filename}.mp4")
        final_video_path = save_video_to_file(generated_list, final_save_path, audio_path, fps=tgt_fps)
        logger.info(f"Saved final video to {final_video_path}")

    elif audio_encode_mode == 'stream':
        chunk_duration = frame_num / tgt_fps
        stream_duration = 3.0  # 
        dynamic_stream_interval = max(1, int(stream_duration / chunk_duration))
        stream_chunk_list = []
        current_audio_sample_idx = 0

        cached_audio_length_sum = sample_rate * cached_audio_duration
        audio_end_idx = cached_audio_duration * tgt_fps
        audio_start_idx = audio_end_idx - frame_num

        audio_dq = deque([0.0] * cached_audio_length_sum, maxlen=cached_audio_length_sum)

        # pad audio with silence to avoid truncating the last chunk
        remainder = len(human_speech_array_all) % human_speech_array_slice_len
        if remainder > 0:
            human_speech_array_all = np.concatenate([human_speech_array_all, np.zeros(human_speech_array_slice_len - remainder, dtype=human_speech_array_all.dtype)])

        human_speech_array_slices = human_speech_array_all.reshape(-1, human_speech_array_slice_len)

        for chunk_idx, human_speech_array in enumerate(human_speech_array_slices):
            audio_dq.extend(human_speech_array.tolist())
            audio_array = np.array(audio_dq)
            audio_embedding = get_audio_embedding(pipeline, audio_array, audio_start_idx, audio_end_idx)

            video = run_pipeline(pipeline, audio_embedding)
            video = video[motion_frames_num:]

            generated_list.append(video.cpu())
            stream_chunk_list.append(video.cpu())

            if (chunk_idx + 1) % dynamic_stream_interval == 0:
                fragment_path = os.path.join(output_dir, f"{base_filename}_part{chunk_idx+1}.ts")

                frames_in_fragment = sum(v.shape[0] for v in stream_chunk_list)
                samples_in_fragment = int(frames_in_fragment * sample_rate / tgt_fps)

                audio_fragment = human_speech_array_all[current_audio_sample_idx : current_audio_sample_idx + samples_in_fragment]
                current_audio_sample_idx += samples_in_fragment

                temp_audio_path = os.path.join(output_dir, f"temp_audio_{chunk_idx}.wav")
                sf.write(temp_audio_path, audio_fragment, sample_rate)

                save_video_fragment(stream_chunk_list, fragment_path, fps=tgt_fps, audio_path=temp_audio_path)
                logger.info(f"Streamed fragment to {fragment_path}")
                yield fragment_path, gr.update(visible=False)
                stream_chunk_list = []

        final_save_path = os.path.join(output_dir, f"{base_filename}.mp4")
        final_video_path = save_video_to_file(generated_list, final_save_path, audio_path, fps=tgt_fps)
        logger.info(f"Saved final video to {final_video_path}")

        yield gr.update(), gr.update(value=final_video_path, visible=True)


with gr.Blocks(title="SoulX-FlashHead Video Generator", theme=gr.themes.Soft()) as app:
    gr.Markdown("# ⚡ SoulX-FlashHead Video Generator")
    gr.Markdown("Upload an image and an audio file to generate a talking head video.")

    with gr.Row():
        with gr.Column(scale=1):
            # 1. Main Inputs Section (Always Visible)
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
                        label="Audio Input", 
                        type="filepath", 
                        value="examples/podcast_sichuan_16k.wav"
                    )

            # 2. Main Action Button
            generate_btn = gr.Button("🚀 Generate Video", variant="primary", size="lg")

            # 3. Advanced Configuration (Collapsed by default to save space)
            with gr.Accordion("⚙️ Advanced Settings & Model Configuration", open=False):
                with gr.Tabs():
                    with gr.TabItem("Execution Mode"):
                        model_type_input = gr.Dropdown(
                            label="FlashHead Model Type", 
                            choices=[
                                ("Pro Version (Multi-GPU Support)", "pro"),
                                ("Lite Version (Single GPU Only)", "lite")
                            ],
                            value="lite",
                            info="Select the model variant. 'pro' supports both single and multi-GPU, 'lite' is single GPU only."
                        )
                        mode_input = gr.Radio(
                            choices=["Single GPU", "Multi-GPU"],
                            value="Single GPU",
                            label="Execution Mode",
                            visible=True,
                            info="Single GPU: Keeps model in memory for fast interactive use. Multi-GPU: Spawns new process, better for stability/isolation."
                        )
                        gpu_ids_input = gr.Textbox(
                            label="GPU IDs (Multi-GPU only)",
                            value="0,1",
                            visible=False,
                            placeholder="0,1,2,3",
                        )

                    with gr.TabItem("Model Paths"):
                        ckpt_dir_input = gr.Textbox(
                            label="FlashHead Checkpoint Directory", 
                            value="models/SoulX-FlashHead-1_3B",
                            info="Path to the FlashHead model checkpoint."
                        )
                        wav2vec_dir_input = gr.Textbox(
                            label="Wav2Vec Directory", 
                            value="models/wav2vec2-base-960h",
                            info="Path to the Wav2Vec model checkpoint."
                        )

                    with gr.TabItem("Inference Params"):
                        audio_encode_mode_input = gr.Radio(
                            label="Audio Encode Mode", 
                            choices=["stream", "once"], 
                            value="stream",
                            info="Stream: chunk-by-chunk; Once: all at once."
                        )
                        use_face_crop_input = gr.Checkbox(
                            label="Use Face Crop", 
                            value=False,
                            info="Enable face detection and crop for condition image."
                        )
                        seed_input = gr.Number(
                            label="Random Seed", 
                            value=9999, 
                            precision=0
                        )

        with gr.Column(scale=1):
            gr.Markdown("### 📺 Output Video")
            # 开启 Gradio 5 原生的 TS 联播引擎
            video_output = gr.Video(
                label="Generated Video", 
                height=500,
                autoplay=True,
                streaming=True,  
            )
            download_output = gr.File(
                label="📥 Download High-Quality Complete Video", 
                visible=False
            )

    # Event Handlers
    def update_visibility(model_type, mode):
        if model_type == "lite":
            return [
                gr.update(visible=False, value="Single GPU"),  # mode_input
                gr.update(visible=False),  # gpu_ids_input
            ]
        else:  # pro
            if mode == "Multi-GPU":
                return [
                    gr.update(visible=True),  # mode_input
                    gr.update(visible=True),  # gpu_ids_input
                ]
            else:  # Single GPU
                return [
                    gr.update(visible=True),  # mode_input
                    gr.update(visible=False),  # gpu_ids_input
                ]

    model_type_input.change(fn=update_visibility, inputs=[model_type_input, mode_input], outputs=[mode_input, gpu_ids_input])
    mode_input.change(fn=update_visibility, inputs=[model_type_input, mode_input], outputs=[mode_input, gpu_ids_input])

    def dispatch_inference(
        mode, gpu_ids, ckpt, wav2vec, model_type, img, audio, enc_mode, seed, use_face_crop
    ):
        if mode == "Single GPU":
            if enc_mode == "stream":
                yield from run_inference(
                    ckpt, wav2vec, model_type, img, audio, enc_mode, seed, use_face_crop,
                )
            else:
                return run_inference(ckpt, wav2vec, model_type, img, audio, enc_mode, seed, use_face_crop)
        else:
            return run_multi_gpu_inference(gpu_ids, ckpt, wav2vec, model_type, img, audio, enc_mode, seed, use_face_crop)

    generate_btn.click(
        fn=dispatch_inference,
        inputs=[
            mode_input,
            gpu_ids_input,
            ckpt_dir_input,
            wav2vec_dir_input,
            model_type_input,
            cond_image_input,
            audio_path_input,
            audio_encode_mode_input,
            seed_input,
            use_face_crop_input
        ],
        outputs=[video_output, download_output],
        queue=True
    )

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
