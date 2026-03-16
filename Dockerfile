FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-devel

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg
RUN pip install uv
COPY requirements-docker.txt /workspace/requirements.txt
# RUN uv venv
# ENV PATH="/workspace/.venv/bin:$PATH"
# RUN uv pip install torch==2.8.0+cu126 torchvision --index-url https://download.pytorch.org/whl/cu126
RUN uv pip install --system -r requirements.txt
# RUN uv pip install flash-attn --no-build-isolation
COPY flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl /workspace
RUN uv pip install --system flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
COPY ./ /workspace/

#monkey hack
RUN sed -i 's/2.8.3/2.8.2/g' /opt/conda/lib/python3.11/site-packages/flash_attn/__init__.py

ENTRYPOINT ["python", "gradio_app.py"]