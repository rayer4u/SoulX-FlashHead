# 安装

```
# 模型
huggingface-cli download Soul-AILab/SoulX-FlashHead-1_3B --local-dir ./models/SoulX-FlashHead-1_3B
huggingface-cli download facebook/wav2vec2-base-960h --local-dir ./models/wav2vec2-base-960h

# flash_attn
curl -JOL https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl

# 编译运行
docker-compose build && docker-compose up -d

```