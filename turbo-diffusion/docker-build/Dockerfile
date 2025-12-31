# 1. Base Image: VESSL AI 제공 CUDA 이미지 (final.yaml과 동일)
FROM quay.io/vessl-ai/cuda:12.4-r4

# 2. 환경 변수 설정
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="8.0"
ENV PATH="/usr/local/bin:$PATH"

# 3. 기본 시스템 패키지 설치 (SSH 서버 포함 - VESSL Connect용)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates wget vim \
    ffmpeg libsm6 libxext6 \
    build-essential ninja-build \
    openssh-server \
    && rm -rf /var/lib/apt/lists/*

# SSH 데몬 설정
RUN mkdir -p /var/run/sshd

# 4. PyTorch 2.6.0 설치 (final.yaml 버전 맞춤)
RUN pip install --upgrade pip ninja cmake packaging wheel && \
    pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# 5. JupyterLab 설치 (VESSL 인터랙티브 모드 필수)
RUN pip install --no-cache-dir jupyterlab ipywidgets

# 6. 소스코드 클론 및 패치 (TurboDiffusion)
WORKDIR /root
RUN git clone https://github.com/thu-ml/TurboDiffusion.git && \
    cd TurboDiffusion && \
    git submodule update --init --recursive

# 패치 적용 (final.yaml의 sed 명령어들)
WORKDIR /root/TurboDiffusion
RUN sed -i '/arch=compute_120a,code=sm_120a/d' setup.py && \
    sed -i '/"torch>=/d' pyproject.toml && \
    sed -i '/"torchvision"/d' pyproject.toml && \
    sed -i '/"triton>=/d' pyproject.toml && \
    sed -i '/"flash-attn"/d' pyproject.toml && \
    sed -i 's/, dtype=tl.float32//g' turbodiffusion/SLA/utils.py

# 7. [핵심] 로컬 휠 파일 복사 및 설치 (빌드 시간 0초)
COPY wheels /root/wheels
RUN pip install /root/wheels/*.whl

# 8. TurboDiffusion 설치 (나머지 가벼운 의존성)
RUN pip install -e . --no-build-isolation

# 작업 디렉토리 설정
WORKDIR /root/TurboDiffusion