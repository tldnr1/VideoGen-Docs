# TurboDiffusion 영상 생성 시간 제약 분석

## 개요

TurboDiffusion Wan2.2-I2V 모델에서 실제 사용된 파라미터와 영상 생성 길이의 최소/최대 제약을 코드 분석을 통해 정리한 문서입니다.

---

## 1. 실제 사용된 파라미터 분석

### 1-1. 스크립트에서 사용된 파라미터

**스크립트 경로**: `scripts/inference_wan2.2_i2v.sh`

```bash
python turbodiffusion/inference/wan2.2_i2v_infer.py \
    --model Wan2.2-A14B \
    --low_noise_model_path checkpoints/TurboWan2.2-I2V-A14B-low-720P-quant.pth \
    --high_noise_model_path checkpoints/TurboWan2.2-I2V-A14B-high-720P-quant.pth \
    --resolution 720p \
    --adaptive_resolution \
    --image_path assets/i2v_inputs/i2v_input_0.jpg \
    --prompt "..." \
    --num_samples 1 \
    --num_steps 4 \
    --quant_linear \
    --attention_type sagesla \
    --sla_topk 0.1 \
    --ode
```

### 1-2. 기본값으로 사용된 파라미터

**코드 경로**: `turbodiffusion/inference/wan2.2_i2v_infer.py::parse_arguments()`

```python
parser.add_argument("--num_frames", type=int, default=81, help="Number of frames to generate")
```

- **`--num_frames`**: 명시되지 않음 → **기본값 81 프레임** 사용
- **FPS**: 코드에서 하드코딩된 **16fps**
- **현재 생성 영상**: 81 프레임 ÷ 16fps = **약 5.06초**

**기타 기본값**:
- `--sigma_max`: 200 (I2V 모드 기본값)
- `--boundary`: 0.9
- `--seed`: 0
- `--aspect_ratio`: "16:9"

---

## 2. 최대 영상 길이 제약 분석

### 2-1. VAE 제약 (실제 최대값)

**코드 경로**: `turbodiffusion/rcm/tokenizers/wan2pt1.py`

```python
# VAE 초기화 시 video_mean/video_std 버퍼 생성
torch.zeros(1, 1, 50, 1, 1, device=device),  # video_mean
torch.ones(1, 1, 50, 1, 1, device=device),   # video_std
```

**VAE 인코딩/디코딩 시 사용**:
```python
def encode(self, state: torch.Tensor) -> torch.Tensor:
    latents = self.model.encode(state)
    num_frames = latents.shape[2]
    if num_frames == 1:
        return (latents - self.model.img_mean.type_as(latents)) / self.model.img_std.type_as(latents)
    else:
        # video_mean[:, :, :num_frames] 슬라이싱 - 최대 50 프레임
        return (latents - self.model.video_mean[:, :, :num_frames].type_as(latents)) / self.model.video_std[:, :, :num_frames].type_as(latents)
```

**제약 사항**:
- **Latent 프레임 최대**: 50 프레임
- **픽셀 프레임 최대**: `(50-1) * 4 + 1 = 197 프레임`
- **영상 길이**: 197 ÷ 16 = **약 12.3초**

### 2-2. 모델 학습 길이 (권장 최대값)

**코드 경로**: `turbodiffusion/rcm/networks/wan2pt2.py`

```python
# 모델 초기화 시 RoPE Position Embedding 설정
self.rope_position_embedding = VideoRopePosition3DEmb(
    head_dim=d, 
    len_h=128, 
    len_w=128, 
    len_t=32  # 시간 차원 최대값
)
```

**Position Embedding 생성**:
```python
def cache_parameters(self) -> None:
    # seq 버퍼는 max(max_h, max_w, max_t) 크기로 생성
    self.seq = torch.arange(max(self.max_h, self.max_w, self.max_t)).float().cuda()
    # max(128, 128, 32) = 128

def generate_embeddings(self, B_T_H_W_C: torch.Size, ...):
    B, T, H, W, _ = B_T_H_W_C
    # 시간 차원에 대한 assertion은 없음 (H, W만 체크)
    freqs_t = torch.outer(self.seq[:T], temporal_freqs)  # T <= 128까지 가능
```

**제약 사항**:
- **Latent 프레임 권장**: 32 프레임 (모델 학습 길이)
- **픽셀 프레임 권장**: `(32-1) * 4 + 1 = 125 프레임`
- **영상 길이**: 125 ÷ 16 = **약 7.8초**

### 2-3. 기술적 최대값

**Position Embedding 버퍼 크기**:
```python
self.seq = torch.arange(max(128, 128, 32)).float().cuda()  # 길이 128
freqs_t = torch.outer(self.seq[:T], temporal_freqs)  # T > 128이면 인덱싱 오류
```

**제약 사항**:
- **Latent 프레임 기술적 최대**: 128 프레임
- **픽셀 프레임 기술적 최대**: `(128-1) * 4 + 1 = 509 프레임`
- **영상 길이**: 509 ÷ 16 = **약 31.8초**
- **주의**: VAE 제약(197 프레임)으로 인해 실제로는 불가능

### 2-4. 프레임 변환 공식

**코드 경로**: `turbodiffusion/rcm/tokenizers/wan2pt1.py`

```python
def get_latent_num_frames(self, num_pixel_frames: int) -> int:
    return 1 + (num_pixel_frames - 1) // 4

def get_pixel_num_frames(self, num_latent_frames: int) -> int:
    return (num_latent_frames - 1) * 4 + 1
```

**Temporal compression factor**: 4

---

## 3. 최소 영상 길이 제약 분석

### 3-1. 명시적 최소값 제약

**코드 경로**: `turbodiffusion/serve/utils.py`

```python
RUNTIME_PARAMS = {
    "num_frames": {"type": int, "min": 1},
    ...
}
```

**제약**: 최소 **1 프레임** (약 0.06초, 16fps 기준)

### 3-2. VAE Encoding 최소 제약

**코드 경로**: `turbodiffusion/rcm/tokenizers/wan2pt1.py`

```python
def encode(self, x, scale):
    t = x.shape[2]  # 프레임 수
    iter_ = 1 + (t - 1) // self.temporal_window  # temporal_window=4
    
    # t=1일 때: iter_ = 1 + (1-1)//4 = 1
    if i == 0:
        out = self._i0_encode(x)  # 첫 프레임만 처리
```

**VAE Interface**:
```python
def encode(self, state: torch.Tensor) -> torch.Tensor:
    latents = self.model.encode(state)
    num_frames = latents.shape[2]
    if num_frames == 1:
        # 이미지 처리 경로 사용
        return (latents - self.model.img_mean.type_as(latents)) / self.model.img_std.type_as(latents)
    else:
        # 비디오 처리 경로 사용
        return (latents - self.model.video_mean[:, :, :num_frames].type_as(latents)) / self.model.video_std[:, :, :num_frames].type_as(latents)
```

**제약**: **1 프레임 이상** 처리 가능 (이미지/비디오 경로 모두 지원)

### 3-3. 모델 Forward 최소 제약

**코드 경로**: `turbodiffusion/rcm/networks/wan2pt2.py`

```python
def forward(self, x_B_C_T_H_W, ...):
    kt, kh, kw = self.patch_size  # (1, 2, 2)
    B, _, T_in, H_in, W_in = x_B_C_T_H_W.shape
    assert (T_in % kt) == 0 and (H_in % kh) == 0 and (W_in % kw) == 0
    # kt=1이므로 T_in % 1 == 0은 항상 참
    T, H, W = T_in // kt, H_in // kh, W_in // kw
```

**Position Embedding**:
```python
def generate_embeddings(self, B_T_H_W_C: torch.Size, ...):
    B, T, H, W, _ = B_T_H_W_C
    # H, W에 대한 최대 제약만 있음, T에 대한 최소 제약 없음
    assert (H <= self.max_h and W <= self.max_w)
    freqs_t = torch.outer(self.seq[:T], temporal_freqs)  # T >= 1이면 동작
```

**제약**: 시간 차원에 대한 **최소 제약 없음** (T >= 1이면 동작)

### 3-4. I2V 모드 특수 처리

**코드 경로**: `turbodiffusion/inference/wan2.2_i2v_infer.py`

```python
# 이미지 + (F-1)개의 zero frames로 구성
frames_to_encode = torch.cat(
    [image_tensor.unsqueeze(2), 
     torch.zeros(1, 3, F - 1, h, w, device=image_tensor.device)], 
    dim=2
)
# F=1일 때: 이미지만 사용 (정적 이미지)
# F>1일 때: 이미지 + (F-1)개의 zero frames
```

**제약**: F >= 1이면 동작 가능

---

## 4. 제약 요약 및 권장사항

### 4-1. 제약 요약

| 제약 소스 | Latent 프레임 | 픽셀 프레임 | 영상 길이 (16fps) | 상태 |
|----------|--------------|------------|------------------|------|
| **VAE 제약** | 50 (최대) | 197 | 12.3초 | 실제 최대값 |
| **모델 학습 길이** | 32 (권장) | 125 | 7.8초 | 권장 최대값 |
| **기술적 최대** | 128 | 509 | 31.8초 | VAE 제약으로 불가능 |
| **최소값** | 1 | 1 | 0.06초 | 기술적 최소 |

### 4-2. 프레임 수 계산 방법

```
프레임 수 = 목표 초 × 16 + 1 (FPS)
```

**예시**:
- 3초: 3 × 16 + 1 = **49 프레임**
- 5초: 5 × 16 + 1 = **81 프레임**
- 10초: 10 × 16 + 1 = **161 프레임**
- 15초: 15 × 16 + 1 = **241 프레임** (⚠️ VAE 제약 초과 예상)

### 4-3. 권장 범위

**안전 범위**:
- **최소**: 32 프레임 (2초)
- **권장**: 48~125 프레임 (3~7.8초)
- **기본값**: 81 프레임 (5초)

**확장 범위**:
- **125~197 프레임** (7.8~12.3초): 기술적으로 가능하나 품질 보장 어려움

**주의사항**:
1. **15초 이상 생성 불가**: VAE 제약으로 인해 197 프레임(약 12.3초)이 실제 최대값
2. **품질 고려**: 모델 학습 길이(125 프레임)를 초과하면 품질 저하 가능
3. **I2V 모드**: 첫 프레임이 입력 이미지이므로, 실제 생성 프레임은 `num_frames - 1`

---

## 5. 사용 예시

### 5-1. 3초 영상 생성
```bash
python turbodiffusion/inference/wan2.2_i2v_infer.py \
    --num_frames 49 \
    ...
```

### 5-2. 5초 영상 생성
```bash
python turbodiffusion/inference/wan2.2_i2v_infer.py \
    --num_frames 81 \
    ...
```

### 5-3. 10초 영상 생성
```bash
python turbodiffusion/inference/wan2.2_i2v_infer.py \
    --num_frames 161 \
    ...
```

---

## 참고사항

- 이 분석은 TurboDiffusion Wan2.2-I2V 모델 기준입니다
- 다른 모델(Wan2.1-T2V 등)도 유사한 제약이 적용됩니다
- FPS는 코드에서 16fps로 하드코딩되어 있습니다
