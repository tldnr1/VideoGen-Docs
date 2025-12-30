# Index
* 요약
* Wan2.2-I2V Params, configs
* Wan API References
* Other Video Gen API

## 요약
- Wan2.2 Github 기준, 해상도는 픽셀 직접 지정이 아닌 `Aspect Ratio` + `Max Area` 방식으로 계산됨 (지원되는 값이 config에 존재)
- JSON API의 경우 일반적으로 아래의 내용들을 사용하는 것으로 보임
``` json
{
  "task_type": "i2v",
  
  // 공통 입력
  "image_url": "https://s3.bucket/input.jpg",
  "prompt": "Positive prompt describing the motion...",
  "negative_prompt": "Low quality, distortion", // (Optional)
  
  // 규격 설정
  "duration": 5,           // 초 단위. (Backend에서 5 * 16 + 1 = 81 프레임으로 변환)
  "resolution": "720p",    // "720p" | "480p" (Backend에서 Max Area 및 Shift 값 자동 매핑)
  "aspect_ratio": "16:9",  // "16:9" | "9:16" | "auto" (Backend에서 이미지 Crop 전처리 수행)

  // 품질 옵션
  "enable_prompt_expansion": true, // LLM을 통한 프롬프트 확장 여부
  "seed": -1                       // -1: Random

  // "shot_type": "multi" 와 같은 옵션이 보이기도 하는데, v2.6 혹은 준하는 기능을 가진 버전에 대한 것으로 보임
  // guidance, step 등은 상위 버전에서 보통 자동으로 수행하는 것으로 보임
}
```

- 계산되는 값
``` json
- resolution mapping
720p (1280, 720): shift=5.0
480p (832, 480): shift=3.0

- duration ~ frame_num
duration=5(초)
5초 * 16fps + 1 = 81 frames
```

## Wan2.2-I2V Params, configs
### Params
``` json
// Wan2.2 Github - Multi-GPU inference using FSDP + DeepSpeed Ulysses
torchrun --nproc_per_node=8 generate.py 
    --task i2v-A14B 
    --size 1280*720 
    --ckpt_dir ./Wan2.2-I2V-A14B 
    --image examples/i2v_input.JPG 
    --dit_fsdp 
    --t5_fsdp 
    --ulysses_size 8 
    --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
```

``` python
# Wan2.2/generate.py - line 530 ~ 540
video = wan_i2v.generate(
            args.prompt,
            img,
            max_area=MAX_AREA_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model)
```

``` python
# Wan2.2/wan/image2video.py - line 206 ~ 217 + Args 설명
def generate(self,
                 input_prompt,
                 img,
                 max_area=720 * 1280,
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=40,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True):
```

### configs
``` python
# Wan2.2/wan/configs/__init__.py 중 i2v, s2v 관련 발췌
import copy
import os

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from .wan_i2v_A14B import i2v_A14B
from .wan_s2v_14B import s2v_14B

WAN_CONFIGS = {
    'i2v-A14B': i2v_A14B,
    's2v-14B': s2v_14B,
}

SIZE_CONFIGS = {
    '720*1280': (720, 1280),
    '1280*720': (1280, 720),
    '480*832': (480, 832),
    '832*480': (832, 480),
    '704*1280': (704, 1280),
    '1280*704': (1280, 704),
    '1024*704': (1024, 704),
    '704*1024': (704, 1024),
}

MAX_AREA_CONFIGS = {
    '720*1280': 720 * 1280,
    '1280*720': 1280 * 720,
    '480*832': 480 * 832,
    '832*480': 832 * 480,
    '704*1280': 704 * 1280,
    '1280*704': 1280 * 704,
    '1024*704': 1024 * 704,
    '704*1024': 704 * 1024,
}

SUPPORTED_SIZES = {
    'i2v-A14B': ('720*1280', '1280*720', '480*832', '832*480'),
    's2v-14B': ('720*1280', '1280*720', '480*832', '832*480', '1024*704',
                '704*1024', '704*1280', '1280*704')
}
```

``` python
# Wan2.2/wan/configs/wan_i2v_A14B.py
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
from easydict import EasyDict

from .shared_config import wan_shared_cfg

#------------------------ Wan I2V A14B ------------------------#

i2v_A14B = EasyDict(__name__='Config: Wan I2V A14B')
i2v_A14B.update(wan_shared_cfg)

i2v_A14B.t5_checkpoint = 'models_t5_umt5-xxl-enc-bf16.pth'
i2v_A14B.t5_tokenizer = 'google/umt5-xxl'

# vae
i2v_A14B.vae_checkpoint = 'Wan2.1_VAE.pth'
i2v_A14B.vae_stride = (4, 8, 8)

# transformer
i2v_A14B.patch_size = (1, 2, 2)
i2v_A14B.dim = 5120
i2v_A14B.ffn_dim = 13824
i2v_A14B.freq_dim = 256
i2v_A14B.num_heads = 40
i2v_A14B.num_layers = 40
i2v_A14B.window_size = (-1, -1)
i2v_A14B.qk_norm = True
i2v_A14B.cross_attn_norm = True
i2v_A14B.eps = 1e-6
i2v_A14B.low_noise_checkpoint = 'low_noise_model'
i2v_A14B.high_noise_checkpoint = 'high_noise_model'

# inference
i2v_A14B.sample_shift = 5.0
i2v_A14B.sample_steps = 40
i2v_A14B.boundary = 0.900
i2v_A14B.sample_guide_scale = (3.5, 3.5)  # low noise, high noise
```

## Wan API References
### [v2.2-i2v / Fal.ai](https://fal.ai/models/fal-ai/wan/v2.2-a14b/image-to-video)
``` json
// INPUT
{
  "image_url": "https://storage.googleapis.com/falserverless/model_tests/wan/dragon-warrior.jpg",
  "prompt": "The white dragon warrior stands still, eyes full of determination and strength. The camera slowly moves closer or circles around the warrior, highlighting the powerful presence and heroic spirit of the character.",
  "num_frames": 81,
  "frames_per_second": 16,
  "resolution": "720p",
  "aspect_ratio": "auto",
  "num_inference_steps": 27,
  "enable_safety_checker": true,
  "enable_output_safety_checker": false,
  "enable_prompt_expansion": false,
  "acceleration": "regular",
  "guidance_scale": 3.5,
  "guidance_scale_2": 3.5,
  "shift": 5,
  "interpolator_model": "film",
  "num_interpolated_frames": 1,
  "adjust_fps_for_interpolation": true,
  "video_quality": "high",
  "video_write_mode": "balanced"
}


// OUTPUT
{
  "video": {
    "url": "https://storage.googleapis.com/falserverless/gallery/wan-i2v-turbo.mp4"
  },
  "prompt": "The white dragon warrior stands still, eyes full of determination and strength. The camera slowly moves closer or circles around the warrior, highlighting the powerful presence and heroic spirit of the character."
}
```

### [v2.2-i2v - Replicate](https://replicate.com/wan-video/wan-2.2-i2v-a14b?output=json)
``` json
// INPUT
{
  "image": "https://replicate.delivery/pbxt/NRZtrR3lUtABKDbQ1E70AdzjuBXhLd1WJ8EovYI9wPQRXcfl/gondola.jpg",
  "prompt": "Golden hour, soft lighting, warm colors, saturated colors, wide shot, left-heavy composition. A weathered gondolier stands in a flat-bottomed boat, propelling it forward with a long wooden pole through the flooded ruins of Venice. The decaying buildings on either side are cloaked in creeping vines and marked by rusted metalwork, their once-proud facades now crumbling into the water. The camera moves slowly forward and tilts left, revealing behind him the majestic remnants of the city bathed in the amber glow of the setting sun. Silhouettes of collapsed archways and broken domes rise against the golden skyline, while the still water reflects the warm hues of the sky and surrounding structures.",
  "go_fast": false,
  "num_frames": 81,
  "resolution": "480p",
  "sample_shift": 5,
  "sample_steps": 30,
  "frames_per_second": 16
}

// OUTPUT
{
  "completed_at": "2025-07-29T19:07:19.392221Z",
  "created_at": "2025-07-29T19:06:51.159000Z",
  "data_removed": false,
  "error": null,
  "id": "43a2hdchjxrma0crayabpygmdc",
  "input": {
    "image": "https://replicate.delivery/pbxt/NRZtrR3lUtABKDbQ1E70AdzjuBXhLd1WJ8EovYI9wPQRXcfl/gondola.jpg",
    "prompt": "Golden hour, soft lighting, warm colors, saturated colors, wide shot, left-heavy composition. A weathered gondolier stands in a flat-bottomed boat, propelling it forward with a long wooden pole through the flooded ruins of Venice. The decaying buildings on either side are cloaked in creeping vines and marked by rusted metalwork, their once-proud facades now crumbling into the water. The camera moves slowly forward and tilts left, revealing behind him the majestic remnants of the city bathed in the amber glow of the setting sun. Silhouettes of collapsed archways and broken domes rise against the golden skyline, while the still water reflects the warm hues of the sky and surrounding structures.",
    "num_frames": 81,
    "resolution": "480p",
    "sample_shift": 5,
    "sample_steps": 30,
    "frames_per_second": 16
  },
  "logs": "0%|          | 0/30 [00:00<?, ?it/s] ~ 100%|██████████| 30/30 [00:22<00:00,  1.34it/s]\nsaved video",
  "metrics": {
    "predict_time": 28.228746549,
    "total_time": 28.233221
  },
  "output": "https://replicate.delivery/xezq/IFCJhDosHSqqEhcKkcEBHsyqtZOkvnfNYfTTub8C9Ryn3sFVA/output.mp4",
  "started_at": "2025-07-29T19:06:51.163475Z",
  "status": "succeeded",
  "urls": {
    "stream": "https://stream.replicate.com/v1/files/bcwr-pq2f73b6c3ndi2mzbzjsh4xm3vf7oxg5uyxmb7qdmgm4qqi33whq",
    "get": "https://api.replicate.com/v1/predictions/43a2hdchjxrma0crayabpygmdc",
    "cancel": "https://api.replicate.com/v1/predictions/43a2hdchjxrma0crayabpygmdc/cancel"
  },
  "version": "hidden"
}
```

### Lora, Turbo 및 2.6 버전 등
- [Fal.ai - 2.2 Lora](https://fal.ai/models/fal-ai/wan/v2.2-a14b/image-to-video)
- [Fal.ai - 2.2 turbo](https://fal.ai/models/fal-ai/wan/v2.2-a14b/image-to-video/turbo)
- [Fal.ai - 2.6](https://fal.ai/models/wan/v2.6/image-to-video/playground)
<br>
<br>
- [Replicate - 2.5 i2v fast](https://replicate.com/wan-video/wan-2.5-i2v-fast?input=json)
- [Replicate - 2.5 i2v](https://replicate.com/wan-video/wan-2.5-i2v?input=json&output=json)
- [Replicate - 2.6 i2v](https://replicate.com/wan-video/wan-2.6-i2v?input=json&output=json)

## Other Video Gen API
- [OpenAI Platform](https://platform.openai.com/docs/guides/video-generation)
- [Veo video gen api](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/model-reference/veo-video-generation#curl_1)
- [Kling AI](https://app.klingai.com/global/dev/document-api/apiReference/model/imageToVideo)
- [Novita - base64 image input](https://blogs.novita.ai/transforming-images-with-ease-image-to-video-ai-api/)