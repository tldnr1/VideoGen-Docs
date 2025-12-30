## TurboDiffusion Quant 결과

### 생성 환경
- image: pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel
- python: 3.11

### 사용된 모델 가중치 (Quant 모델)
- [0.5GB]  Wan-AI/Wan2.1-T2V-1.3B/resolve/main/Wan2.1_VAE.pth
- [11.4GB] Wan-AI/Wan2.1-T2V-1.3B/resolve/main/models_t5_umt5-xxl-enc-bf16.pth
- [14.5GB] TurboDiffusion/TurboWan2.2-I2V-A14B-720P/resolve/main/TurboWan2.2-I2V-A14B-high-720P-quant.pth
- [14.5GB] TurboDiffusion/TurboWan2.2-I2V-A14B-720P/resolve/main/TurboWan2.2-I2V-A14B-low-720P-quant.pth

### 사용한 코드 및 결과 (영상 별도 첨부)
``` python
python turbodiffusion/inference/wan2.2_i2v_infer.py \
    --model Wan2.2-A14B \
    --low_noise_model_path checkpoints/TurboWan2.2-I2V-A14B-low-720P-quant.pth \
    --high_noise_model_path checkpoints/TurboWan2.2-I2V-A14B-high-720P-quant.pth \
    --resolution 720p \
    --adaptive_resolution \
    --image_path assets/i2v_inputs/i2v_input_1.jpg \
    --prompt "A colorless, rugged, six-wheeled lunar rover—with exposed suspension arms, roll-cage framing, and broad low-gravity tires—glides into view from left to right, kicking up billowing plumes of moon dust that drift slowly in the vacuum. Astronauts in white spacesuits perform light, bouncing lunar strides as they hop aboard the rover’s open chassis. In the far distance, a VTOL lander with a vertical, thruster-based descent profile touches down silently on the gray surface. Above it all, vast aurora-like plasma ribbons ripple across the star-filled sky, casting shimmering green, blue, and purple light over the barren lunar plains, giving the entire scene an otherworldly, magical glow." \
    --num_samples 1 \
    --num_steps 4 \
    --quant_linear \
    --attention_type sagesla \
    --sla_topk 0.15 \
    --save_path output/generated_video1.mp4 \
    --ode
```

- sla_topk 값 0.10이 기본이며, 품질을 위해 0.15 권장 (테스트 결과 눈에 띄게 큰 차이는 없는 듯 함)
- 텍스트 이해는 영어가 더 뛰어난 모습을 보이고, 다른 모델들과 동일하게 세밀하고 연관되도록 프롬프트를 작성해야 반영이 잘 되는 모습을 보임


---

### 특이사항
- quant의 생성 결과가 unquantized 버전과 큰 차이가 없음 (두 버전 모두 turbodiffusion에서 제공)
- gpu-a100-80g-small 에서도 동작 가능함
    - 총 가중치 합이 80GB 이내
    - 또한, infer.py 코드 동작 방식이 high noise 작동 후 gpu에서 내리고, low noise로 교체하는 방식이라 최대 32GB를 넘기지 않음
- base image 교체로 인해 jupyter 접속 불가능 >> gradio에 포트를 열어서 사용함