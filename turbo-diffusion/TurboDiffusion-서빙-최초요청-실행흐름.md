# TurboDiffusion 서빙 모드 최초 요청 실행 흐름

## 개요

TurboDiffusion의 서빙 모드에서 최초 요청 시 실행되는 코드 흐름을 정리한 문서입니다. 서빙 모드는 모델을 한 번만 로드하고 메모리에 유지하여 여러 비디오 생성 요청을 처리할 수 있는 대화형 인터페이스를 제공합니다.

## 실행 흐름 개요

```
1. 진입점 → 2. 초기화 → 3. 모델 로딩 → 4. TUI 루프 시작 → 5. 최초 요청 처리
```

---

## 1단계: 진입점 (Entry Point)

### 방법 A: Python 모듈로 실행
```bash
python -m turbodiffusion.serve --mode i2v [args]
```

**코드 경로**: `turbodiffusion/serve/__main__.py`
```python
from .tui import main

if __name__ == "__main__":
    main()
```

### 방법 B: Inference 스크립트에 `--serve` 플래그
```bash
python turbodiffusion/inference/wan2.2_i2v_infer.py --serve [args]
```

**코드 경로**: `turbodiffusion/inference/wan2.2_i2v_infer.py`
```python
if args.serve:
    args.mode = "i2v"
    from serve.tui import main as serve_main
    serve_main(args)
    exit(0)
```

---

## 2단계: TUI 서버 초기화

**코드 경로**: `turbodiffusion/serve/tui.py::main()`

```python
def main(passed_args: argparse.Namespace = None):
    args = passed_args if passed_args is not None else parse_args()
    validate_args(args)
    
    console.print("[dim]Loading models...[/dim]")
    models = load_models(args)  # 모델 로딩 (최초 1회)
    
    try:
        run_tui(models, args)  # 대화형 루프 시작
    except KeyboardInterrupt:
        console.print("\n\n[dim]Interrupted. Goodbye![/dim]")
```

### 2-1. 인자 파싱 및 검증

**코드 경로**: `turbodiffusion/serve/arg_utils.py`

- `parse_args()`: 명령행 인자 파싱
- `validate_args()`: 인자 검증 및 모드별 기본값 설정
  - I2V 모드: `model=Wan2.2-A14B`, `resolution=720p`, `sigma_max=200`
  - 필수 인자 검증: `high_noise_model_path`, `low_noise_model_path`

---

## 3단계: 모델 로딩 (최초 1회)

**코드 경로**: `turbodiffusion/serve/pipeline.py::load_models()`

### 3-1. VAE 로딩
```python
tokenizer = Wan2pt1VAEInterface(vae_pth=args.vae_path)
```

### 3-2. I2V 모델 로딩
```python
def load_models_i2v(args):
    # High noise 모델 생성 및 CPU에 유지
    high_noise_model = create_model(dit_path=args.high_noise_model_path, args=args)
    high_noise_model.cpu().eval()
    
    # Low noise 모델 생성 및 CPU에 유지
    low_noise_model = create_model(dit_path=args.low_noise_model_path, args=args)
    low_noise_model.cpu().eval()
    
    return {"high_noise_model": high_noise_model, "low_noise_model": low_noise_model}
```

### 3-3. 모델 생성 과정

**코드 경로**: `turbodiffusion/inference/modify_model.py::create_model()`

```python
def create_model(dit_path: str, args: argparse.Namespace):
    # 1. 모델 아키텍처 선택 (Wan2.2-A14B)
    with torch.device("meta"):
        net = select_model(args.model)
    
    # 2. 체크포인트 로드
    state_dict = load_state_dict(dit_path)
    
    # 3. Attention 교체 (옵션: SLA/SageSLA)
    if args.attention_type in ['sla', 'sagesla']:
        net = replace_attention(net, attention_type=args.attention_type, sla_topk=args.sla_topk)
    
    # 4. Linear/Norm 레이어 최적화 (옵션)
    replace_linear_norm(net, replace_linear=args.quant_linear, replace_norm=not args.default_norm)
    
    # 5. 가중치 로드 및 GPU로 이동
    net.load_state_dict(state_dict, assign=True)
    net = net.to(tensor_kwargs["device"]).eval()
    
    return net
```

**특징**:
- High/Low noise 모델은 CPU에 유지 (메모리 절약)
- 필요 시 GPU로 이동하여 사용

---

## 4단계: TUI 대화형 루프 시작

**코드 경로**: `turbodiffusion/serve/tui.py::run_tui()`

```python
def run_tui(models: dict, args: argparse.Namespace):
    defaults = {param: getattr(args, param) for param in RUNTIME_PARAMS}
    print_header(args)  # 헤더 출력
    
    while True:
        # 프롬프트 입력 대기
        user_input = get_prompt_input(prompt_history)
        
        # 슬래시 명령어 처리 (/help, /set, /quit 등)
        if user_input.startswith("/"):
            handle_command(user_input, args, defaults)
            continue
        
        # I2V 모드: 이미지 경로 입력
        if args.mode == "i2v":
            image_path = get_path_input("image", last_image_path, must_exist=True)
        
        # 출력 경로 입력
        output_path = get_path_input("output", last_output_path)
        
        # 비디오 생성
        result_path = generate_i2v(models, args, prompt_text, image_path, output_path)
```

---

## 5단계: 최초 요청 처리

**코드 경로**: `turbodiffusion/serve/pipeline.py::generate_i2v()`

### 주요 처리 단계

1. **텍스트 임베딩 생성**
   ```python
   text_emb = get_umt5_embedding(checkpoint_path=args.text_encoder_path, prompts=prompt)
   ```

2. **이미지 전처리**
   - 이미지 로드 및 리사이즈
   - 정규화 (mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
   - Adaptive resolution 처리 (옵션)

3. **VAE 인코딩**
   ```python
   frames_to_encode = torch.cat([image_tensor.unsqueeze(2), 
                                  torch.zeros(1, 3, F-1, h, w)], dim=2)
   encoded_latents = tokenizer.encode(frames_to_encode)
   ```

4. **조건부 정보 구성**
   ```python
   condition = {
       "crossattn_emb": text_emb,  # 텍스트 임베딩
       "y_B_C_T_H_W": y  # 인코딩된 이미지 + 마스크
   }
   ```

5. **샘플링 루프**
   ```python
   # High noise 모델로 시작
   high_noise_model.cuda()
   net = high_noise_model
   
   for t_cur, t_next in t_steps:
       # Boundary에서 Low noise 모델로 전환
       if t_cur.item() < args.boundary and not switched:
           high_noise_model.cpu()
           low_noise_model.cuda()
           net = low_noise_model
           switched = True
       
       # Velocity 예측 및 업데이트
       v_pred = net(x_B_C_T_H_W=x, timesteps_B_T=timesteps, **condition)
       x = update_step(x, v_pred, t_cur, t_next)  # ODE 또는 SDE
   ```

6. **VAE 디코딩 및 저장**
   ```python
   video = tokenizer.decode(samples)
   save_image_or_video(video, output_path, fps=16)
   ```

---

## 주요 특징

### 메모리 관리
- **모델 로딩**: 서버 시작 시 한 번만 로드
- **High/Low 모델**: CPU에 유지, 필요 시 GPU로 이동
- **텍스트 인코더**: 요청마다 사용하되 메모리 해제하지 않음 (재사용)
- **VAE**: 서버 시작 시 로드되어 재사용

### 런타임 파라미터 조정
다음 파라미터는 `/set` 명령어로 런타임에 조정 가능:
- `num_steps`: 샘플링 스텝 (1-4)
- `num_samples`: 생성할 비디오 개수
- `num_frames`: 프레임 수
- `sigma_max`: 초기 sigma 값

### 대화형 명령어
- `/help`: 사용 가능한 명령어 표시
- `/show`: 현재 설정 표시
- `/set <param> <value>`: 런타임 파라미터 변경
- `/reset`: 런타임 파라미터를 기본값으로 리셋
- `/quit`: 서버 종료

---

## 실행 흐름 요약

```
진입점
  └─> tui.py::main()
      ├─> arg_utils.py::parse_args()        # 인자 파싱
      ├─> arg_utils.py::validate_args()    # 인자 검증
      └─> pipeline.py::load_models()        # 모델 로딩 (최초 1회)
          ├─> Wan2pt1VAEInterface()         # VAE 로드
          └─> load_models_i2v()
              ├─> create_model()            # High noise 모델
              └─> create_model()            # Low noise 모델
                  ├─> select_model()        # 아키텍처 선택
                  ├─> replace_attention()   # Attention 교체 (옵션)
                  ├─> replace_linear_norm() # 최적화 (옵션)
                  └─> load_state_dict()     # 가중치 로드

TUI 루프 시작
  └─> tui.py::run_tui()
      └─> while True:
          ├─> get_prompt_input()            # 프롬프트 입력
          ├─> get_path_input()              # 이미지 경로 (I2V)
          └─> generate_i2v()                # 비디오 생성
              ├─> get_umt5_embedding()      # 텍스트 임베딩
              ├─> 이미지 전처리
              ├─> tokenizer.encode()        # VAE 인코딩
              ├─> 샘플링 루프
              │   ├─> High noise 모델 (초기)
              │   └─> Low noise 모델 (boundary 이후)
              └─> tokenizer.decode()        # VAE 디코딩
```

---

## 참고사항

- 모델은 서버 시작 시 한 번만 로드되어 메모리에 유지
- High/Low noise 모델은 CPU에 유지, 필요 시 GPU로 이동하여 사용
- 텍스트 인코더(umT5)는 요청마다 사용하되 메모리 해제하지 않음
- VAE는 서버 시작 시 로드되어 재사용
- 대화형 루프로 여러 요청 처리 가능

