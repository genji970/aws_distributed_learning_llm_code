from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

class Model_build():
    # LoRA 설정
    def config(self):
        lora_config = LoraConfig(
            r=32,  # Low-rank 업데이트 행렬 차원
            lora_alpha=16,  # 스케일링 팩터
            lora_dropout=0.1,  # 드롭아웃 비율
            target_modules=["q_proj"],  # QLoRA가 적용될 대상 모듈
        )
    def inner_model_structure(self):
        # pretrained_model 다운로드
        # 모델 및 토크나이저 로드
        model_name = "ibm-granite/granite-3.0-8b-instruct"
        base_model = AutoModelForCausalLM.from_pretrained(model_name)

        #기존 model freeze
        for param in base_model.parameters():
            param.requires_grad = False

        model = get_peft_model(base_model, lora_config)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        #model gpu로 옮기기
        model = model.to(device)