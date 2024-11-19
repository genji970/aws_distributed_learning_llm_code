from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

def Arg():
    # 훈련 설정
    training_args = TrainingArguments(
        output_dir="./results",
        overwrite_output_dir=True,
        per_device_train_batch_size=1,
        num_train_epochs=1,
        logging_dir='./logs',
        logging_steps=5,
    )

def Training():
    # Trainer 설정 및 훈련 시작
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data,
    )

