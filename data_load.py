import pandas as pd
import Context_Embedding_Space

def data_variable_generating():
    train_path = 'Your_S3_link'
    test_path = 'Your_S3_link'
    map_path = 'Your_S3_link'
    sub_path = 'Your_S3_link'

    # 데이터 읽기
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    map_data = pd.read_csv(map_path)
    sub_data = pd.read_csv(sub_path)

def data_engineering():
    # 제외할 열 이름
    exclude_columns = ['MisconceptionAId',
                       'MisconceptionBId', 'MisconceptionCId', 'MisconceptionDId']

    # 제외한 나머지 열 이름 리스트에 담기
    remaining_columns = [col for col in df_train.columns if col not in exclude_columns]

    # 만약 answer = None -> 0으로 교체
    df_train = df_train.fillna(0)

    # 각 행(row)에 대해 지정된 문장을 생성하고 리스트에 저장
    train_data = [
        {
            "instruction": "look at data structure and choose 25 misconceptionId in context by choosing 25 misconceptionname which are appropriate for data structure ,",
            "data structure": ",".join([f"{col} is {row[col]}" for col in df_train[remaining_columns]]),
            "label_structure": ",".join([f"{col} is {row[col]}" for col in df_train[exclude_columns]])
            }
        for _, row in df_train.iterrows()
    ]

    # 특수부호 제거
    train_data = [
        {key: value.replace("\n", "").replace("\\", "") if isinstance(value, str) else value
         for key, value in item.items()}
        for item in train_data
    ]

def Combine():
    # instruction과 data structure 결합
    combined_data = [
        {
            "combined_text": f"{item['instruction']} {item['data structure']}",
            "label": f"{item['label_structure']}"
        }
        for item in train_data
    ]

def Tokenizer():
    # embedding space 생성
    # encode 모델이 없는 경우 전용 encoding해주는 모델 사용
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    # input_ids, attention_mask, labels 생성
    tokenized_data = [
        {
            **tokenizer(
                item['combined_text'] + " " + " ".join(choose_k_misconceptions(item['combined_text'])),
                padding='max_length',  # 또는 'longest' 등 원하는 padding 방식 선택
                truncation=True,
                max_length=1024  # max_length는 필요에 맞게 조정
            ),
            'labels': tokenizer(
                item['label'],
                padding='max_length',  # 동일한 padding 방식을 적용
                truncation=True,
                max_length=128  # 동일한 max_length를 적용
            )['input_ids']
        }
        for item in combined_data
    ]
