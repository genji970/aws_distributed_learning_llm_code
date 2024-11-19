import os
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import warnings
import pandas as pd
from torchvision import transforms
import glob
from tqdm import tqdm
from urllib.request import urlopen
from PIL import Image
import faiss
from sentence_transformers import SentenceTransformer

from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import pandas as pd
from datasets import Dataset

from peft import LoraConfig, get_peft_model

# custome import
import data_load
import model_build
import Context_Embedding_Space
import train

# 분산학습 library
import torch.distributed as dist
# random seed 고정
import random

seed = 40
deterministic = True

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
if deterministic:
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
warnings.filterwarnings('ignore')



# gpu device 선언
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 인스턴스들
model_build_instance = model_build.Model_build()

if __name__ == "__main__":
    # SageMaker 환경 변수
    world_size = int(os.environ.get('WORLD_SIZE', 1))  # 총 프로세스 수
    rank = int(os.environ.get('RANK', 0))  # 현재 프로세스의 순번
    local_rank = int(os.environ.get('LOCAL_RANK', 0))  # 로컬 GPU 번호

    # PyTorch 분산 초기화
    dist.init_process_group(
        backend='nccl',  # GPU 사용 시 nccl
        init_method='env://',  # SageMaker 환경 변수를 통해 초기화
        world_size=world_size,
        rank=rank
    )

    data_load.data_variable_generating()
    model_build_instance.config()
    model_build_instance.inner_model_structure()

    # 데이터 병렬화
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    data_load.data_engineering()
    Context_Embedding_Space.Embedding_space(mapping)
    data_load.Combine()
    data_load.Tokenizer()
    train.Arg()
    train.Training()
    trainer.train()
