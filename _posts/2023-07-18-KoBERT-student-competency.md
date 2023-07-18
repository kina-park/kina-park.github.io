---
title: "KoBERT를 이용한 학생 역량 점수 분류"
layout: single
---

```python
from google.colab import drive
drive.mount('/content/drive')
```


# 라이브러리 설치 및 SKT KoBERT 파일 로드


```python
!pip install gluonnlp pandas tqdm
!pip install mxnet
!pip install sentencepiece==0.1.91
!pip install transformers==4.8.1
!pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
!pip install --no-cache-dir transformers sentencepiece
!pip install 'git+https://github.com/SKTBrain/KoBERT.git#egg=kobert_tokenizer&subdirectory=kobert_hf'
!pip install git+https://git@github.com/SKTBrain/KoBERT.git@master
```

# 라이브러리 불러오기


```python
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm, tqdm_notebook
import pandas as pd
from kobert import get_tokenizer
from kobert import get_pytorch_kobert_model

#transformers
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
from transformers import BertModel

#GPU 사용 시
device = torch.device("cuda:0")
```

# 오버샘플링 사용을 위한 추가 패키지 설치


```python
!pip install imblearn
from imblearn.over_sampling import SMOTE
```

###KoBERT 모델 선언


```python
from kobert import get_pytorch_kobert_model
from kobert_tokenizer import KoBERTTokenizer
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
bertmodel, vocab  = get_pytorch_kobert_model()
```


    Downloading:   0%|          | 0.00/371k [00:00<?, ?B/s]



    Downloading:   0%|          | 0.00/244 [00:00<?, ?B/s]



    Downloading:   0%|          | 0.00/432 [00:00<?, ?B/s]


    /content/.cache/kobert_v1.zip[██████████████████████████████████████████████████]
    /content/.cache/kobert_news_wiki_ko_cased-1087f8699e.spiece[██████████████████████████████████████████████████]


# 데이터셋 (추가적인) 전처리


```python
def df_newlabel(df):
  df2 = pd.melt(df, id_vars = df.columns[~df.columns.str.contains('competency')])

  #역량명 숫자로 변경
  df2['variable'] = [int(var[-1]) for var in df2['variable']]
  df2.rename(columns = {'variable': 'label', 'value': 'score'}, inplace = True)

  return df2
```


```python
def df_preprocessing(df, column_list : list):
  df1 = df.loc[:, column_list]

  #특수문자 및 불필요한 공백제거
  df1['student_assessment'] = df1['student_assessment'].str.replace("[^0-9ㄱ-ㅎㅏ-ㅣ가-힣 ]"," ")
  df1['student_assessment'] = df1['student_assessment'].str.replace(" +"," ")

  add_column = ['\t'.join(list(map(str, df1.loc[x, column_list].tolist()))) for x in range(len(df1))]
  score_list = df['score'] - 1

  data = [[add_column[i], score_list[i]] for i in range(len(df1))]

  return data
```


```python
df = pd.read_csv("/content/drive/MyDrive/PROJECT/1. data/df_3points_subject_concepts.csv")

#사용할 칼럼명 리스트로 지정
column_list = ['label', 'program_category', 'mission_category', 'student_assessment']

#사용할 역량번호 지정
target_number = 1

df2 = df_newlabel(df)
data = df_preprocessing(df2, column_list)
data[:5]
```

# 입력 데이터셋 토큰화 및 SMOTE 적용


```python
class BERTOverDataset(Dataset):
  def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, vocab, max_len, pad, pair):
    transform = nlp.data.BERTSentenceTransform(
        bert_tokenizer, max_seq_length = max_len, vocab = vocab, pad = pad, pair = pair)

    self.sentences = [transform([i[sent_idx]]) for i in dataset]
    self.labels = [np.int32(i[label_idx]) for i in dataset]

    self.sentences1 = [self.sentences[i][0] for i in range(len(self.sentences))]

    smote = SMOTE(random_state = 2022)
    self.sen1_over, self.labels_over = smote.fit_resample(self.sentences1, self.labels)

    self.sen1_over = [np.array(sen) for sen in self.sen1_over]
    self.sen2_over = [sum(sen != 1) for sen in self.sen1_over]
    self.sen3_over = [np.zeros(max_len) for _ in self.sen1_over]

    self.sen_over = [(self.sen1_over[i], np.array(self.sen2_over[i]), self.sen3_over[i]) for i in range(len(self.sen1_over))]

  def __getitem__(self, i):
    return (self.sen_over[i] + (self.labels_over[i], ))

  def __len__(self):
    return (len(self.labels_over))
```


```python
# Setting parameters
max_len = 256
batch_size = 40
warmup_ratio = 0.1
num_epochs = 50
max_grad_norm = 1
log_interval = 200
learning_rate =  5e-5
```

###Train & Test 데이터셋


```python
from sklearn.model_selection import train_test_split

dataset_train, dataset_test = train_test_split(data, test_size=0.2, shuffle=True, random_state=34)

tok=tokenizer.tokenize
data_train = BERTOverDataset(dataset_train, 0, 1, tok, vocab, max_len, True, False)
data_test = BERTOverDataset(dataset_test,0, 1, tok, vocab,  max_len, True, False)

train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, num_workers=5, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, num_workers=5, shuffle=True)
```

# KoBERT 학습 모델 생성


```python
class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes= 3,   # 다중분류 클래스 수만큼 조정#
                 dr_rate= None,
                 params= None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        self.hidden1 = nn.Linear(hidden_size , 256)
        self.relu1 = nn.ReLU()
        self.hidden2 = nn.Linear(256 , 128)
        self.relu2 = nn.ReLU()
        self.classifier = nn.Linear(128 , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device),return_dict=False)
        if self.dr_rate:
            out = self.dropout(pooler) # 768 dim

        hidden1_out = self.relu1(self.hidden1(out))
        hidden2_out = self.relu2(self.hidden2(hidden1_out))
        return self.classifier(hidden2_out)
```


```python
#BERT 모델 불러오기
model = BERTClassifier(bertmodel,  dr_rate=0.3).to(device)

#optimizer와 schedule 설정
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
loss_fn = nn.CrossEntropyLoss() # 다중분류를 위한 대표적인 loss func

t_total = len(train_dataloader) * num_epochs
warmup_step = int(t_total * warmup_ratio)

scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

#정확도 측정을 위한 함수 정의
def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc
```


```python
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
```


```python
# 돌리기 전에 GPU 최적화
import torch, gc
gc.collect()
torch.cuda.empty_cache()
```

# KoBERT 모델 학습


```python
train_history=[]
test_history=[]
loss_history=[]
for e in range(num_epochs):
    train_acc = 0.0
    test_acc = 0.0
    model.train()
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm_notebook(train_dataloader)):
        optimizer.zero_grad()
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)

        # print(label.shape,out.shape)
        loss = loss_fn(out, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()  # Update learning rate schedule
        train_acc += calc_accuracy(out, label)
        if batch_id % log_interval == 0:
            print("epoch {} batch id {} loss {} train acc {}".format(e+1, batch_id+1, loss.data.cpu().numpy(), train_acc / (batch_id+1)))
            train_history.append(train_acc / (batch_id+1))
            loss_history.append(loss.data.cpu().numpy())
    print("epoch {} train acc {}".format(e+1, train_acc / (batch_id+1)))
    #train_history.append(train_acc / (batch_id+1))

    model.eval()
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm_notebook(test_dataloader)):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        test_acc += calc_accuracy(out, label)

    print("epoch {} test acc {}".format(e+1, test_acc / (batch_id+1)))
    test_history.append(test_acc / (batch_id+1))
```

    <ipython-input-36-6a96544cc4d5>:8: TqdmDeprecationWarning: This function will be removed in tqdm==5.0.0
    Please use `tqdm.notebook.tqdm` instead of `tqdm.tqdm_notebook`
      for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm_notebook(train_dataloader)):



      0%|          | 0/1189 [00:00<?, ?it/s]


    epoch 1 batch id 1 loss 1.1017167568206787 train acc 0.225
    epoch 1 batch id 201 loss 1.0570727586746216 train acc 0.4608208955223881
    epoch 1 batch id 401 loss 0.9007512331008911 train acc 0.5832917705735662
    epoch 1 batch id 601 loss 0.7621575593948364 train acc 0.6314059900166389
    epoch 1 batch id 801 loss 0.6439598202705383 train acc 0.6557116104868914
    epoch 1 batch id 1001 loss 0.6400141716003418 train acc 0.6696303696303697
    epoch 1 train acc 0.6811916776803234


    <ipython-input-36-6a96544cc4d5>:31: TqdmDeprecationWarning: This function will be removed in tqdm==5.0.0
    Please use `tqdm.notebook.tqdm` instead of `tqdm.tqdm_notebook`
      for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm_notebook(test_dataloader)):



      0%|          | 0/304 [00:00<?, ?it/s]


    epoch 1 test acc 0.7100877192982449



      0%|          | 0/1189 [00:00<?, ?it/s]


    epoch 2 batch id 1 loss 0.5438047647476196 train acc 0.775
    epoch 2 batch id 201 loss 0.40222564339637756 train acc 0.7468905472636815
    epoch 2 batch id 401 loss 0.5673054456710815 train acc 0.7468204488778044
    epoch 2 batch id 601 loss 0.5383986830711365 train acc 0.7438435940099825
    epoch 2 batch id 801 loss 0.4466973841190338 train acc 0.743976279650436
    epoch 2 batch id 1001 loss 0.472039133310318 train acc 0.7458041958041953
    epoch 2 train acc 0.7460511033681764



      0%|          | 0/304 [00:00<?, ?it/s]


    epoch 2 test acc 0.734703947368421



      0%|          | 0/1189 [00:00<?, ?it/s]


    epoch 3 batch id 1 loss 0.7023351788520813 train acc 0.7
    epoch 3 batch id 201 loss 0.6127676963806152 train acc 0.757089552238806
    epoch 3 batch id 401 loss 0.48463940620422363 train acc 0.7575436408977557
    epoch 3 batch id 601 loss 0.3596824109554291 train acc 0.76206322795341
    epoch 3 batch id 801 loss 0.41794174909591675 train acc 0.7631086142322092
    epoch 3 batch id 1001 loss 0.30158546566963196 train acc 0.7640109890109883
    epoch 3 train acc 0.764602106612199



      0%|          | 0/304 [00:00<?, ?it/s]


    epoch 3 test acc 0.7171326754385963



      0%|          | 0/1189 [00:00<?, ?it/s]


    epoch 4 batch id 1 loss 0.3389321267604828 train acc 0.85
    epoch 4 batch id 201 loss 0.34675151109695435 train acc 0.7886815920398013
    epoch 4 batch id 401 loss 0.4170960783958435 train acc 0.7857855361596017
    epoch 4 batch id 601 loss 0.516179084777832 train acc 0.7829034941763727
    epoch 4 batch id 801 loss 0.6598507165908813 train acc 0.7822409488139831
    epoch 4 batch id 1001 loss 0.7842841148376465 train acc 0.7814185814185811
    epoch 4 train acc 0.7810615162801874



      0%|          | 0/304 [00:00<?, ?it/s]


    epoch 4 test acc 0.7342379385964911



      0%|          | 0/1189 [00:00<?, ?it/s]


    epoch 5 batch id 1 loss 0.4276917576789856 train acc 0.825
    epoch 5 batch id 201 loss 0.3399460017681122 train acc 0.8033582089552237
    epoch 5 batch id 401 loss 0.34584763646125793 train acc 0.8043017456359103
    epoch 5 batch id 601 loss 0.4994891285896301 train acc 0.8040765391014975
    epoch 5 batch id 801 loss 0.40357404947280884 train acc 0.8014669163545568
    epoch 5 batch id 1001 loss 0.49078670144081116 train acc 0.8003746253746251
    epoch 5 train acc 0.7999259081260769



      0%|          | 0/304 [00:00<?, ?it/s]


    epoch 5 test acc 0.7301809210526317



      0%|          | 0/1189 [00:00<?, ?it/s]


    epoch 6 batch id 1 loss 0.4435085654258728 train acc 0.8
    epoch 6 batch id 201 loss 0.4870987832546234 train acc 0.8310945273631842
    epoch 6 batch id 401 loss 0.4570940434932709 train acc 0.829551122194514
    epoch 6 batch id 601 loss 0.49191540479660034 train acc 0.826913477537438
    epoch 6 batch id 801 loss 0.21974137425422668 train acc 0.8240324594257185
    epoch 6 batch id 1001 loss 0.3808949291706085 train acc 0.8235764235764242
    epoch 6 train acc 0.8226090352036534



      0%|          | 0/304 [00:00<?, ?it/s]


    epoch 6 test acc 0.7404057017543861



      0%|          | 0/1189 [00:00<?, ?it/s]


    epoch 7 batch id 1 loss 0.34778133034706116 train acc 0.825
    epoch 7 batch id 201 loss 0.17207273840904236 train acc 0.8536069651741295
    epoch 7 batch id 401 loss 0.3060862421989441 train acc 0.8564214463840396
    epoch 7 batch id 601 loss 0.44226664304733276 train acc 0.8544509151414307
    epoch 7 batch id 801 loss 0.3388817608356476 train acc 0.8556179775280899
    epoch 7 batch id 1001 loss 0.461137056350708 train acc 0.8555444555444556
    epoch 7 train acc 0.8543123473106654



      0%|          | 0/304 [00:00<?, ?it/s]


    epoch 7 test acc 0.7333607456140347



      0%|          | 0/1189 [00:00<?, ?it/s]


    epoch 8 batch id 1 loss 0.23890528082847595 train acc 0.925
    epoch 8 batch id 201 loss 0.1997285932302475 train acc 0.8962686567164181
    epoch 8 batch id 401 loss 0.3741191029548645 train acc 0.894887780548629
    epoch 8 batch id 601 loss 0.31672394275665283 train acc 0.8910565723793675
    epoch 8 batch id 801 loss 0.46512752771377563 train acc 0.8912921348314597
    epoch 8 batch id 1001 loss 0.26555585861206055 train acc 0.8890359640359634
    epoch 8 train acc 0.8869718450879084



      0%|          | 0/304 [00:00<?, ?it/s]


    epoch 8 test acc 0.7123355263157894



      0%|          | 0/1189 [00:00<?, ?it/s]


    epoch 9 batch id 1 loss 0.19044199585914612 train acc 0.925
    epoch 9 batch id 201 loss 0.22186148166656494 train acc 0.9166666666666669
    epoch 9 batch id 401 loss 0.18050462007522583 train acc 0.9172693266832919
    epoch 9 batch id 601 loss 0.20054399967193604 train acc 0.9157237936772034
    epoch 9 batch id 801 loss 0.1873438060283661 train acc 0.9150436953807731
    epoch 9 batch id 1001 loss 0.2054639607667923 train acc 0.9145104895104886
    epoch 9 train acc 0.9127878569426086



      0%|          | 0/304 [00:00<?, ?it/s]


    epoch 9 test acc 0.7063048245614038



      0%|          | 0/1189 [00:00<?, ?it/s]


    epoch 10 batch id 1 loss 0.16318579018115997 train acc 0.95
    epoch 10 batch id 201 loss 0.2711047828197479 train acc 0.9348258706467658
    epoch 10 batch id 401 loss 0.17745158076286316 train acc 0.9301745635910226
    epoch 10 batch id 601 loss 0.09664060920476913 train acc 0.928577371048253
    epoch 10 batch id 801 loss 0.19888146221637726 train acc 0.9297128589263413
    epoch 10 batch id 1001 loss 0.0953061431646347 train acc 0.9292207792207793
    epoch 10 train acc 0.9279516600584711



      0%|          | 0/304 [00:00<?, ?it/s]


    epoch 10 test acc 0.7210800438596492



      0%|          | 0/1189 [00:00<?, ?it/s]


    epoch 11 batch id 1 loss 0.14634694159030914 train acc 0.975
    epoch 11 batch id 201 loss 0.06676371395587921 train acc 0.9456467661691536
    epoch 11 batch id 401 loss 0.10862831771373749 train acc 0.9437655860349127
    epoch 11 batch id 601 loss 0.16588763892650604 train acc 0.943801996672214
    epoch 11 batch id 801 loss 0.0630996897816658 train acc 0.9423220973782783
    epoch 11 batch id 1001 loss 0.1405034214258194 train acc 0.9405594405594422
    epoch 11 train acc 0.9396761984861236



      0%|          | 0/304 [00:00<?, ?it/s]


    epoch 11 test acc 0.7169682017543861



      0%|          | 0/1189 [00:00<?, ?it/s]


    epoch 12 batch id 1 loss 0.15361329913139343 train acc 0.975
    epoch 12 batch id 201 loss 0.016130510717630386 train acc 0.9527363184079589
    epoch 12 batch id 401 loss 0.37218526005744934 train acc 0.9518703241895261
    epoch 12 batch id 601 loss 0.08515758812427521 train acc 0.9502079866888536
    epoch 12 batch id 801 loss 0.030823102220892906 train acc 0.949375780274659
    epoch 12 batch id 1001 loss 0.25562167167663574 train acc 0.9487012987013014
    epoch 12 train acc 0.9486373102647309



      0%|          | 0/304 [00:00<?, ?it/s]


    epoch 12 test acc 0.7177357456140357



      0%|          | 0/1189 [00:00<?, ?it/s]


    epoch 13 batch id 1 loss 0.04118505120277405 train acc 1.0
    epoch 13 batch id 201 loss 0.03435292840003967 train acc 0.9600746268656702
    epoch 13 batch id 401 loss 0.10399508476257324 train acc 0.9592269326683296
    epoch 13 batch id 601 loss 0.08363153785467148 train acc 0.9562396006655588
    epoch 13 batch id 801 loss 0.06410279124975204 train acc 0.9559612983770325
    epoch 13 batch id 1001 loss 0.18932946026325226 train acc 0.9550949050949099
    epoch 13 train acc 0.9543333733829978



      0%|          | 0/304 [00:00<?, ?it/s]


    epoch 13 test acc 0.7116776315789481



      0%|          | 0/1189 [00:00<?, ?it/s]


    epoch 14 batch id 1 loss 0.013376357033848763 train acc 1.0
    epoch 14 batch id 201 loss 0.03915310651063919 train acc 0.9609452736318397
    epoch 14 batch id 401 loss 0.08313523232936859 train acc 0.9610349127182056
    epoch 14 batch id 601 loss 0.09374390542507172 train acc 0.9608153078203028
    epoch 14 batch id 801 loss 0.22838518023490906 train acc 0.9597690387016272
    epoch 14 batch id 1001 loss 0.15864714980125427 train acc 0.9594905094905146
    epoch 14 train acc 0.9593796307421241



      0%|          | 0/304 [00:00<?, ?it/s]


    epoch 14 test acc 0.7150219298245613



      0%|          | 0/1189 [00:00<?, ?it/s]


    epoch 15 batch id 1 loss 0.1485740840435028 train acc 0.925
    epoch 15 batch id 201 loss 0.008742861449718475 train acc 0.967910447761193
    epoch 15 batch id 401 loss 0.11650178581476212 train acc 0.9663965087281802
    epoch 15 batch id 601 loss 0.18028581142425537 train acc 0.9653494176372743
    epoch 15 batch id 801 loss 0.23527267575263977 train acc 0.965012484394513
    epoch 15 batch id 1001 loss 0.029835671186447144 train acc 0.9643856143856215
    epoch 15 train acc 0.9643858384396684



      0%|          | 0/304 [00:00<?, ?it/s]


    epoch 15 test acc 0.7099506578947371



      0%|          | 0/1189 [00:00<?, ?it/s]


    epoch 16 batch id 1 loss 0.05306854099035263 train acc 0.975
    epoch 16 batch id 201 loss 0.061275314539670944 train acc 0.9752487562189038
    epoch 16 batch id 401 loss 0.10735230147838593 train acc 0.9740648379052372
    epoch 16 batch id 601 loss 0.01816520467400551 train acc 0.9717138103161428
    epoch 16 batch id 801 loss 0.11671749502420425 train acc 0.9707553058676713
    epoch 16 batch id 1001 loss 0.29020771384239197 train acc 0.9700299700299766
    epoch 16 train acc 0.9691968040370105



      0%|          | 0/304 [00:00<?, ?it/s]


    epoch 16 test acc 0.7260416666666664



      0%|          | 0/1189 [00:00<?, ?it/s]


    epoch 17 batch id 1 loss 0.10715267807245255 train acc 0.95
    epoch 17 batch id 201 loss 0.3543071150779724 train acc 0.9783582089552224
    epoch 17 batch id 401 loss 0.0018963798647746444 train acc 0.9748129675810475
    epoch 17 batch id 601 loss 0.014463691040873528 train acc 0.9734193011647281
    epoch 17 batch id 801 loss 0.026752274483442307 train acc 0.9737203495630515
    epoch 17 batch id 1001 loss 0.3004460036754608 train acc 0.973451548451555
    epoch 17 train acc 0.9730445752733422



      0%|          | 0/304 [00:00<?, ?it/s]


    epoch 17 test acc 0.7318804824561402



      0%|          | 0/1189 [00:00<?, ?it/s]


    epoch 18 batch id 1 loss 0.24151721596717834 train acc 0.95
    epoch 18 batch id 201 loss 0.19718043506145477 train acc 0.9766169154228838
    epoch 18 batch id 401 loss 0.04403974860906601 train acc 0.9767456359102255
    epoch 18 batch id 601 loss 0.0802532508969307 train acc 0.9754159733777082
    epoch 18 batch id 801 loss 0.15635094046592712 train acc 0.9750312109862742
    epoch 18 batch id 1001 loss 0.0908583477139473 train acc 0.9738261738261824
    epoch 18 train acc 0.9739697224558495



      0%|          | 0/304 [00:00<?, ?it/s]


    epoch 18 test acc 0.7279057017543868



      0%|          | 0/1189 [00:00<?, ?it/s]


    epoch 19 batch id 1 loss 0.035527508705854416 train acc 1.0
    epoch 19 batch id 201 loss 0.005809093359857798 train acc 0.9793532338308444
    epoch 19 batch id 401 loss 0.2246818244457245 train acc 0.9800498753117217
    epoch 19 batch id 601 loss 0.14322158694267273 train acc 0.9799500831946785
    epoch 19 batch id 801 loss 0.08107805997133255 train acc 0.9799001248439501
    epoch 19 batch id 1001 loss 0.07561932504177094 train acc 0.979020979020986
    epoch 19 train acc 0.9786605390684467



      0%|          | 0/304 [00:00<?, ?it/s]


    epoch 19 test acc 0.7076480263157896



      0%|          | 0/1189 [00:00<?, ?it/s]


    epoch 20 batch id 1 loss 0.007348415441811085 train acc 1.0
    epoch 20 batch id 201 loss 0.0036182322073727846 train acc 0.9844527363184061
    epoch 20 batch id 401 loss 0.06050135940313339 train acc 0.9830423940149635
    epoch 20 batch id 601 loss 0.08994241803884506 train acc 0.9816971713810357
    epoch 20 batch id 801 loss 0.0199610386043787 train acc 0.9812734082397059
    epoch 20 batch id 1001 loss 0.09901021420955658 train acc 0.980669330669338
    epoch 20 train acc 0.980214465937766



      0%|          | 0/304 [00:00<?, ?it/s]


    epoch 20 test acc 0.7216557017543861



      0%|          | 0/1189 [00:00<?, ?it/s]


    epoch 21 batch id 1 loss 0.00239363219588995 train acc 1.0
    epoch 21 batch id 201 loss 0.027215102687478065 train acc 0.9858208955223865
    epoch 21 batch id 401 loss 0.005861632991582155 train acc 0.9837281795511231
    epoch 21 batch id 601 loss 0.026034126058220863 train acc 0.9827371048252952
    epoch 21 batch id 801 loss 0.009118080139160156 train acc 0.9824594257178587
    epoch 21 batch id 1001 loss 0.102273128926754 train acc 0.98231768231769
    epoch 21 train acc 0.9817934238455724



      0%|          | 0/304 [00:00<?, ?it/s]


    epoch 21 test acc 0.710800438596491



      0%|          | 0/1189 [00:00<?, ?it/s]


    epoch 22 batch id 1 loss 0.00372278387658298 train acc 1.0
    epoch 22 batch id 201 loss 0.20389561355113983 train acc 0.9880597014925363
    epoch 22 batch id 401 loss 0.000832450925372541 train acc 0.9865336658354127
    epoch 22 batch id 601 loss 0.0022994298487901688 train acc 0.9857321131447621
    epoch 22 batch id 801 loss 0.06965973228216171 train acc 0.9855493133583076
    epoch 22 batch id 1001 loss 0.07716672122478485 train acc 0.9853396603396674
    epoch 22 train acc 0.9848612279226262



      0%|          | 0/304 [00:00<?, ?it/s]


    epoch 22 test acc 0.7266995614035089



      0%|          | 0/1189 [00:00<?, ?it/s]


    epoch 23 batch id 1 loss 0.005060949362814426 train acc 1.0
    epoch 23 batch id 201 loss 0.004952941555529833 train acc 0.9884328358208943
    epoch 23 batch id 401 loss 0.0018322430551052094 train acc 0.9875311720698263
    epoch 23 batch id 601 loss 0.08975706994533539 train acc 0.9867304492512511
    epoch 23 batch id 801 loss 0.21216166019439697 train acc 0.9863607990012541
    epoch 23 batch id 1001 loss 0.12415780872106552 train acc 0.9868631368631431
    epoch 23 train acc 0.9864381833473528



      0%|          | 0/304 [00:00<?, ?it/s]


    epoch 23 test acc 0.7186677631578948



      0%|          | 0/1189 [00:00<?, ?it/s]


    epoch 24 batch id 1 loss 0.0034569837152957916 train acc 1.0
    epoch 24 batch id 201 loss 0.0010122025851160288 train acc 0.9910447761194019
    epoch 24 batch id 401 loss 0.0672229677438736 train acc 0.989650872817956
    epoch 24 batch id 601 loss 0.01720970682799816 train acc 0.9890183028286226
    epoch 24 batch id 801 loss 0.004236584063619375 train acc 0.9886704119850236
    epoch 24 batch id 1001 loss 0.005389336962252855 train acc 0.9888361638361693
    epoch 24 train acc 0.9888351555929372



      0%|          | 0/304 [00:00<?, ?it/s]


    epoch 24 test acc 0.7231359649122813



      0%|          | 0/1189 [00:00<?, ?it/s]


    epoch 25 batch id 1 loss 0.0035844899248331785 train acc 1.0
    epoch 25 batch id 201 loss 0.09134367853403091 train acc 0.9909203980099491
    epoch 25 batch id 401 loss 0.0057221706956624985 train acc 0.9890274314214477
    epoch 25 batch id 601 loss 0.0019282425055280328 train acc 0.9889351081530813
    epoch 25 batch id 801 loss 0.004067698959261179 train acc 0.989044943820229
    epoch 25 batch id 1001 loss 0.016406428068876266 train acc 0.9891358641358693
    epoch 25 train acc 0.989192598822541



      0%|          | 0/304 [00:00<?, ?it/s]


    epoch 25 test acc 0.7316063596491231



      0%|          | 0/1189 [00:00<?, ?it/s]


    epoch 26 batch id 1 loss 0.0008717264863662422 train acc 1.0
    epoch 26 batch id 201 loss 0.0006598674808628857 train acc 0.9916666666666658
    epoch 26 batch id 401 loss 0.002634765114635229 train acc 0.9913341645885299
    epoch 26 batch id 601 loss 0.0011452882317826152 train acc 0.9911397670549118
    epoch 26 batch id 801 loss 0.0009303910774178803 train acc 0.9911985018726637
    epoch 26 batch id 1001 loss 0.08342456817626953 train acc 0.990709290709296
    epoch 26 train acc 0.9909167367535758



      0%|          | 0/304 [00:00<?, ?it/s]


    epoch 26 test acc 0.7182017543859646



      0%|          | 0/1189 [00:00<?, ?it/s]


    epoch 27 batch id 1 loss 0.0011099178809672594 train acc 1.0
    epoch 27 batch id 201 loss 0.2523899972438812 train acc 0.9944029850746263
    epoch 27 batch id 401 loss 0.00336417811922729 train acc 0.9938279301745649
    epoch 27 batch id 601 loss 0.04362693801522255 train acc 0.9931364392678899
    epoch 27 batch id 801 loss 0.00018811605696100742 train acc 0.992634207240953
    epoch 27 batch id 1001 loss 0.002643485087901354 train acc 0.9922827172827222
    epoch 27 train acc 0.9920311185870494



      0%|          | 0/304 [00:00<?, ?it/s]


    epoch 27 test acc 0.7288925438596483



      0%|          | 0/1189 [00:00<?, ?it/s]


    epoch 28 batch id 1 loss 0.001131740165874362 train acc 1.0
    epoch 28 batch id 201 loss 0.0005285875522531569 train acc 0.9937810945273625
    epoch 28 batch id 401 loss 0.001098060398362577 train acc 0.9937032418952626
    epoch 28 batch id 601 loss 0.0153378676623106 train acc 0.9932612312812006
    epoch 28 batch id 801 loss 0.002647559391334653 train acc 0.9927590511860213
    epoch 28 batch id 1001 loss 0.0005728270043618977 train acc 0.992507492507497
    epoch 28 train acc 0.9928301093355778



      0%|          | 0/304 [00:00<?, ?it/s]


    epoch 28 test acc 0.7313596491228066



      0%|          | 0/1189 [00:00<?, ?it/s]


    epoch 29 batch id 1 loss 0.0014599306741729379 train acc 1.0
    epoch 29 batch id 201 loss 0.00048192794201895595 train acc 0.9946517412935318
    epoch 29 batch id 401 loss 0.00037156703183427453 train acc 0.9949501246882796
    epoch 29 batch id 601 loss 0.0019246317679062486 train acc 0.9940515806988377
    epoch 29 batch id 801 loss 0.00040261639514937997 train acc 0.9936641697877684
    epoch 29 batch id 1001 loss 0.002389008877798915 train acc 0.9936063936063972
    epoch 29 train acc 0.9929352396972243



      0%|          | 0/304 [00:00<?, ?it/s]


    epoch 29 test acc 0.7280427631578947



      0%|          | 0/1189 [00:00<?, ?it/s]


    epoch 30 batch id 1 loss 0.00034077424788847566 train acc 1.0
    epoch 30 batch id 201 loss 0.00038731301901862025 train acc 0.9935323383084571
    epoch 30 batch id 401 loss 0.08362691104412079 train acc 0.9941396508728192
    epoch 30 batch id 601 loss 0.02466263249516487 train acc 0.9940099833610672
    epoch 30 batch id 801 loss 0.00024603845668025315 train acc 0.9946004993757828
    epoch 30 batch id 1001 loss 0.13242633640766144 train acc 0.9943306693306725
    epoch 30 train acc 0.9944280908326336



      0%|          | 0/304 [00:00<?, ?it/s]


    epoch 30 test acc 0.7312225877192983



      0%|          | 0/1189 [00:00<?, ?it/s]


    epoch 31 batch id 1 loss 0.00024717350606806576 train acc 1.0
    epoch 31 batch id 201 loss 0.00014242733595892787 train acc 0.9962686567164172
    epoch 31 batch id 401 loss 0.0016416057478636503 train acc 0.9951995012468836
    epoch 31 batch id 601 loss 0.00046318318345583975 train acc 0.9948419301164744
    epoch 31 batch id 801 loss 0.11831218004226685 train acc 0.9951310861423243
    epoch 31 batch id 1001 loss 0.10196544975042343 train acc 0.9951548451548476
    epoch 31 train acc 0.9950378469301937



      0%|          | 0/304 [00:00<?, ?it/s]


    epoch 31 test acc 0.7304550438596491



      0%|          | 0/1189 [00:00<?, ?it/s]


    epoch 32 batch id 1 loss 0.0006567946984432638 train acc 1.0
    epoch 32 batch id 201 loss 0.0003974636201746762 train acc 0.9953980099502482
    epoch 32 batch id 401 loss 0.10692177712917328 train acc 0.9950124688279312
    epoch 32 batch id 601 loss 0.0005372531595639884 train acc 0.9950915141430968
    epoch 32 batch id 801 loss 0.00022055480803828686 train acc 0.9952247191011261
    epoch 32 batch id 1001 loss 0.00014373197336681187 train acc 0.9948551448551479
    epoch 32 train acc 0.9948275862068967



      0%|          | 0/304 [00:00<?, ?it/s]


    epoch 32 test acc 0.7243421052631577



      0%|          | 0/1189 [00:00<?, ?it/s]


    epoch 33 batch id 1 loss 0.000260232831351459 train acc 1.0
    epoch 33 batch id 201 loss 0.00017899609520100057 train acc 0.9967661691542286
    epoch 33 batch id 401 loss 0.19148144125938416 train acc 0.9964463840399014
    epoch 33 batch id 601 loss 0.0003354734508320689 train acc 0.9962978369384378
    epoch 33 batch id 801 loss 0.00304773123934865 train acc 0.9964419475655452
    epoch 33 batch id 1001 loss 0.0006572320125997066 train acc 0.9962787212787237
    epoch 33 train acc 0.99594196804037



      0%|          | 0/304 [00:00<?, ?it/s]


    epoch 33 test acc 0.7278782894736843



      0%|          | 0/1189 [00:00<?, ?it/s]


    epoch 34 batch id 1 loss 0.00011507681483635679 train acc 1.0
    epoch 34 batch id 201 loss 5.604793113889173e-05 train acc 0.9966417910447757
    epoch 34 batch id 401 loss 4.8301615606760606e-05 train acc 0.9971321695760602
    epoch 34 batch id 601 loss 0.0012939878506585956 train acc 0.9968386023294521
    epoch 34 batch id 801 loss 0.0004015875165350735 train acc 0.9967852684144836
    epoch 34 batch id 1001 loss 0.0001313415268668905 train acc 0.9966533466533487
    epoch 34 train acc 0.9966778805719093



      0%|          | 0/304 [00:00<?, ?it/s]


    epoch 34 test acc 0.7292489035087718



      0%|          | 0/1189 [00:00<?, ?it/s]


    epoch 35 batch id 1 loss 0.0002709450200200081 train acc 1.0
    epoch 35 batch id 201 loss 0.1422850638628006 train acc 0.9965174129353229
    epoch 35 batch id 401 loss 0.00012707019050139934 train acc 0.9966957605985045
    epoch 35 batch id 601 loss 0.0001624890574021265 train acc 0.9968386023294523
    epoch 35 batch id 801 loss 4.585255010169931e-05 train acc 0.9967852684144837
    epoch 35 batch id 1001 loss 4.836085281567648e-05 train acc 0.9967282717282738
    epoch 35 train acc 0.9963835155592932



      0%|          | 0/304 [00:00<?, ?it/s]


    epoch 35 test acc 0.7318804824561403



      0%|          | 0/1189 [00:00<?, ?it/s]


    epoch 36 batch id 1 loss 0.025365635752677917 train acc 0.975
    epoch 36 batch id 201 loss 7.451898272847757e-05 train acc 0.9975124378109449
    epoch 36 batch id 401 loss 0.0003887308412231505 train acc 0.9975062344139658
    epoch 36 batch id 601 loss 4.227882163831964e-05 train acc 0.9973377703826968
    epoch 36 batch id 801 loss 7.590462337248027e-05 train acc 0.997253433208491
    epoch 36 batch id 1001 loss 4.614747376763262e-05 train acc 0.997177822177824
    epoch 36 train acc 0.9970142977291838



      0%|          | 0/304 [00:00<?, ?it/s]


    epoch 36 test acc 0.7271929824561406



      0%|          | 0/1189 [00:00<?, ?it/s]


    epoch 37 batch id 1 loss 7.283810555236414e-05 train acc 1.0
    epoch 37 batch id 201 loss 0.00012422920553945005 train acc 0.9968905472636812
    epoch 37 batch id 401 loss 0.00011671053653117269 train acc 0.9973192019950128
    epoch 37 batch id 601 loss 4.974353942088783e-05 train acc 0.9969633943427632
    epoch 37 batch id 801 loss 3.679807559819892e-05 train acc 0.9971285892634221
    epoch 37 batch id 1001 loss 2.709832915570587e-05 train acc 0.9972527472527487
    epoch 37 train acc 0.9971825063078218



      0%|          | 0/304 [00:00<?, ?it/s]


    epoch 37 test acc 0.7306469298245614



      0%|          | 0/1189 [00:00<?, ?it/s]


    epoch 38 batch id 1 loss 3.752591510419734e-05 train acc 1.0
    epoch 38 batch id 201 loss 4.465406891540624e-05 train acc 0.9976368159203978
    epoch 38 batch id 401 loss 3.920613380614668e-05 train acc 0.9973815461346641
    epoch 38 batch id 601 loss 7.331536471610889e-05 train acc 0.9976289517470892
    epoch 38 batch id 801 loss 0.00010001605551224202 train acc 0.9974406991260939
    epoch 38 batch id 1001 loss 0.0007880543125793338 train acc 0.9974275724275741
    epoch 38 train acc 0.9974348191757786



      0%|          | 0/304 [00:00<?, ?it/s]


    epoch 38 test acc 0.7296326754385966



      0%|          | 0/1189 [00:00<?, ?it/s]


    epoch 39 batch id 1 loss 1.9502354916767217e-05 train acc 1.0
    epoch 39 batch id 201 loss 2.3284021153813228e-05 train acc 0.9980099502487559
    epoch 39 batch id 401 loss 7.266753527801484e-05 train acc 0.9979426433915216
    epoch 39 batch id 601 loss 2.2541860744240694e-05 train acc 0.9977537437604004
    epoch 39 batch id 801 loss 3.4416869311826304e-05 train acc 0.9978464419475668
    epoch 39 batch id 1001 loss 1.56996011355659e-05 train acc 0.9978771228771243
    epoch 39 train acc 0.9978763666947014



      0%|          | 0/304 [00:00<?, ?it/s]


    epoch 39 test acc 0.7294407894736841



      0%|          | 0/1189 [00:00<?, ?it/s]


    epoch 40 batch id 1 loss 6.421625585062429e-05 train acc 1.0
    epoch 40 batch id 201 loss 2.041674633801449e-05 train acc 0.997014925373134
    epoch 40 batch id 401 loss 1.4453937183134258e-05 train acc 0.9973192019950128
    epoch 40 batch id 601 loss 0.002224566647782922 train acc 0.9977953410981703
    epoch 40 batch id 801 loss 1.639973561395891e-05 train acc 0.9979088639201006
    epoch 40 batch id 1001 loss 1.2376780432532541e-05 train acc 0.9978021978021989
    epoch 40 train acc 0.997687132043734



      0%|          | 0/304 [00:00<?, ?it/s]


    epoch 40 test acc 0.7344846491228071



      0%|          | 0/1189 [00:00<?, ?it/s]


    epoch 41 batch id 1 loss 1.4918754459358752e-05 train acc 1.0
    epoch 41 batch id 201 loss 1.6441726984339766e-05 train acc 0.9975124378109449
    epoch 41 batch id 401 loss 5.483616405399516e-05 train acc 0.9976932668329183
    epoch 41 batch id 601 loss 9.810842129809316e-06 train acc 0.9976705490848595
    epoch 41 batch id 801 loss 0.00019151667947880924 train acc 0.9975967540574295
    epoch 41 batch id 1001 loss 2.382580714765936e-05 train acc 0.9976523476523491
    epoch 41 train acc 0.9976871320437344



      0%|          | 0/304 [00:00<?, ?it/s]


    epoch 41 test acc 0.7308114035087714



      0%|          | 0/1189 [00:00<?, ?it/s]


    epoch 42 batch id 1 loss 1.6110760043375194e-05 train acc 1.0
    epoch 42 batch id 201 loss 0.000879845698364079 train acc 0.9981343283582087
    epoch 42 batch id 401 loss 2.7050071366829798e-05 train acc 0.9980673316708232
    epoch 42 batch id 601 loss 4.681710561271757e-05 train acc 0.9980033277870223
    epoch 42 batch id 801 loss 1.0266778190270998e-05 train acc 0.9980337078651694
    epoch 42 batch id 1001 loss 2.808023418765515e-05 train acc 0.9980019980019992
    epoch 42 train acc 0.9980445752733392



      0%|          | 0/304 [00:00<?, ?it/s]


    epoch 42 test acc 0.7327850877192987



      0%|          | 0/1189 [00:00<?, ?it/s]


    epoch 43 batch id 1 loss 9.250545190297998e-06 train acc 1.0
    epoch 43 batch id 201 loss 1.5711368178017437e-05 train acc 0.9982587064676615
    epoch 43 batch id 401 loss 0.0018204068765044212 train acc 0.9980049875311725
    epoch 43 batch id 601 loss 1.0102868145622779e-05 train acc 0.9979201331114816
    epoch 43 batch id 801 loss 6.742840923834592e-05 train acc 0.9980649188514367
    epoch 43 batch id 1001 loss 1.1911854016943835e-05 train acc 0.9980269730269742
    epoch 43 train acc 0.9979394449116905



      0%|          | 0/304 [00:00<?, ?it/s]


    epoch 43 test acc 0.729194078947368



      0%|          | 0/1189 [00:00<?, ?it/s]


    epoch 44 batch id 1 loss 1.5777033695485443e-05 train acc 1.0
    epoch 44 batch id 201 loss 3.48812245647423e-05 train acc 0.9991293532338308
    epoch 44 batch id 401 loss 1.2460184734663926e-05 train acc 0.9985037406483795
    epoch 44 batch id 601 loss 0.015018999576568604 train acc 0.9983361064891855
    epoch 44 batch id 801 loss 1.2740047168335877e-05 train acc 0.998096129837704
    epoch 44 batch id 1001 loss 1.0257865142193623e-05 train acc 0.9981268731268744
    epoch 44 train acc 0.9981727341903966



      0%|          | 0/304 [00:00<?, ?it/s]


    epoch 44 test acc 0.7305646929824566



      0%|          | 0/1189 [00:00<?, ?it/s]


    epoch 45 batch id 1 loss 1.0421765182400122e-05 train acc 1.0
    epoch 45 batch id 201 loss 0.0002971302601508796 train acc 0.9987562189054725
    epoch 45 batch id 401 loss 1.4504359569400549e-05 train acc 0.9984413965087287
    epoch 45 batch id 601 loss 0.011165144853293896 train acc 0.9983361064891857
    epoch 45 batch id 801 loss 9.366763151774649e-06 train acc 0.9981897627965056
    epoch 45 batch id 1001 loss 0.02021697722375393 train acc 0.9981518481518495
    epoch 45 train acc 0.9982548359966366



      0%|          | 0/304 [00:00<?, ?it/s]


    epoch 45 test acc 0.7303728070175443



      0%|          | 0/1189 [00:00<?, ?it/s]


    epoch 46 batch id 1 loss 7.855830517655704e-06 train acc 1.0
    epoch 46 batch id 201 loss 9.885338840831537e-06 train acc 0.9986318407960197
    epoch 46 batch id 401 loss 6.261425824050093e-06 train acc 0.998379052369078
    epoch 46 batch id 601 loss 1.3118714377924334e-05 train acc 0.998252911813645
    epoch 46 batch id 801 loss 0.014059795066714287 train acc 0.9983458177278413
    epoch 46 batch id 1001 loss 1.019823866954539e-05 train acc 0.9984265734265745
    epoch 46 train acc 0.9983809924306138



      0%|          | 0/304 [00:00<?, ?it/s]


    epoch 46 test acc 0.7302083333333329



      0%|          | 0/1189 [00:00<?, ?it/s]


    epoch 47 batch id 1 loss 1.6118605344672687e-05 train acc 1.0
    epoch 47 batch id 201 loss 8.460790922981687e-06 train acc 0.9986318407960197
    epoch 47 batch id 401 loss 7.381964678643271e-06 train acc 0.9980049875311725
    epoch 47 batch id 601 loss 6.7621126618178096e-06 train acc 0.9980865224625632
    epoch 47 batch id 801 loss 1.0454550647409633e-05 train acc 0.9982521847690395
    epoch 47 batch id 1001 loss 6.5385802372475155e-06 train acc 0.9982767232767242
    epoch 47 train acc 0.9983389402859547



      0%|          | 0/304 [00:00<?, ?it/s]


    epoch 47 test acc 0.7311951754385966



      0%|          | 0/1189 [00:00<?, ?it/s]


    epoch 48 batch id 1 loss 9.107485311687924e-06 train acc 1.0
    epoch 48 batch id 201 loss 8.32965724839596e-06 train acc 0.9986318407960196
    epoch 48 batch id 401 loss 7.6084688771516085e-06 train acc 0.9985037406483793
    epoch 48 batch id 601 loss 7.3134447120537516e-06 train acc 0.9983361064891855
    epoch 48 batch id 801 loss 8.08531422080705e-06 train acc 0.9982521847690398
    epoch 48 batch id 1001 loss 1.172401516669197e-05 train acc 0.9984265734265744
    epoch 48 train acc 0.9984861227922627



      0%|          | 0/304 [00:00<?, ?it/s]


    epoch 48 test acc 0.7309758771929823



      0%|          | 0/1189 [00:00<?, ?it/s]


    epoch 49 batch id 1 loss 8.937640814110637e-06 train acc 1.0
    epoch 49 batch id 201 loss 1.626415723876562e-05 train acc 0.9986318407960199
    epoch 49 batch id 401 loss 6.771048447262729e-06 train acc 0.9986284289276812
    epoch 49 batch id 601 loss 6.9111147240619175e-06 train acc 0.9986688851913484
    epoch 49 batch id 801 loss 0.016338784247636795 train acc 0.9987827715355813
    epoch 49 batch id 1001 loss 1.089848592528142e-05 train acc 0.9988011988011996
    epoch 49 train acc 0.9985912531539107



      0%|          | 0/304 [00:00<?, ?it/s]


    epoch 49 test acc 0.7307839912280703



      0%|          | 0/1189 [00:00<?, ?it/s]


    epoch 50 batch id 1 loss 6.87535566612496e-06 train acc 1.0
    epoch 50 batch id 201 loss 7.331263077503536e-06 train acc 0.9978855721393032
    epoch 50 batch id 401 loss 0.0003719524247571826 train acc 0.9981296758104743
    epoch 50 batch id 601 loss 8.481675649818499e-06 train acc 0.9982945091514152
    epoch 50 batch id 801 loss 8.160841389326379e-05 train acc 0.9983770287141083
    epoch 50 batch id 1001 loss 6.976682925596833e-06 train acc 0.9983266733266742
    epoch 50 train acc 0.998402018502944



      0%|          | 0/304 [00:00<?, ?it/s]


    epoch 50 test acc 0.7309484649122806


###train_history, test_history, loss_history 확인


```python
train_history
```




    [0.225,
     0.4608208955223881,
     0.5832917705735662,
     0.6314059900166389,
     0.6557116104868914,
     0.6696303696303697,
     0.775,
     0.7468905472636815,
     0.7468204488778044,
     0.7438435940099825,
     0.743976279650436,
     0.7458041958041953,
     0.7,
     0.757089552238806,
     0.7575436408977557,
     0.76206322795341,
     0.7631086142322092,
     0.7640109890109883,
     0.85,
     0.7886815920398013,
     0.7857855361596017,
     0.7829034941763727,
     0.7822409488139831,
     0.7814185814185811,
     0.825,
     0.8033582089552237,
     0.8043017456359103,
     0.8040765391014975,
     0.8014669163545568,
     0.8003746253746251,
     0.8,
     0.8310945273631842,
     0.829551122194514,
     0.826913477537438,
     0.8240324594257185,
     0.8235764235764242,
     0.825,
     0.8536069651741295,
     0.8564214463840396,
     0.8544509151414307,
     0.8556179775280899,
     0.8555444555444556,
     0.925,
     0.8962686567164181,
     0.894887780548629,
     0.8910565723793675,
     0.8912921348314597,
     0.8890359640359634,
     0.925,
     0.9166666666666669,
     0.9172693266832919,
     0.9157237936772034,
     0.9150436953807731,
     0.9145104895104886,
     0.95,
     0.9348258706467658,
     0.9301745635910226,
     0.928577371048253,
     0.9297128589263413,
     0.9292207792207793,
     0.975,
     0.9456467661691536,
     0.9437655860349127,
     0.943801996672214,
     0.9423220973782783,
     0.9405594405594422,
     0.975,
     0.9527363184079589,
     0.9518703241895261,
     0.9502079866888536,
     0.949375780274659,
     0.9487012987013014,
     1.0,
     0.9600746268656702,
     0.9592269326683296,
     0.9562396006655588,
     0.9559612983770325,
     0.9550949050949099,
     1.0,
     0.9609452736318397,
     0.9610349127182056,
     0.9608153078203028,
     0.9597690387016272,
     0.9594905094905146,
     0.925,
     0.967910447761193,
     0.9663965087281802,
     0.9653494176372743,
     0.965012484394513,
     0.9643856143856215,
     0.975,
     0.9752487562189038,
     0.9740648379052372,
     0.9717138103161428,
     0.9707553058676713,
     0.9700299700299766,
     0.95,
     0.9783582089552224,
     0.9748129675810475,
     0.9734193011647281,
     0.9737203495630515,
     0.973451548451555,
     0.95,
     0.9766169154228838,
     0.9767456359102255,
     0.9754159733777082,
     0.9750312109862742,
     0.9738261738261824,
     1.0,
     0.9793532338308444,
     0.9800498753117217,
     0.9799500831946785,
     0.9799001248439501,
     0.979020979020986,
     1.0,
     0.9844527363184061,
     0.9830423940149635,
     0.9816971713810357,
     0.9812734082397059,
     0.980669330669338,
     1.0,
     0.9858208955223865,
     0.9837281795511231,
     0.9827371048252952,
     0.9824594257178587,
     0.98231768231769,
     1.0,
     0.9880597014925363,
     0.9865336658354127,
     0.9857321131447621,
     0.9855493133583076,
     0.9853396603396674,
     1.0,
     0.9884328358208943,
     0.9875311720698263,
     0.9867304492512511,
     0.9863607990012541,
     0.9868631368631431,
     1.0,
     0.9910447761194019,
     0.989650872817956,
     0.9890183028286226,
     0.9886704119850236,
     0.9888361638361693,
     1.0,
     0.9909203980099491,
     0.9890274314214477,
     0.9889351081530813,
     0.989044943820229,
     0.9891358641358693,
     1.0,
     0.9916666666666658,
     0.9913341645885299,
     0.9911397670549118,
     0.9911985018726637,
     0.990709290709296,
     1.0,
     0.9944029850746263,
     0.9938279301745649,
     0.9931364392678899,
     0.992634207240953,
     0.9922827172827222,
     1.0,
     0.9937810945273625,
     0.9937032418952626,
     0.9932612312812006,
     0.9927590511860213,
     0.992507492507497,
     1.0,
     0.9946517412935318,
     0.9949501246882796,
     0.9940515806988377,
     0.9936641697877684,
     0.9936063936063972,
     1.0,
     0.9935323383084571,
     0.9941396508728192,
     0.9940099833610672,
     0.9946004993757828,
     0.9943306693306725,
     1.0,
     0.9962686567164172,
     0.9951995012468836,
     0.9948419301164744,
     0.9951310861423243,
     0.9951548451548476,
     1.0,
     0.9953980099502482,
     0.9950124688279312,
     0.9950915141430968,
     0.9952247191011261,
     0.9948551448551479,
     1.0,
     0.9967661691542286,
     0.9964463840399014,
     0.9962978369384378,
     0.9964419475655452,
     0.9962787212787237,
     1.0,
     0.9966417910447757,
     0.9971321695760602,
     0.9968386023294521,
     0.9967852684144836,
     0.9966533466533487,
     1.0,
     0.9965174129353229,
     0.9966957605985045,
     0.9968386023294523,
     0.9967852684144837,
     0.9967282717282738,
     0.975,
     0.9975124378109449,
     0.9975062344139658,
     0.9973377703826968,
     0.997253433208491,
     0.997177822177824,
     1.0,
     0.9968905472636812,
     0.9973192019950128,
     0.9969633943427632,
     0.9971285892634221,
     0.9972527472527487,
     1.0,
     0.9976368159203978,
     0.9973815461346641,
     0.9976289517470892,
     0.9974406991260939,
     0.9974275724275741,
     1.0,
     0.9980099502487559,
     0.9979426433915216,
     0.9977537437604004,
     0.9978464419475668,
     0.9978771228771243,
     1.0,
     0.997014925373134,
     0.9973192019950128,
     0.9977953410981703,
     0.9979088639201006,
     0.9978021978021989,
     1.0,
     0.9975124378109449,
     0.9976932668329183,
     0.9976705490848595,
     0.9975967540574295,
     0.9976523476523491,
     1.0,
     0.9981343283582087,
     0.9980673316708232,
     0.9980033277870223,
     0.9980337078651694,
     0.9980019980019992,
     1.0,
     0.9982587064676615,
     0.9980049875311725,
     0.9979201331114816,
     0.9980649188514367,
     0.9980269730269742,
     1.0,
     0.9991293532338308,
     0.9985037406483795,
     0.9983361064891855,
     0.998096129837704,
     0.9981268731268744,
     1.0,
     0.9987562189054725,
     0.9984413965087287,
     0.9983361064891857,
     0.9981897627965056,
     0.9981518481518495,
     1.0,
     0.9986318407960197,
     0.998379052369078,
     0.998252911813645,
     0.9983458177278413,
     0.9984265734265745,
     1.0,
     0.9986318407960197,
     0.9980049875311725,
     0.9980865224625632,
     0.9982521847690395,
     0.9982767232767242,
     1.0,
     0.9986318407960196,
     0.9985037406483793,
     0.9983361064891855,
     0.9982521847690398,
     0.9984265734265744,
     1.0,
     0.9986318407960199,
     0.9986284289276812,
     0.9986688851913484,
     0.9987827715355813,
     0.9988011988011996,
     1.0,
     0.9978855721393032,
     0.9981296758104743,
     0.9982945091514152,
     0.9983770287141083,
     0.9983266733266742]




```python
test_history
```




    [0.7100877192982449,
     0.734703947368421,
     0.7171326754385963,
     0.7342379385964911,
     0.7301809210526317,
     0.7404057017543861,
     0.7333607456140347,
     0.7123355263157894,
     0.7063048245614038,
     0.7210800438596492,
     0.7169682017543861,
     0.7177357456140357,
     0.7116776315789481,
     0.7150219298245613,
     0.7099506578947371,
     0.7260416666666664,
     0.7318804824561402,
     0.7279057017543868,
     0.7076480263157896,
     0.7216557017543861,
     0.710800438596491,
     0.7266995614035089,
     0.7186677631578948,
     0.7231359649122813,
     0.7316063596491231,
     0.7182017543859646,
     0.7288925438596483,
     0.7313596491228066,
     0.7280427631578947,
     0.7312225877192983,
     0.7304550438596491,
     0.7243421052631577,
     0.7278782894736843,
     0.7292489035087718,
     0.7318804824561403,
     0.7271929824561406,
     0.7306469298245614,
     0.7296326754385966,
     0.7294407894736841,
     0.7344846491228071,
     0.7308114035087714,
     0.7327850877192987,
     0.729194078947368,
     0.7305646929824566,
     0.7303728070175443,
     0.7302083333333329,
     0.7311951754385966,
     0.7309758771929823,
     0.7307839912280703,
     0.7309484649122806]




```python
loss_history
```




    [array(1.1017168, dtype=float32),
     array(1.0570728, dtype=float32),
     array(0.90075123, dtype=float32),
     array(0.76215756, dtype=float32),
     array(0.6439598, dtype=float32),
     array(0.6400142, dtype=float32),
     array(0.54380476, dtype=float32),
     array(0.40222564, dtype=float32),
     array(0.56730545, dtype=float32),
     array(0.5383987, dtype=float32),
     array(0.44669738, dtype=float32),
     array(0.47203913, dtype=float32),
     array(0.7023352, dtype=float32),
     array(0.6127677, dtype=float32),
     array(0.4846394, dtype=float32),
     array(0.3596824, dtype=float32),
     array(0.41794175, dtype=float32),
     array(0.30158547, dtype=float32),
     array(0.33893213, dtype=float32),
     array(0.3467515, dtype=float32),
     array(0.41709608, dtype=float32),
     array(0.5161791, dtype=float32),
     array(0.6598507, dtype=float32),
     array(0.7842841, dtype=float32),
     array(0.42769176, dtype=float32),
     array(0.339946, dtype=float32),
     array(0.34584764, dtype=float32),
     array(0.49948913, dtype=float32),
     array(0.40357405, dtype=float32),
     array(0.4907867, dtype=float32),
     array(0.44350857, dtype=float32),
     array(0.48709878, dtype=float32),
     array(0.45709404, dtype=float32),
     array(0.4919154, dtype=float32),
     array(0.21974137, dtype=float32),
     array(0.38089493, dtype=float32),
     array(0.34778133, dtype=float32),
     array(0.17207274, dtype=float32),
     array(0.30608624, dtype=float32),
     array(0.44226664, dtype=float32),
     array(0.33888176, dtype=float32),
     array(0.46113706, dtype=float32),
     array(0.23890528, dtype=float32),
     array(0.1997286, dtype=float32),
     array(0.3741191, dtype=float32),
     array(0.31672394, dtype=float32),
     array(0.46512753, dtype=float32),
     array(0.26555586, dtype=float32),
     array(0.190442, dtype=float32),
     array(0.22186148, dtype=float32),
     array(0.18050462, dtype=float32),
     array(0.200544, dtype=float32),
     array(0.1873438, dtype=float32),
     array(0.20546396, dtype=float32),
     array(0.16318579, dtype=float32),
     array(0.27110478, dtype=float32),
     array(0.17745158, dtype=float32),
     array(0.09664061, dtype=float32),
     array(0.19888146, dtype=float32),
     array(0.09530614, dtype=float32),
     array(0.14634694, dtype=float32),
     array(0.06676371, dtype=float32),
     array(0.10862832, dtype=float32),
     array(0.16588764, dtype=float32),
     array(0.06309969, dtype=float32),
     array(0.14050342, dtype=float32),
     array(0.1536133, dtype=float32),
     array(0.01613051, dtype=float32),
     array(0.37218526, dtype=float32),
     array(0.08515759, dtype=float32),
     array(0.0308231, dtype=float32),
     array(0.25562167, dtype=float32),
     array(0.04118505, dtype=float32),
     array(0.03435293, dtype=float32),
     array(0.10399508, dtype=float32),
     array(0.08363154, dtype=float32),
     array(0.06410279, dtype=float32),
     array(0.18932946, dtype=float32),
     array(0.01337636, dtype=float32),
     array(0.03915311, dtype=float32),
     array(0.08313523, dtype=float32),
     array(0.09374391, dtype=float32),
     array(0.22838518, dtype=float32),
     array(0.15864715, dtype=float32),
     array(0.14857408, dtype=float32),
     array(0.00874286, dtype=float32),
     array(0.11650179, dtype=float32),
     array(0.18028581, dtype=float32),
     array(0.23527268, dtype=float32),
     array(0.02983567, dtype=float32),
     array(0.05306854, dtype=float32),
     array(0.06127531, dtype=float32),
     array(0.1073523, dtype=float32),
     array(0.0181652, dtype=float32),
     array(0.1167175, dtype=float32),
     array(0.2902077, dtype=float32),
     array(0.10715268, dtype=float32),
     array(0.35430712, dtype=float32),
     array(0.00189638, dtype=float32),
     array(0.01446369, dtype=float32),
     array(0.02675227, dtype=float32),
     array(0.300446, dtype=float32),
     array(0.24151722, dtype=float32),
     array(0.19718044, dtype=float32),
     array(0.04403975, dtype=float32),
     array(0.08025325, dtype=float32),
     array(0.15635094, dtype=float32),
     array(0.09085835, dtype=float32),
     array(0.03552751, dtype=float32),
     array(0.00580909, dtype=float32),
     array(0.22468182, dtype=float32),
     array(0.14322159, dtype=float32),
     array(0.08107806, dtype=float32),
     array(0.07561933, dtype=float32),
     array(0.00734842, dtype=float32),
     array(0.00361823, dtype=float32),
     array(0.06050136, dtype=float32),
     array(0.08994242, dtype=float32),
     array(0.01996104, dtype=float32),
     array(0.09901021, dtype=float32),
     array(0.00239363, dtype=float32),
     array(0.0272151, dtype=float32),
     array(0.00586163, dtype=float32),
     array(0.02603413, dtype=float32),
     array(0.00911808, dtype=float32),
     array(0.10227313, dtype=float32),
     array(0.00372278, dtype=float32),
     array(0.20389561, dtype=float32),
     array(0.00083245, dtype=float32),
     array(0.00229943, dtype=float32),
     array(0.06965973, dtype=float32),
     array(0.07716672, dtype=float32),
     array(0.00506095, dtype=float32),
     array(0.00495294, dtype=float32),
     array(0.00183224, dtype=float32),
     array(0.08975707, dtype=float32),
     array(0.21216166, dtype=float32),
     array(0.12415781, dtype=float32),
     array(0.00345698, dtype=float32),
     array(0.0010122, dtype=float32),
     array(0.06722297, dtype=float32),
     array(0.01720971, dtype=float32),
     array(0.00423658, dtype=float32),
     array(0.00538934, dtype=float32),
     array(0.00358449, dtype=float32),
     array(0.09134368, dtype=float32),
     array(0.00572217, dtype=float32),
     array(0.00192824, dtype=float32),
     array(0.0040677, dtype=float32),
     array(0.01640643, dtype=float32),
     array(0.00087173, dtype=float32),
     array(0.00065987, dtype=float32),
     array(0.00263477, dtype=float32),
     array(0.00114529, dtype=float32),
     array(0.00093039, dtype=float32),
     array(0.08342457, dtype=float32),
     array(0.00110992, dtype=float32),
     array(0.25239, dtype=float32),
     array(0.00336418, dtype=float32),
     array(0.04362694, dtype=float32),
     array(0.00018812, dtype=float32),
     array(0.00264349, dtype=float32),
     array(0.00113174, dtype=float32),
     array(0.00052859, dtype=float32),
     array(0.00109806, dtype=float32),
     array(0.01533787, dtype=float32),
     array(0.00264756, dtype=float32),
     array(0.00057283, dtype=float32),
     array(0.00145993, dtype=float32),
     array(0.00048193, dtype=float32),
     array(0.00037157, dtype=float32),
     array(0.00192463, dtype=float32),
     array(0.00040262, dtype=float32),
     array(0.00238901, dtype=float32),
     array(0.00034077, dtype=float32),
     array(0.00038731, dtype=float32),
     array(0.08362691, dtype=float32),
     array(0.02466263, dtype=float32),
     array(0.00024604, dtype=float32),
     array(0.13242634, dtype=float32),
     array(0.00024717, dtype=float32),
     array(0.00014243, dtype=float32),
     array(0.00164161, dtype=float32),
     array(0.00046318, dtype=float32),
     array(0.11831218, dtype=float32),
     array(0.10196545, dtype=float32),
     array(0.00065679, dtype=float32),
     array(0.00039746, dtype=float32),
     array(0.10692178, dtype=float32),
     array(0.00053725, dtype=float32),
     array(0.00022055, dtype=float32),
     array(0.00014373, dtype=float32),
     array(0.00026023, dtype=float32),
     array(0.000179, dtype=float32),
     array(0.19148144, dtype=float32),
     array(0.00033547, dtype=float32),
     array(0.00304773, dtype=float32),
     array(0.00065723, dtype=float32),
     array(0.00011508, dtype=float32),
     array(5.604793e-05, dtype=float32),
     array(4.8301616e-05, dtype=float32),
     array(0.00129399, dtype=float32),
     array(0.00040159, dtype=float32),
     array(0.00013134, dtype=float32),
     array(0.00027095, dtype=float32),
     array(0.14228506, dtype=float32),
     array(0.00012707, dtype=float32),
     array(0.00016249, dtype=float32),
     array(4.585255e-05, dtype=float32),
     array(4.8360853e-05, dtype=float32),
     array(0.02536564, dtype=float32),
     array(7.451898e-05, dtype=float32),
     array(0.00038873, dtype=float32),
     array(4.227882e-05, dtype=float32),
     array(7.590462e-05, dtype=float32),
     array(4.6147474e-05, dtype=float32),
     array(7.2838106e-05, dtype=float32),
     array(0.00012423, dtype=float32),
     array(0.00011671, dtype=float32),
     array(4.974354e-05, dtype=float32),
     array(3.6798076e-05, dtype=float32),
     array(2.709833e-05, dtype=float32),
     array(3.7525915e-05, dtype=float32),
     array(4.465407e-05, dtype=float32),
     array(3.9206134e-05, dtype=float32),
     array(7.3315365e-05, dtype=float32),
     array(0.00010002, dtype=float32),
     array(0.00078805, dtype=float32),
     array(1.9502355e-05, dtype=float32),
     array(2.3284021e-05, dtype=float32),
     array(7.2667535e-05, dtype=float32),
     array(2.254186e-05, dtype=float32),
     array(3.441687e-05, dtype=float32),
     array(1.5699601e-05, dtype=float32),
     array(6.4216256e-05, dtype=float32),
     array(2.0416746e-05, dtype=float32),
     array(1.4453937e-05, dtype=float32),
     array(0.00222457, dtype=float32),
     array(1.6399736e-05, dtype=float32),
     array(1.237678e-05, dtype=float32),
     array(1.49187545e-05, dtype=float32),
     array(1.6441727e-05, dtype=float32),
     array(5.4836164e-05, dtype=float32),
     array(9.810842e-06, dtype=float32),
     array(0.00019152, dtype=float32),
     array(2.3825807e-05, dtype=float32),
     array(1.611076e-05, dtype=float32),
     array(0.00087985, dtype=float32),
     array(2.7050071e-05, dtype=float32),
     array(4.6817106e-05, dtype=float32),
     array(1.0266778e-05, dtype=float32),
     array(2.8080234e-05, dtype=float32),
     array(9.250545e-06, dtype=float32),
     array(1.5711368e-05, dtype=float32),
     array(0.00182041, dtype=float32),
     array(1.0102868e-05, dtype=float32),
     array(6.742841e-05, dtype=float32),
     array(1.1911854e-05, dtype=float32),
     array(1.5777034e-05, dtype=float32),
     array(3.4881225e-05, dtype=float32),
     array(1.2460185e-05, dtype=float32),
     array(0.015019, dtype=float32),
     array(1.2740047e-05, dtype=float32),
     array(1.0257865e-05, dtype=float32),
     array(1.0421765e-05, dtype=float32),
     array(0.00029713, dtype=float32),
     array(1.450436e-05, dtype=float32),
     array(0.01116514, dtype=float32),
     array(9.366763e-06, dtype=float32),
     array(0.02021698, dtype=float32)

```

# 예측(다중분류) 테스트 

```python
def predict(predict_sentence):

    data = [predict_sentence, '0']
    dataset_another = [data]

    another_test = BERTDataset(dataset_another, 0, 1, tok, vocab, max_len, True, False)
    test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=batch_size, num_workers=5)

    model.eval()

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)

        valid_length= valid_length
        label = label.long().to(device)

        out = model(token_ids, valid_length, segment_ids)


        test_eval=[]
        for i in out:
            logits=i
            logits = logits.detach().cpu().numpy()
            print(logits)
            if np.argmax(logits) == 0:
                    test_eval.append("1")
            elif np.argmax(logits) == 1:
                test_eval.append("2")
            elif np.argmax(logits) == 2:
                test_eval.append("3")
            elif np.argmax(logits) == 3:
                test_eval.append("4")
            elif np.argmax(logits) == 4:
                test_eval.append("5")

        print(">> 역량은 : " + test_eval[0] + " 입니다.")
        return test_eval[0]
```
