---
title: "KoBERT를 이용한 학생 역량 점수 분류"
toc: true
toc_sticky: true
categories: NLP
sidebar_main: true
toc_label: "Table of Contents"
toc_icon: "bookmark"
breadcrumbs: false
---

<br>아래 본문은 2023 한국교육공학회 춘계학술대회 <인공지능 시대, 교육공학의 새로운 접근 방향> 포스터 세션에서 발표한 내용으로, 본문 마지막에는 사용한 코드를 함께 정리하였다. 

### 1. 서론

* 최근 인공지능에 대한 교육학계의 관심이 높아지고 있다. 그 중 인공지능의 한 분야인 자연어처리는 ChatGPT의 등장으로 크게 주목 받고 있는 기술이다. 한편, 자연어처리와 관련해 교육학계에서는 이것의 교육적 활용에 대해 논의를 시작한 단계에 있으며, 자연어처리 기술을 적용해 교육 데이터를 분석한 연구는 다소 부족한 실정이다. 

* 이에 본 연구는 딥러닝 기반의 KoBERT를 활용하여, 학생 수업 관찰기록 텍스트 데이터를 바탕으로 학생의 역량(자기관리, 대인관계, 시민, 문제해결역량) 점수를 예측하는 다중분류모델을 개발함으로써, 자연어처리 기술을 적용한 교육 연구의 저변을 확대하는 데 기여하고자 한다.

### 2. BERT, KoBERT

* BERT(Bidirectional Encoder Representations from Transformers)는 2018년 구글에서 발표한 임베딩 모델로서, 트랜스포머 모델을 기반으로 한다. 

* BERT는 크게 사전 학습(pre-training)과 파인 튜닝(fine-tuning)의 두 단계로 진행되는데, 대용량의 언어 데이터를 바탕으로 이미 학습(사전 학습)된 모델을, 해결하고자 하는 목표에 맞게 미세 조정(파인 튜닝)하는 과정을 포함한다. 따라서, BERT는 파인 튜닝을 통해 적은 데이터와 짧은 시간으로 학습이 가능하다는 장점을 지닌다. 

* 구체적으로, 사전 학습은 ‘마스크 언어 모델링(MLM)’과 ‘다음 문장 예측(NSP)’에 대한 학습으로 이루어지며, 파인 튜닝은 다운스트림 태스크(예. 텍스트 분류, 질의 응답 등)의 성격에 따라 BERT 구조 뒤에 레이어를 추가하거나, 파라미터 조정을 통해 가중치를 업데이트 하는 과정을 포함한다(이진기, 2022; Ravichandiran, 2021). 

* 본 연구에서는 SKTBrain에서 대규모의 한국어 말뭉치를 학습시킴으로써, 한국어에 특화되도록 개발한 BERT 모델인 KoBERT를 활용하였다. 

### 3. 데이터 수집 ∙ 전처리 및 모델 학습<br/>
**1) 데이터 수집 및 설명**

  * AI Hub에서 제공한 ‘학생 청소년 핵심역량분석 교육 데이터’를 사용하였으며, 원천 데이터는 다음과 같이 개별 데이터셋으로 구성되었다: (1)학생 정보(성별, 지역, 학교급, 학년 등), (2)수업 정보(수업 성격, 과제 유형, 교과목정보 등), (3)학생관찰기록 텍스트 및 역량 별 점수 데이터셋

  * 최종 분석을 위해 개별 데이터셋을 병합하고, 그 중 고등학생 데이터를 선별하였으며, 데이터 크기는 11,173건에 달했다.

**2) 데이터 전처리**  
<br>(1)피처 셀렉션 (Feature selection)<br/>

* 유용한 피처에 집중하여 모델의 성능을 높이고자, 학생관찰기록 텍스트 데이터(Student assessment)와 함께 수업 정보와 관련된 데이터 Program_category, Mission_category, Subject_category를 모델의 피처로 활용하였다. 

* 수업의 성격 및 목표에 따라 해당 수업에서 발현될 수 있는 학생의 역량 유형과 역량 수준이 다를 수 있다는 가정 하에 위와 같이 피처를 선정하였다. 

* 형태소(일반 명사)의 빈도를 분석해 본 결과, 위와 같은 가정이 충족될 수 있음을 확인하였다. 이를 테면, 수업 과제 유형 Mission_category 피처에 해당하는 데이터가 '도와주기' 일 때, 학생관찰기록 텍스트 데이터 즉, Student assessment 피처에서 출현 빈도가 높은 명사는 '공동체', '배려', '팀원' 등으로 나타났다.

(2) 클래스 불균형(Class Imbalance) 해결

* 데이터 탐색적 분석 결과, 클래스 불균형이 다소 심한 문제가 있어 5개의 클래스(1: 매우 낮음, 2: 낮음, 3: 관측 안됨, 4: 높음, 5: 매우 높음)를 3개의 클래스(0: 매우 낮음, 낮음, 1: 관측 안됨, 2: 높음, 매우 높음)로 일차 재구성하였다. 
* 그럼에도 불구하고, 클래스 불균형 문제가 해소되지 않은 관계로 대표적인 오버샘플링 기법인 SMOTE를 적용하여, 소수 클래스의 샘플 개수를 조정해 데이터셋의 균형을 도모하였다. 그 결과, 클래스별 분포는 아래 그림과 같다.  <br/>
<br><img src="/assets/images/oversamp.png" width="100%" height="100%" title="oversampling"/>  

(3) 데이터셋 재구조화

* 기존 데이터셋은 역량 유형이 4개의 컬럼으로, 각 역량 별 점수가 컬럼 값으로 구성 되었으나, 역량 유형 또한 새로운 피처로 사용 가능하도록 데이터셋을 재구조화 했다. 이를 통해, 데이터 증식 및 모델 1개로 4가지 역량에 대한 점수를 분류하는 효과를 얻을 수 있었다. 최종 데이터셋은 아래 표와 같다. (Label 컬럼의 컬럼 값 1은 자기관리역량, 2는 대인관계역량, 3은 시민역량, 4는 문제해결역량을 의미한다.)

**3) 모델 학습**  

* 모델 학습은 Google Colab Pro 환경에서 진행하였고, KoBERT의 PyTorch API를 활용하였다. 학습 데이터와 테스트 데이터는 8:2의 비율로 구성하였다.
* 본 연구 목표인 분류 문제를 수행하고자, BertClassifier를 활용해 파인 튜닝 하였다. KoBERT는 768차원의 벡터를 출력하므로, 은닉층(hidden layer)의 뉴런 개수(hidden_size)는 그대로 768을 사용하였고, 역량 점수인 클래스는 3개로 구분(0, 1, 2)되므로 클래스의 수(num_classes)는 3으로 설정하였다. 모델 성능 향상을 위해 차례로 256차원, 128 차원을 출력하는 두 개의 은닉층을 추가하였으며, 은닉층에서의 활성화 함수는 ReLU로 설정하였다. 마지막으로, 학습된 모델의 Loss를 측정하기 위해 다중분류 모델의 대표적 손실 함수인 교차 엔트로피(Cross Entropy)를 사용하였으며, 모델의 최적화 함수로는 Adam을 사용하였다. 
* 본 연구에서 지정한 모델의 하이퍼 파라미터 값은 아래 표에 정리하였다. 

### 4. 모델 평가

* 본 연구의 모델 학습 결과, 테스트 데이터의 정확도는 약 78.20%로 나타났다. 
* 참고로 AI Hub에서는 원천 데이터의 더 다양한 피처를 포함하여 MLP, KoBERT 모델을 결합한 앙상블 기법의 모델을 제시하였으며, 테스트 데이터의 정확도는 약 69.8%로 나타나, 본 연구의 모델이 더 우수한 성능을 보인 점을 확인하였다. 

### 5. 결론 

* 본 연구는 교육적 맥락에서 KoBERT를 활용하여 다중분류모델을 개발하고, 약 78.20%의 비교적 높은 수준의 모델 정확도를 구현함으로써, KoBERT를 활용한 자연어처리 기술의 교육적 활용 가능성을 확대하였다는 점에서 의의가 있다. 
* 최근 일부 해외 대학의 경우 신입생 선발과정에 있어 인공지능 도입을 시작하였으며, 국내에서도 관련 논의가 진행된 바 있다(권정민 외, 2021). 따라서, 본 연구에서 구현한 교사의 학생평가 데이터를 바탕으로 학생의 역량을 예측(분류)하는 기술은 앞으로 입학과 관련한 대학기관의 의사결정에 실질적 기여를 할 수 있을 것으로 기대된다. 


### 코드

```python
# 라이브러리 설치 및 SKT KoBERT 파일 로드
!pip install mxnet
!pip install gluonnlp==0.8.0
!pip install tqdm pandas
!pip install sentencepiece
!pip install transformers
!pip install torch
!pip install 'git+https://github.com/SKTBrain/KoBERT.git#egg=kobert_tokenizer&subdirectory=kobert_hf'
```

```python
# 라이브러리 불러오기 
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm, tqdm_notebook
```

```python
# Hugging Face를 통한 모델 및 토크나이저 Import
from kobert_tokenizer import KoBERTTokenizer
from transformers import BertModel
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

# GPU 사용 시
device = torch.device("cuda:0")
```

```python
# KoBERT 모델 선언
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
bertmodel = BertModel.from_pretrained('skt/kobert-base-v1', return_dict=False)
vocab = nlp.vocab.BERTVocab.from_sentencepiece(tokenizer.vocab_file, padding_token='[PAD]')
```

```python
# https://blog.naver.com/newyearchive/223097878715 참고하여 BERTSentenceTransform 클래스 직접 수정 

class BERTSentenceTransform:
    r"""BERT style data transformation.

    Parameters
    ----------
    tokenizer : BERTTokenizer.
        Tokenizer for the sentences.
    max_seq_length : int.
        Maximum sequence length of the sentences.
    pad : bool, default True
        Whether to pad the sentences to maximum length.
    pair : bool, default True
        Whether to transform sentences or sentence pairs.
    """

    def __init__(self, tokenizer, max_seq_length,vocab, pad=True, pair=True):
        self._tokenizer = tokenizer
        self._max_seq_length = max_seq_length
        self._pad = pad
        self._pair = pair
        self._vocab = vocab

    def __call__(self, line):
        """Perform transformation for sequence pairs or single sequences.

        The transformation is processed in the following steps:
        - tokenize the input sequences
        - insert [CLS], [SEP] as necessary
        - generate type ids to indicate whether a token belongs to the first
        sequence or the second sequence.
        - generate valid length

        For sequence pairs, the input is a tuple of 2 strings:
        text_a, text_b.

        Inputs:
            text_a: 'is this jacksonville ?'
            text_b: 'no it is not'
        Tokenization:
            text_a: 'is this jack ##son ##ville ?'
            text_b: 'no it is not .'
        Processed:
            tokens: '[CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]'
            type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
            valid_length: 14

        For single sequences, the input is a tuple of single string:
        text_a.

        Inputs:
            text_a: 'the dog is hairy .'
        Tokenization:
            text_a: 'the dog is hairy .'
        Processed:
            text_a: '[CLS] the dog is hairy . [SEP]'
            type_ids: 0     0   0   0  0     0 0
            valid_length: 7

        Parameters
        ----------
        line: tuple of str
            Input strings. For sequence pairs, the input is a tuple of 2 strings:
            (text_a, text_b). For single sequences, the input is a tuple of single
            string: (text_a,).

        Returns
        -------
        np.array: input token ids in 'int32', shape (batch_size, seq_length)
        np.array: valid length in 'int32', shape (batch_size,)
        np.array: input token type ids in 'int32', shape (batch_size, seq_length)

        """

        # convert to unicode
        text_a = line[0]
        if self._pair:
            assert len(line) == 2
            text_b = line[1]

        tokens_a = self._tokenizer.tokenize(text_a)
        tokens_b = None

        if self._pair:
            tokens_b = self._tokenizer(text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            self._truncate_seq_pair(tokens_a, tokens_b,
                                    self._max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > self._max_seq_length - 2:
                tokens_a = tokens_a[0:(self._max_seq_length - 2)]

        # The embedding vectors for `type=0` and `type=1` were learned during
        # pre-training and are added to the wordpiece embedding vector
        # (and position vector). This is not *strictly* necessary since
        # the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.

        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        #vocab = self._tokenizer.vocab
        vocab = self._vocab
        tokens = []
        tokens.append(vocab.cls_token)
        tokens.extend(tokens_a)
        tokens.append(vocab.sep_token)
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens.extend(tokens_b)
            tokens.append(vocab.sep_token)
            segment_ids.extend([1] * (len(tokens) - len(segment_ids)))

        input_ids = self._tokenizer.convert_tokens_to_ids(tokens)

        # The valid length of sentences. Only real  tokens are attended to.
        valid_length = len(input_ids)

        if self._pad:
            # Zero-pad up to the sequence length.
            padding_length = self._max_seq_length - valid_length
            # use padding tokens for the rest
            input_ids.extend([vocab[vocab.padding_token]] * padding_length)
            segment_ids.extend([0] * padding_length)

        return np.array(input_ids, dtype='int32'), np.array(valid_length, dtype='int32'),\
            np.array(segment_ids, dtype='int32')
```

```python
# 입력 데이터셋 토큰화
class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, vocab, max_len, pad, pair):
      # 위에서 수정한 BERTSentenceTransform 클래스를 적용해야 함
        transform = BERTSentenceTransform(bert_tokenizer, max_seq_length=max_len, vocab=vocab, pad=pad, pair=pair)
        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))
```

```python
# 입력 데이터셋 토큰화 및 오버샘플링 적용
from imblearn.over_sampling import SMOTE

class BERTOverDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, vocab, max_len, pad, pair):
      # 위에서 수정한 BERTSentenceTransform 클래스를 적용해야 함
        transform = BERTSentenceTransform(bert_tokenizer, max_seq_length=max_len,vocab=vocab, pad=pad, pair=pair)
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

```python
# dataset 예시 (전처리 과정 생략, 최종 dataset 일부 출력 결과 아래와 같음)
dataset[:5]
```

    [['1\t진로탐색\t실습/연습\t미션 1에서는 이마만 보여주며 기록할 것이 크게 없음 미션 1에서는 이마만 보여주며 기록할 것이 크게 없음 ',
  1.0],
 ['1\t진로탐색\t실습/연습\t강사가 설명 하는 부분의 프린트물을 확인하고 고개를 끄덕이는 반응을 보임 수업 내내 강사의 말에 집중하고 있는 모습을 보여주며 이에 대해 비언어적인 반응을 보여줌 이에 집중력 및 공감능력이 높은 것으로 판단됨 ',
  2.0],
 ['1\t진로탐색\t실습/연습\t질문이 있냐는 강사님의 물음에 유일하게 질문을 한 학생이다 학습을 효과적으로 효율적으로 참여했다고 여겨질 수 있다 분석적 사고로 질문을 통해서 문제에 대한 해결 방안을 찾고 대안을 도출해 낼 수 있는 능력을 갖춘것으로 판단된다 강사님의 수업에 박수를 쳐준것으로 보아 연대와 배려가 높은 학생이라고 여겨진다 ',
  2.0],
 ['1\t진로탐색\t실습/연습\t수업과 관련된 자료를 펼쳐 확인함 수업에 필요한 자료를 사전에 미리 준비하고 확인하는 모습을 통해 준비능력이 높은 것으로 판단됨 ',
  2.0],
 ['1\t진로탐색\t실습/연습\t사전에 나눠준 프린트물을 준비해 강사의 설명에 맞춰 확인함 사전에 준비 된 프린트물과 강사님의 설명을 함께 듣는 모습을 통해 수업에 집중하고 있음과 동시에 프로그램에 대한 사전 이해가 있는 것으로 판단됨 ',
  2.0]]

```python
from sklearn.model_selection import train_test_split

dataset_train, dataset_test = train_test_split(dataset, test_size=0.2, shuffle=True, random_state=34)

data_train = BERTOverDataset(dataset_train, 0, 1, tokenizer, vocab, max_len, True, False)
data_test = BERTDataset(dataset_test, 0, 1, tokenizer, vocab, max_len, True, False)

train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, num_workers=5, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, num_workers=5, shuffle=True)
```

```python
# KoBERT 학습 모델 생성
class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=3, # 클래스 수 만큼 조정
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)

        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)
```

```python
# BERT 모델 불러오기
model = BERTClassifier(bertmodel,  dr_rate=0.3).to(device)
```

```python
# optimizer와 schedule 설정
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
loss_fn = nn.CrossEntropyLoss() # 다중분류 

t_total = len(train_dataloader) * num_epochs
warmup_step = int(t_total * warmup_ratio)

scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)
```

```python
#정확도 측정을 위한 함수 정의
def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc
```

```python
# KoBERT 모델 학습 
train_history=[]
test_history=[]
train_loss_history=[]
test_loss_history=[]

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
        loss = loss_fn(out, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()  # Update learning rate schedule
        train_acc += calc_accuracy(out, label)
        if batch_id % log_interval == 0:
            print("epoch {} batch id {} loss {} train acc {}".format(e+1, batch_id+1, loss.data.cpu().numpy(), train_acc / (batch_id+1)))
    print("epoch {} train acc {}".format(e+1, train_acc / (batch_id+1)))
    train_history.append(train_acc / (batch_id+1))
    train_loss_history.append(loss.data.cpu().numpy())

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
    test_loss_history.append(loss.data.cpu().numpy())
```

```python
# train_history, test_history 그래프 
!pip install matplotlib

import matplotlib.pyplot as plt
import numpy as np

markers = {'train': 'o', 'test': 's'}
x = np.arange(num_epochs)
plt.plot(x, train_history, label='train')
plt.plot(x, test_history, label='test')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0.5, 1.0)
plt.legend(loc='lower right')
plt.show()
```

<br><img src="/assets/images/train_test_accuracy.png" width="50%" height="50%" title="oversampling"/>  



