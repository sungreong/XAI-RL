# XAI-RL
RL에 XAI 적용해보기

강화학습 파이토치 모델을 만든 후에 Captum을 사용하여 XAI 적용해보기 

# 설치

## git clone 

```
git clone https://github.com/sungreong/XAI-RL.git
cd XAI-RL
```

## 파이썬 환경 
```
conda create -n xai python=3.8
conda activate xai 
pip install -r requirements.xtx
./installaion.sh
python setup.py develop
```

# 실행


## 훈련 

```
./jupyter/train.ipynb
```

## XAI

```
./jupyter/xai.ipynb
```


