# RSSN for MHCH+SSA
Core implementation of EMNLP-2021 paper: [A Role-Selected Sharing Network for Joint Machine-Human Chatting Handoff and Service Satisfaction Analysis](https://arxiv.org/abs/2109.08412)

<div align=center><img src="./resources/exemplar.png" height="500"/></div>

# Requirements
- Python 3.6 or higher
- tensorflow==1.14.0
- Keras==2.2.5
- tqdm==4.35.0
- jieba==0.39

# Environment
- Tesla V100 16GB GPU
- CUDA 10.1

# Data Format
Each json file is a data list that includes dialogue samples. The format of a dialogue sample is shown as follows:
```json
{
   "session": [
     {
       "content": "啥时候能收到",
       "label": 0,
       "round": 1,
       "role": "c2b",
       "senti": "3"
     },
     {
       "content": "一般2到3天哦",
       "label": 0,
       "round": 2,
       "role": "b2c",
       "senti": "3"
     },
     {
       "content": "皮肤黑，不知道选哪个",
       "label": 0,
       "round": 3,
       "role": "c2b",
       "senti": "3"
     },
     {
       "content": "不知道选哪个？适合自己的才是最好的，推荐直接下单体验，7天内可无理由退货。若问题还没解决，可以请“人工”",
       "label": 1,
       "round": 4,
       "role": "b2c",
       "senti": "3"
     },
     {
       "content": "废话啊",
       "label": 1,
       "round": 5,
       "role": "b2c",
       "senti": "1"
     },
     {
       "content": "皮肤黑，我不知道买哪个, 不推荐我就不买了!",
       "label": 0,
       "round": 6,
       "role": "c2b",
       "senti": "2"
     }
   ],
   "sessionID": "4",
   "score": "1"
 }
```
Here we use json format to save these dialogues:

- “session”: the current dialogue
  - “content”: utterance content
  - “label”: the current utterance is transferable or normal
  - “round”: the order of the utterance in the current dialogue
  - “role”: role information of the current utterance
  - “senti”: sentiment of the current utterance(1 and 2 denote negative, 3 denotes neutral, 4 and 5 denote positive)
- “sessionid”: ID of the current dialogue
- “score”: Overall satisfaction of the current dialogue ( 1 denotes unsatisfied, 2 denotes met satisfied, 3 denotes well satisfied)

Our experiments are conducted based on two publicly available Chinese customer service dialogue datasets, namely Clothes and Makeup, collected by [Song et al. (2019)](https://github.com/songkaisong/ssa) from [Taobao](https://www.taobao.com/). Meanwhile, we also annotate the *transferable/normal* labels for both datasets according to the existing specifications ([Liu et al., 2020](https://arxiv.org/abs/2012.07610)).
For the security of private information from customers, we performed the data desensitization and converted words to IDs following [Song et al. (2019)](https://github.com/songkaisong/ssa).
The vocab.pkl contains the pre-trained glove word embeddings of token ids.

# Usage
- Data processing

To construct the vocabulary from the pre-trained word embeddings and corpus. For the security of private information from customers, we performed the data desensitization and converted words to IDs. We save the processed data into pickle file.

- Train the model (including training, validation, and testing)
```bash
CUDA_VISIBLE_DEVICES=1 nohup python -u -W ignore main.py --phase train --suffix .128 --mode train --ways mt --model_name cmhch --data_name clothes --log_info weight_satisfaction > ./logs/clothes_weight_satisfaction.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python -u -W ignore main.py --phase train --suffix .128 --mode train --ways counterfactual --model_name cmhch --data_name makeup --log_info counterfactual_cost_loss_pre_train > ./logs/counterfactual.log 2>&1 &
```
nohup python -u -W ignore main.py --phase train --suffix .128 --mode train --ways counterfactual --model_name cmhch --data_name clothes --log_info counterfactual_cost_loss_pre_train > ./logs/clothes_counterfactual.log 2>&1 &

dpkg -i 