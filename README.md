# CMHCH

## Data Format
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

## Usage
- Data processing

To construct the vocabulary from the pre-trained word embeddings and corpus. For the security of private information from customers, we performed the data desensitization and converted words to IDs. We save the processed data into pickle file.

- Train the model (including training, validation, and testing)

```bash
# clothes

CUDA_VISIBLE_DEVICES=2,3 nohup python -u -W ignore main.py --task train --model cmhch --data clothes --is_only_ssa 1 --info only_ssa > ./logs/clothes_only_ssa.log 2>&1 &

CUDA_VISIBLE_DEVICES=3,0 nohup python -u -W ignore main.py --task train --model cmhch --data clothes --is_only_cf 1 --info only_cf > ./logs/clothes_only_cf.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python -u -W ignore main.py --task train --model cmhch --data clothes > ./logs/clothes.log 2>&1 &

--batch_size 64 
# makeup

CUDA_VISIBLE_DEVICES=0 nohup python -u -W ignore main.py --task train --model cmhch --data clothes --is_only_ssa 1 --info only_ssa > ./logs/clothes_only_ssa_2.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python -u -W ignore main.py --task train --model cmhch --data clothes --is_only_cf 1 --info only_cf > ./logs/clothes_only_cf_2.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python -u -W ignore main.py --task train --model cmhch --data clothes --is_only_ssa 1 --info only_ssa > ./logs/clothes_only_ssa_1.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python -u -W ignore main.py --task train --model cmhch --data clothes --is_only_cf 1 --info only_cf > ./logs/clothes_only_cf_1.log 2>&1 &

CUDA_VISIBLE_DEVICES=3,0 nohup python -u -W ignore main.py --task train --model cmhch --data makeup --info add_senti_loss --batch_size 64 > ./logs/makeup_add_senti_loss.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python -u -W ignore main.py --task train --model cmhch --data makeup --info tune > ./logs/makeup_tune.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python -u -W ignore main.py --task train --model cmhch --data clothes --info 0 > ./logs/clothes_0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python -u -W ignore main.py --task train --model cmhch --data clothes --info 1 > ./logs/clothes_1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python -u -W ignore main.py --task train --model cmhch --data clothes --info 2 > ./logs/clothes_2.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python -u -W ignore main.py --task train --model cmhch --data clothes --info 3 > ./logs/clothes_3.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python -u -W ignore main.py --task train --model cmhch --data makeup --info 0 > ./logs/makeup_0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python -u -W ignore main.py --task train --model cmhch --data makeup --info 1 > ./logs/makeup_1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python -u -W ignore main.py --task train --model cmhch --data makeup --info 2 > ./logs/makeup_2.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python -u -W ignore main.py --task train --model cmhch --data makeup --info 3 > ./logs/makeup_3.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python -u -W ignore main.py --task train --model cmhch --data makeup --info pre_epoch > ./logs/makeup_pre_epoch.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python -u -W ignore main.py --task train --model cmhch --data clothes --info pre_epoch > ./logs/clothes_pre_epoch.log 2>&1 &

--memory 0.9
--batch_size 64
```
kill -9 `ps -ef | grep CMHCH/main | awk '{print $2}'`

- Test the model

```bash
nohup python -u -W ignore main.py --task test --model cmhch --data clothes --info test --model_path /data1/user02/zss/CMHCH/nor_weights/clothes/cmhch.ordinary.total_epoch80.pre_epoch20/best > ./logs/test_clothes.log 2>&1 &

nohup python -u -W ignore main.py --task test --model cmhch --data makeup --info test --model_path /data1/user02/zss/CMHCH/nor_weights/makeup/cmhch.tune.total_epoch80.pre_epoch20/best > ./logs/test_makeup.log 2>&1 &
```
根据统计，1表示切换人工，0表示不切换人工

