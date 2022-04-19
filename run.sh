# crf mhch 
# python main.py \
#    --phase train \
#    --model_name hblstmcrf \
#    --suffix .128 \
#    --mode train \
#    --ways mhch_crf \
#    --data_name makeup

# mhch
# python main.py \
#    --phase train \
#    --suffix .128 \
#    --mode train \
#    --ways mhch \
#    --model_name han \
#    --data_name makeup


# multi-task
# python main.py  --phase train  --suffix .128  --mode train  --ways mt  --model_name rssn  --data_name makeup 


# bert
#python bert_main.py --phase train --mode train --suffix .8 --ways mhch --model_name bertlstm --data_name makeup

