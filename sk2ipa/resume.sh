#!/bin/bash


#NVIDIA_PATH="/usr/local/cuda-11.6/targets/x86_64-linux/lib/"
#LD_LIBRARY_PATH=$NVIDIA_PATH python3 whatever.py --resume=yes

model=`ls -1 -t out/ | grep checkpoint | head -n 1 | tr -d ':'`
rm my2.model
ln -s -T "out/$model" my2.model

export WANDB_MODE=offline

python3 code/run_translation.py --vocab_size=63 --model_name_or_path=my2.model --output_dir=out --train_file=dataset.txt --validation_file=evalset.txt  --source_lang sk --target_lang ipa --do_train --do_eval --overwrite_output_dir --num_train_epochs=16 --save_total_limit=3 --per_device_train_batch_size=32 --per_device_eval_batch_size=32 --fp16=True --use_fast_tokenizer --evaluation_strategy=epoch --resume=True
