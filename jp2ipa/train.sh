#!/bin/bash


#NVIDIA_PATH="/usr/local/cuda-11.6/targets/x86_64-linux/lib/"
#LD_LIBRARY_PATH=$NVIDIA_PATH python3 code/main.py


WANDB_DISABLED="true" python3 code/run_translation.py --vocab_size=2489 \
--model_name_or_path=t5-small --output_dir=out --train_file=dataset.txt \
--validation_file=evalset.txt  --source_lang jp --target_lang ipa \
--do_train --do_eval --overwrite_output_dir --num_train_epochs=700 \
--save_total_limit=3 --per_device_train_batch_size=32 \
--per_device_eval_batch_size=32 --fp16=True --use_fast_tokenizer \
--evaluation_strategy=epoch
