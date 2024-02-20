export MODEL="3-parallel-medium-nli-pm-64-128-3e-5-205000/checkpoint-205000"
export DATASETNAME='allnli'
# in the training script, we currently remove all huggingface dataset names  for anonymity
export POOLING_MODE="mean"
export SEQ_LEN=64
export FALLBACK_FONTS_DIR="data/fallback_fonts"
export BSZ=128
export GRAD_ACCUM=1
export LR=3e-5
export SEED=42
export NUM_STEPS=2600
export WARM_STEPS=200
# export NUM_STEPS=205000
# export WARM_STEPS=20000

export CIRCLE_RUN_NAME='pm-a'
export RUN_NAME="3-${DATASETNAME}-${CIRCLE_RUN_NAME}-${SEQ_LEN}-${BSZ}-${LR}-${NUM_STEPS}"

python scripts/training/run_contrastive_training_eval.py \
  --model_name_or_path=${MODEL} \
  --remove_unused_columns=False \
  --dataset_name=${DATASETNAME} \
  --do_train \
  --do_eval \
  --metric_for_best_model="spearman_cosine" \
  --pooling_mode=${POOLING_MODE} \
  --pooler_add_layer_norm=True \
  --dropout_prob=0.1 \
  --max_seq_length=${SEQ_LEN} \
  --max_steps=${NUM_STEPS} \
  --num_train_epochs=10 \
  --per_device_train_batch_size=${BSZ} \
  --per_device_eval_batch_size=16 \
  --gradient_accumulation_steps=${GRAD_ACCUM} \
  --learning_rate=${LR} \
  --warmup_steps=${WARM_STEPS} \
  --run_name=${RUN_NAME} \
  --output_dir=${RUN_NAME} \
  --overwrite_cache \
  --logging_strategy=steps \
  --logging_steps=100 \
  --evaluation_strategy=steps \
  --eval_steps=200 \
  --save_strategy=steps \
  --save_steps=200 \
  --save_total_limit=20 \
  --load_best_model_at_end=True \
  --seed=${SEED} \
  --fallback_fonts_dir=${FALLBACK_FONTS_DIR}

