# POS Tagging
export DATA_DIR="data/ud-treebanks-v2.10/UD_Vietnamese-VTB"
export MODEL="Team-PIXEL/pixel-base-finetuned-pos-ud-vietnamese-vtb"

python scripts/training/run_pos.py \
  --model_name_or_path=${MODEL} \
  --data_dir=${DATA_DIR} \
  --remove_unused_columns=False \
  --do_eval \
  --do_predict \
  --max_seq_length=256 \
  --output_dir='test_output/pos' \
  --overwrite_cache \
  --fallback_fonts_dir='data/fallback_fonts'  # not necessary here, but good to check that it works


# NER
#export LANG="amh"
#export DATA_DIR="data/masakhane-ner/data/${LANG}"
#export MODEL="Team-PIXEL/pixel-base-finetuned-masakhaner-${LANG}"
#
#python scripts/training/run_ner.py \
#  --model_name_or_path=${MODEL} \
#  --data_dir=${DATA_DIR} \
#  --remove_unused_columns=False \
#  --do_eval \
#  --do_predict \
#  --max_seq_length=196 \
#  --output_dir='test_output/ner' \
#  --overwrite_cache \
#  --fallback_fonts_dir='data/fallback_fonts'  # not necessary here, but good to check that it works
