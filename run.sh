# Model dir to save logs, ckpts, etc. in "gs://model_dir" format.
export PROJECT_DIR=~/dir1/user_dir
export MODEL_DIR=gs://pongo-bucket/nima/t5x/experiment-msmarco

# Data dir to save the processed dataset in "gs://data_dir" format.
export TFDS_DATA_DIR=gs://pongo-bucket/nima/t5x/data-2
export T5X_DIR=~/t5x
export PYTHONPATH=~/dir1/user_dir

sed -i '254s/(content,/(json.dumps(content),/'  /usr/local/lib/python3.7/dist-packages/tensorflow_datasets/core/features/feature.py

cd ~/t5x/msmarco_v1

python3 ${T5X_DIR}/t5x/train.py \
  --gin_search_paths=${PROJECT_DIR} \
  --gin_file="t5_1_1_base_msmarco_v1.gin" \
  --gin.MODEL_DIR=\"${MODEL_DIR}\" \
  --tfds_data_dir=${TFDS_DATA_DIR}
