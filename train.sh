DATA_DIR="/USER-DEFINED-PATH/Dataset/"
CONFIG_FILE="configs/osnet_ain_x0_5.yaml"

echo "Begin to train."
export PYTHONPATH=$PYTHONPATH:`pwd`
export CUDA_VISIBLE_DEVICES=0
python scripts/main.py \
	--root ${DATA_DIR} \
	--config-file ${CONFIG_FILE}
