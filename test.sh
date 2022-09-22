DATA_DIR="/USER-DEFINED-PATH/Dataset/"
CONFIG_FILE="configs/test.yaml"

echo "Begin to test."
export CUDA_VISIBLE_DEVICES=0
python scripts/main.py \
	--config-file ${CONFIG_FILE} \
	--root ${DATA_DIR} \