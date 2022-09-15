DATA_DIR="/home/lab314/HDD/Dataset"
CONFIG_FILE="configs/test.yaml"

echo "Begin to test."
export CUDA_VISIBLE_DEVICES=4
python scripts/main.py \
	--config-file ${CONFIG_FILE} \
	--root ${DATA_DIR} \