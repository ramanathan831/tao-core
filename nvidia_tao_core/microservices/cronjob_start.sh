#!/usr/bin/env bash
umask 0
PYTHON_LIB_PATH=$(python3 -c 'import sysconfig; print(sysconfig.get_path("purelib"))')

python3 $PYTHON_LIB_PATH/nvidia_tao_core/microservices/pretrained_models.py --shared-folder-path ptms --org-teams $1

## Clear users session cache of expired tokens
python3 $PYTHON_LIB_PATH/nvidia_tao_core/microservices/mongo_users_cleanup.py

python3 $PYTHON_LIB_PATH/nvidia_tao_core/microservices/mongodb_backup.py --access-key $3 --secret-key $4 --s3-bucket-name $5 --s3-bucket-region $6