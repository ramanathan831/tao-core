#!/usr/bin/env bash
umask 0
json_string="$2"
ngcKey=$(jq -r '.auths."nvcr.io".password' <<< "$json_string")
python3 /usr/local/lib/python3.10/dist-packages/nvidia_tao_core/microservices/pretrained_models.py --shared-folder-path ptms --org-teams $1 --ngc-key $ngcKey

## Clear users session cache of expired tokens
python3 /usr/local/lib/python3.10/dist-packages/nvidia_tao_core/microservices/mongo_users_cleanup.py

## Install mongodump
apt update
wget https://fastdl.mongodb.org/tools/db/mongodb-database-tools-ubuntu2204-x86_64-100.10.0.deb ## Update this when upgrading from Ubuntu 20.04 -> 22.04 OR x86 -> ARM
apt install ./mongodb-database-tools-*-100.10.0.deb
rm -f mongodb-database-tools-*.deb
python3 /usr/local/lib/python3.10/dist-packages/nvidia_tao_core/microservices/mongodb_backup.py --access-key $3 --secret-key $4 --s3-bucket-name $5 --s3-bucket-region $6