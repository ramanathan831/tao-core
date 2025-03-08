#!/usr/bin/env bash
umask 0
rm -rf /shared/orgs/00000000-0000-0000-0000-000000000000/*
# cp -r shared/* /shared/ ; chmod 777 /shared/orgs ; chmod -R 777 /shared/orgs/00000000-0000-0000-0000-000000000000 2>/dev/null ; true
# cp -r notebooks /shared/
service nginx start
export _PYTHON_LIB_PATH=$(python3 -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
uwsgi --ini $_PYTHON_LIB_PATH/nvidia_tao_core/microservices/uwsgi.ini