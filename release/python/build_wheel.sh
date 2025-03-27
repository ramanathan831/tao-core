#!/bin/bash

echo "Installing required packages"
pip install --upgrade pip setuptools
pip install pyarmor==8.5.8 pyinstaller pybind11
echo "Registering pyarmor"
pyarmor -d reg release/python/pyarmor-regfile-1219.zip || exit $?

echo "Clearing build and dists"
python setup.py clean --all
echo "Clearing pycache and pycs"
find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf

echo "Obfuscating the code using pyarmor"
# This makes sure the non-py files are retained.
pyarmor cfg data_files=*
pyarmor -d gen --recursive --output obf_src/ nvidia_tao_core/ || exit $?

echo "Migrating codebase"
# Move sources to orig_src
rm -rf orig_src
mkdir orig_src
mv nvidia_tao_core/* orig_src/

# Move obf_src files to src
mv obf_src/* ./

echo "Building bdist wheel"
python setup.py bdist_wheel || exit $?

echo "Restoring the original project structure"
# Move the obf_src files.
rm -rf nvidia_tao_core/*

# Move back the original files
mv orig_src/* nvidia_tao_core/

# # Remove the tmp folders.
rm -rf orig_src
rm -rf obf_src
rm -rf pyarmor_runtime_001219
