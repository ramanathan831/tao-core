#!/bin/bash

echo "Installing required packages"
pip install pyarmor==7.7.4 pyinstaller pybind11
echo "Registering pyarmor"
pyarmor register release/python/pyarmor-regfile-1219.zip || exit $?

echo "Clearing build and dists"
python setup.py clean --all
rm -rf dist
echo "Clearing pycache and pycs"
find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf

#This makes sure the non-py files are retained. Py files are repplaced in th next step
mkdir dist
cp -r nvidia_tao_core/* dist/

echo "Obfuscating the code using pyarmor"
python -c "from release.python.utils import encrypt_source_code; encrypt_source_code.encrypt_files('nvidia_tao_core')"

echo "Migrating codebase"
# Move sources to orig_src
rm -rf orig_src
mkdir orig_src
mv nvidia_tao_core/* orig_src/

# Move obf_src files to src
mv dist/* nvidia_tao_core/
mv nvidia_tao_core/pytransform_vax_001219 .

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
rm -rf pytransform_vax_001219
