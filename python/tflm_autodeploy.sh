#!/bin/bash

tflite_filename=$1
dst_dir=tflm
evb=evb
tflm_version=ns_tflm_v1_0_0
cp def_nn3_se.{c,h} ../${evb}/src/$dst_dir/
cp ./$tflite_filename.tflite \
    ../../neuralSPOT/tools/$tflite_filename.tflite
cd ../../neuralSPOT

TOML_FILE="pyproject.toml"
# Replace tensorflow version
sed -i.bak 's|tensorflow~=2\.16\.2|tensorflow==2.15.1|' "$TOML_FILE"
echo "Updated TensorFlow version to 2.15.1 in $TOML_FILE"

mkdir projects/autodeploy
python -m venv .venv
pip install --upgrade pip
pip install .

# fix for ubuntu
mkdir -p extern/AmbiqSuite/R5.2.0/pack/svd
cp extern/AmbiqSuite/R5.2.0/pack/SVD/apollo5b.svd \
    extern/AmbiqSuite/R5.2.0/pack/svd/apollo5b.svd
# endfix
source .venv/bin/activate
cd tools
ns_autodeploy --tflite-filename ./$tflite_filename.tflite --tensorflow-version ${tflm_version}

src_h=../projects/autodeploy/$tflite_filename/tflm_validator/src/mut_model_metadata

# awk '/#define TFLM_VALIDATOR_ARENA_SIZE/ {
#     $3 = $3 + 80
# } { print }' ${src_h}.h > ${src_h}_tmp.h && mv ${src_h}_tmp.h ${src_h}.h


for file in mut_model_data.h mut_model_init.cc mut_model_metadata.h tflm_ns_model.h; do
cp ../projects/autodeploy/$tflite_filename/tflm_validator/src/$file \
    ../../nnse/${evb}/src/$dst_dir/$file
done