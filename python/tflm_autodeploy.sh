#!/bin/bash

tflite_filename=$1
dst_dir=tflm
mkdir ../evb/src/$dst_dir
cp def_nn3_se.{c,h} ../evb/src/$dst_dir/
cp ./$tflite_filename.tflite \
    ../../neuralSPOT/tools/$tflite_filename.tflite
cd ../../neuralSPOT
python -m venv .venv
pip install .
source .venv/bin/activate
cd tools
ns_autodeploy --tflite-filename ./$tflite_filename.tflite --tensorflow-version ns_tflm_2025_03_19

for file in mut_model_data.h mut_model_init.cc mut_model_metadata.h tflm_ns_model.h; do
cp ../projects/autodeploy/$tflite_filename/tflm_validator/src/$file \
    ../../nnse/evb/src/$dst_dir/$file
done