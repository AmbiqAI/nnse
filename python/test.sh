epoch_loaded=137
mode=test
for dtype in int16; do
python train_se.py  \
    --epoch_loaded $epoch_loaded \
    --mode $mode \
    --dtype_tflite $dtype
done