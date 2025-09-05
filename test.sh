batch=1
seqLen=4096
headQ=8
headK=1
hdim=128
topk=4
buffer=2
device_id=4

rm -rf msprof/*

python scripts/gen_tiling.py $batch $seqLen $headQ $headK $hdim $topk $buffer
python scripts/gen_data.py $batch $seqLen $headQ $headK $hdim $topk

# ./ascendc_kernels_bbit $batch $seqLen $headQ $headK $hdim $topk $device_id

msprof op --output=msprof ./ascendc_kernels_bbit $batch $seqLen $headQ $headK $hdim $topk $device_id