batch=1
seqLen=1024
headQ=8
headK=1
hdim=128
topk=128
buffer=2
device_id=5

python scripts/gen_tiling.py $batch $seqLen $headQ $headK $hdim $topk $buffer
python scripts/gen_data.py $batch $seqLen $headQ $headK $hdim $topk

./ascendc_kernels_bbit $batch $seqLen $headQ $headK $hdim $topk $device_id
