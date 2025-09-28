batch=1
seqLen=512
headQ=1
headK=1
hdim=128
topk=4
buffer=2
device_id=4

export LD_LIBRARY_PATH=$(pwd)/out/lib:$(pwd)/out/lib64:$LD_LIBRARY_PATH

rm -rf msprof/*

# python scripts/gen_tiling.py $batch $seqLen $headQ $headK $hdim $topk $buffer
python scripts/gen_golden_data.py $batch $seqLen $headQ $headK $hdim $topk

# ./build/topk_direct_kernel_op $batch $seqLen $headQ $headK $hdim $topk $device_id

msprof op --output=msprof ./build/topk_direct_kernel_op $batch $seqLen $headQ $headK $hdim $topk $device_id

# python scripts/verify_index_golden.py ./output/output_topk_idx.bin ./output/golden_topk_index.bin $((batch*headK*topk))

# python scripts/verify_index_golden.py ./output/output_topk_idx.bin ./output/golden_topk_value.bin $((batch*headK*topk))