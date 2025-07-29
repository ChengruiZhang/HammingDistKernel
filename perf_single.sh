batchSize=$1
seqLen=$2
headQ=$3
headK=$4
hidDim=$5
topK=$6
deviceID=$7


# transA=0
# transB=1
repeat=5
batch=20

EXEC_PATH=./ascendc_kernels_bbit
PROF_FLAG=false
PROF_DIR=msprof
filename=M${M}_N${N}_K${K}_${transA}${transB}

export LD_LIBRARY_PATH=$(pwd)/out/lib:$(pwd)/out/lib64:${_ASCEND_INSTALL_PATH}/lib64:$LD_LIBRARY_PATH

python3 scripts/golden.py $batchSize $seqLen $headQ $headK $hidDim $topK
if $PROF_FLAG; then
    if [ -d $PROF_DIR ]; then
        rm -rf $PROF_DIR
    fi
    mkdir -p $PROF_DIR
    chmod 744 $PROF_DIR
    msprof op --output=$PROF_DIR/$filename \
                $EXEC_PATH $batchSize $seqLen $headQ $headK $hidDim $topK $deviceID
else
    echo "Executing $EXEC_PATH $batchSize $seqLen $headQ $headK $hidDim $topK $deviceID"
    $EXEC_PATH $batchSize $seqLen $headQ $headK $hidDim $topK $deviceID
fi
# python3 scripts/verify_result.py output/output.bin output/golden.bin