#@IgnoreInspection BashAddShebang
ONMT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

#======= EXPERIMENT SETUP ======
NAME=data.ste.ko-en
TOOLS_DIR="/home/hkh/tools"
DATA_DIR="/home/hkh/data/ted2013/$NAME"
OUTPUT_DIR="/home/hkh/data/ted2013/opennmt.$NAME"

# update these variables
SRC="ko"
TGT="en"

TRAIN_SRC="$DATA_DIR/train.ko-en.tok.clean.ste32k.$SRC"
TRAIN_TGT="$DATA_DIR/train.ko-en.tok.clean.ste32k.$TGT"
VALID_SRC="$DATA_DIR/tst2016.en-ko.tok.ste32k.$SRC"
VALID_TGT="$DATA_DIR/tst2016.en-ko.tok.$TGT"
TEST_SRC="$DATA_DIR/tst2017.en-ko.tok.ste32k.$SRC"
TEST_TGT="$DATA_DIR/tst2017.en-ko.tok.$TGT"

GPUARG="0" # default
#GPUARG="0 1"

#====== EXPERIMENT BEGIN ======

# Check if input exists
for f in $TRAIN_SRC $TRAIN_TGT $VALID_SRC $VALID_TGT $TEST_SRC $TEST_TGT; do
    if [[ ! -f "$f" ]]; then
        echo "Input File $f doesnt exist. Please fix the paths"
        exit 1
    fi
done

echo "Output dir = $OUTPUT_DIR "
[ -d $OUTPUT_DIR ] || mkdir -p $OUTPUT_DIR
[ -d $OUTPUT_DIR/data ] || mkdir -p $OUTPUT_DIR/data
[ -d $OUTPUT_DIR/models ] || mkdir $OUTPUT_DIR/models
[ -d $OUTPUT_DIR/test ] || mkdir -p  $OUTPUT_DIR/test

#: <<EOF
echo "Step 1b: Preprocess"
python $ONMT/preprocess.py \
    -train_src $TRAIN_SRC \
    -train_tgt $TRAIN_TGT \
    -valid_src $VALID_SRC \
    -valid_tgt $VALID_TGT \
    -save_data $OUTPUT_DIR/data/processed


echo "Step 2: Train"
GPU_OPTS=""
if [[ ! -z $GPUARG ]]; then
    GPU_OPTS="-gpu_ranks $GPUARG"
fi
CMD="python $ONMT/train.py -data $OUTPUT_DIR/data/processed -save_model $OUTPUT_DIR/models/$NAME $GPU_OPTS \
-config $ONMT/config/ted2013.ko-en.yml \
-tensorboard -tensorboard_log_dir $OUTPUT_DIR/models/report"

echo "Training command :: $CMD"
eval "$CMD"

#EOF

GPU_OPTS=""
if [ ! -z $GPUARG ]; then
    GPU_OPTS="-gpu $GPUARG"
fi

# clear record files
echo "" > $OUTPUT_DIR/test/test.tc.bleu
echo "" > $OUTPUT_DIR/test/test.lc.bleu
echo "" > $OUTPUT_DIR/test/valid.tc.bleu
echo "" > $OUTPUT_DIR/test/valid.lc.bleu

for model in `ls -tr $OUTPUT_DIR/models/*.pt`; do
    echo "Translating with model: $model"

    echo "Step 3a: Translate Test"
    python $ONMT/translate.py -model $model \
      -src $TEST_SRC \
      -output $OUTPUT_DIR/test/test.out \
      -replace_unk -verbose $GPU_OPTS > $OUTPUT_DIR/test/test.log

    echo "Step 3b: Translate Dev"
    python $ONMT/translate.py -model $model \
      -src $VALID_SRC \
      -output $OUTPUT_DIR/test/valid.out \
      -replace_unk -verbose $GPU_OPTS > $OUTPUT_DIR/test/valid.log

    echo "STE decoding/detokenising target to match with references"
    mv $OUTPUT_DIR/test/test.out{,.ste}
    mv $OUTPUT_DIR/test/valid.out{,.ste}
    python ${TOOLS_DIR}/ste_tokenizer/decode_ste.py \
    "${DATA_DIR}/vocab.ste32k.shared" < $OUTPUT_DIR/test/valid.out.ste > $OUTPUT_DIR/test/valid.out
    python ${TOOLS_DIR}/ste_tokenizer/decode_ste.py \
    "${DATA_DIR}/vocab.ste32k.shared" < $OUTPUT_DIR/test/test.out.ste > $OUTPUT_DIR/test/test.out


    echo "Step 4a: Evaluate Test"
    echo $model >> $OUTPUT_DIR/test/test.tc.bleu
    echo $model >> $OUTPUT_DIR/test/test.lc.bleu
    $ONMT/tools/multi-bleu.perl $TEST_TGT < $OUTPUT_DIR/test/test.out >> $OUTPUT_DIR/test/test.tc.bleu
    $ONMT/tools/multi-bleu.perl -lc $TEST_TGT < $OUTPUT_DIR/test/test.out >> $OUTPUT_DIR/test/test.lc.bleu

    echo "Step 4b: Evaluate Dev"
    echo $model >> $OUTPUT_DIR/test/valid.tc.bleu
    echo $model >> $OUTPUT_DIR/test/valid.lc.bleu
    $ONMT/tools/multi-bleu.perl $VALID_TGT < $OUTPUT_DIR/test/valid.out >> $OUTPUT_DIR/test/valid.tc.bleu
    $ONMT/tools/multi-bleu.perl -lc $VALID_TGT < $OUTPUT_DIR/test/valid.out >> $OUTPUT_DIR/test/valid.lc.bleu
done

#===== EXPERIMENT END ======
