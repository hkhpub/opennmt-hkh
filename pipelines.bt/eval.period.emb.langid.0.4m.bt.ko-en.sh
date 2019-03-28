#@IgnoreInspection BashAddShebang
ONMT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

#======= EXPERIMENT SETUP ======
NAME=data.period.emb.langid.0.4m.bt.ko-en
DATA_DIR="/home/hkh/data/ted2013/data.ko-en"
OUTPUT_DIR="/home/hkh/data/ted2013/opennmt.$NAME"
BT_DIR=/home/hkh/data/wmt16_en_data/emb_ranked

# update these variables
SRC="ko"
TGT="en"


TRAIN_SRC="$OUTPUT_DIR/train.merge.tok.clean.bpe32k.$SRC"
TRAIN_TGT="$OUTPUT_DIR/train.merge.tok.clean.bpe32k.$TGT"
VALID_SRC="$DATA_DIR/tst2016.en-ko.tok.bpe32k.$SRC"
VALID_TGT="$DATA_DIR/tst2016.en-ko.tok.$TGT"
TEST_SRC="$DATA_DIR/tst2017.en-ko.tok.bpe32k.$SRC"
TEST_TGT="$DATA_DIR/tst2017.en-ko.tok.$TGT"

GPUARG="0" # default
#GPUARG="0 1"

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
      --replace_unk -verbose -gpu $GPUARG > $OUTPUT_DIR/test/test.log

    echo "Step 3b: Translate Dev"
    python $ONMT/translate.py -model $model \
      -src $VALID_SRC \
      -output $OUTPUT_DIR/test/valid.out \
      --replace_unk -verbose -gpu $GPUARG > $OUTPUT_DIR/test/valid.log

    echo "BPE decoding/detokenising target to match with references"
    mv $OUTPUT_DIR/test/test.out{,.bpe}
    mv $OUTPUT_DIR/test/valid.out{,.bpe}
    cat $OUTPUT_DIR/test/valid.out.bpe | sed -E 's/(@@ )|(@@ ?$)//g' > $OUTPUT_DIR/test/valid.out
    cat $OUTPUT_DIR/test/test.out.bpe | sed -E 's/(@@ )|(@@ ?$)//g' > $OUTPUT_DIR/test/test.out

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
