# Created by hkh at 2019-03-21
#
ONMT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

MODEL=/home/hkh/data/ted2013/opennmt.data.ko-en/models/data.ko-en_step_20000.pt
TEST_SRC=/home/hkh/data/ted2013/data.ko-en/tst2017.en-ko.tok.bpe32k.ko
OUTPUT=/tmp/output1.txt

echo "hello"
python $ONMT/sent_vectors.py -gpu 0 -model $MODEL \
  -src $TEST_SRC \
  -output $OUTPUT

# MODEL=/home/hkh/data/ted2013/opennmt.data.ko-en/models/data.ko-en_step_20000.pt
#TEST_SRC=/home/hkh/data/ted2013/data.ko-en/tst2017.en-ko.tok.bpe32k.ko
#OUTPUT=/tmp/output1.txt

#python sent_vectors.py -gpu 0 -model /home/hkh/data/ted2013/opennmt.data.ko-en/models/data.ko-en_step_20000.pt -src /home/hkh/data/ted2013/data.ko-en/tst2017.en-ko.tok.bpe32k.ko -output /tmp/output1.txt
