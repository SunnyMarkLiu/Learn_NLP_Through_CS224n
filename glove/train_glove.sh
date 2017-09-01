#!/usr/bin/env bash
set -e

make

# 设置语料库的路径
CORPUS=/data/sunnymarkliu/wikimedia/enwiki/cleaned_corpus/wiki.en.text
# 根据语料库获取的词汇列表
VOCAB_FILE=/data/sunnymarkliu/pretrained_models/glove/enwiki_vocab.txt
# cooccurrence 矩阵存放路径
COOCCURRENCE_FILE=/data/sunnymarkliu/pretrained_models/glove/cooccurrence.bin
# cooccurrence shuf 矩阵存放路径
COOCCURRENCE_SHUF_FILE=/data/sunnymarkliu/pretrained_models/glove/cooccurrence.shuf.bin
# make glove 生成的可执行文件
BUILDDIR=build
# 生成的词向量文件
SAVE_FILE=/data/sunnymarkliu/pretrained_models/glove/enwiki_glove_word_vectors
# 日志输出级别
VERBOSE=2
MEMORY=4.0
# 去除语料库中出现次数小于 5 的词汇
VOCAB_MIN_COUNT=5
# 词向量大小
VECTOR_SIZE=300
# 最大迭代次数
MAX_ITER=20
# 上下文窗口大小
WINDOW_SIZE=15
BINARY=2
# 开启线程数
NUM_THREADS=8
X_MAX=10

# training glove model
echo "----------------- generate vocabulary from corpus -----------------"
echo "$ $BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE"
$BUILDDIR/vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE

echo "----------------- generate global co-occurrence matrix -----------------"
echo "$ $BUILDDIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $COOCCURRENCE_FILE"
$BUILDDIR/cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $COOCCURRENCE_FILE

echo "$ $BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE"
$BUILDDIR/shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE

echo "----------------- training glove model -----------------"
echo "$ $BUILDDIR/glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE"
$BUILDDIR/glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE

echo "done!"
