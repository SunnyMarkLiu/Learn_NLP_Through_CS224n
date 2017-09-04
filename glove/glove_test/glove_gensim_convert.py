#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: MarkLiu
@time  : 17-9-4 下午2:31
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

import shutil
import smart_open
from sys import platform

import gensim


def prepend_line(infile, outfile, line):
    """ 
    Function use to prepend lines using bash utilities in Linux. 
    (source: http://stackoverflow.com/a/10850588/610569)
    """
    with open(infile, 'r') as old:
        with open(outfile, 'w') as new:
            new.write(str(line) + "\n")
            shutil.copyfileobj(old, new)


def prepend_slow(infile, outfile, line):
    """
    Slower way to prepend the line by re-creating the inputfile.
    """
    with open(infile, 'r') as fin:
        with open(outfile, 'w') as fout:
            fout.write(line + "\n")
            for line in fin:
                fout.write(line)


def get_lines(glove_file_name):
    """Return the number of vectors and dimensions in a file in GloVe format."""
    with smart_open.smart_open(glove_file_name, 'r') as f:
        num_lines = sum(1 for line in f)
    with smart_open.smart_open(glove_file_name, 'r') as f:
        num_dims = len(f.readline().split()) - 1
    return num_lines, num_dims


def convert_glove_model_2_gensim_w2c(glove_file, save_gensim_model_path):
    """
    load glove model and convert to gensim word2vec model and save it
    :param glove_file: GloVe Model File, glove.6B.300d.txt, download from http://nlp.stanford.edu/projects/glove/
    :param save_gensim_model_path: save Gensim Model path
    :return: 
    """
    glove_model_filename = glove_file.split('/')[-1]
    print 'glove model: {}'.format(glove_model_filename)

    print 'convert glove model format to gensim model...'
    num_lines, dims = get_lines(glove_file)
    gensim_first_line = "{} {}".format(num_lines, dims)

    # Prepends the line.
    new_gensime_text = save_gensim_model_path + '/' + glove_model_filename + '.converted_gensim_w2c.txt'

    if platform == "linux" or platform == "linux2":
        prepend_line(glove_file, new_gensime_text, gensim_first_line)
    else:
        prepend_slow(glove_file, new_gensime_text, gensim_first_line)

    print 'loads the newly created gensim model into gensim api...'
    # Demo: Loads the newly created glove_model.txt into gensim API.
    model = gensim.models.KeyedVectors.load_word2vec_format(new_gensime_text, binary=False)  # GloVe Model

    model.save(new_gensime_text + '.model')
    model.save_word2vec_format(new_gensime_text + '.vector', binary=False)
    print 'done!'

if __name__ == '__main__':
    glove_pretrained_model = '/data/sunnymarkliu/pretrained_models/glove/Wikipedia_2014/glove.6B.300d.txt'
    glove_2_gensim_pretrained_model_path = '/data/sunnymarkliu/pretrained_models/glove/Wikipedia_2014/convert2gensim'
    convert_glove_model_2_gensim_w2c(glove_pretrained_model, glove_2_gensim_pretrained_model_path)

    f_model = '/data/sunnymarkliu/pretrained_models/glove/Wikipedia_2014/convert2gensim/glove.6B.300d.txt.converted_gensim_w2c.txt.model'
    model = gensim.models.KeyedVectors.load(f_model)
    print model.most_similar("queen")
