## GloVe: Global Vectors for Word Representation

## Download pre-trained word vectors
The links below contain word vectors obtained from the respective corpora. If you want word vectors trained on massive web datasets, you need only download one of these text files! Pre-trained word vectors are made available under the <a href="http://opendatacommons.org/licenses/pddl/">Public Domain Dedication and License</a>. 
<div class="entry">
<ul style="padding-left:0px; margin-top:0px; margin-bottom:0px">
  <li> Common Crawl (42B tokens, 1.9M vocab, uncased, 300d vectors, 1.75 GB download): <a href="http://nlp.stanford.edu/data/wordvecs/glove.42B.300d.zip">glove.42B.300d.zip</a> </li>
  <li> Common Crawl (840B tokens, 2.2M vocab, cased, 300d vectors, 2.03 GB download): <a href="http://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip">glove.840B.300d.zip</a> </li>
  <li> Wikipedia 2014 + Gigaword 5 (6B tokens, 400K vocab, uncased, 300d vectors, 822 MB download): <a href="http://nlp.stanford.edu/data/wordvecs/glove.6B.zip">glove.6B.zip</a> </li>
  <li> Twitter (2B tweets, 27B tokens, 1.2M vocab, uncased, 200d vectors, 1.42 GB download): <a href="http://nlp.stanford.edu/data/wordvecs/glove.twitter.27B.zip">glove.twitter.27B.zip</a>
</ul>
</div>

## Train word vectors on a new corpus

<img src="https://travis-ci.org/stanfordnlp/GloVe.svg?branch=master"></img>

If the web datasets above don't match the semantics of your end use case, you can train word vectors on your own corpus.

    $ ./train_glove.sh

### License
All work contained in this package is licensed under the Apache License, Version 2.0. See the include LICENSE file.