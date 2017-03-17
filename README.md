# KeyedVectorsANN
Word2Vec + ANN + Trimmed GoogleNewsVec - Fast and lightweight  
Extension of gensim's KeyedVectors using pysparnn's ANN indexer. Depends on gensim, numpy, sklearn and scipy.  
Also includes a utility to load GoogleNews' vector and collapse down to a manageable size.  
Copyright (C) 2017 Hiep Huu Nguyen <ontocord@gmail.com>  
Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html  
See additional licenses at https://github.com/facebookresearch/pysparnn and https://radimrehurek.com/gensim/  


Usage:
~~~ 
import gensim
from KeyedVectorsANN import *
#assumes you have already created the ANN file by calling:
#kv = prepareANNModel("./nativedata/GoogleNews-vectors-negative300.bin", "GoogleNewsANN.bin", createSynonyms=True)
kv = KeyedVectorsANN.load('GoogleNewsANN.bin')
#simple analogy lookup
#print (kv.most_similar(['dog',], [], 10))
#do an accuracy test
acc_data = kv.accuracy_indexer("./nativedata/questions-words.txt")
for section in acc_data:
  if len(section['correct']) +  len(section['incorrect']) > 0:
    if section['section'] == 'total':
      print (section['section'], len(section['correct'])/(len(section['correct']) +  len(section['incorrect'])), len(section['correct']),  len(section['correct']) + len(section['incorrect']))
    else:
      print (section['section'], len(section['correct'])/(len(section['correct']) +  len(section['incorrect'])))
~~~ 

 
~~~

capital-common-countries 0.9263157894736842
capital-world 0.9041731066460588
currency 0.4262948207171315
city-in-state 0.7990165949600492
family 0.8201581027667985
gram1-adjective-to-adverb 0.3326612903225806
gram2-opposite 0.44950738916256155
gram3-comparative 0.7507507507507507
gram4-superlative 0.8368983957219251
gram5-present-participle 0.7395833333333334
gram6-nationality-adjective 0.9441571871768356
gram7-past-tense 0.6730769230769231
gram8-plural 0.7942942942942943
gram9-plural-verbs 0.732183908045977
total 0.7252097774534841 9939 13705
~~~
