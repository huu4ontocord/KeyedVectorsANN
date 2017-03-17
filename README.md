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
# assumes you have already created the ANN file by calling:
# kv = prepareANNModel("./nativedata/GoogleNews-vectors-negative300.bin", "GoogleNewsANN.bin", createSynonyms=True)
kv = KeyedVectorsANN.load('GoogleNewsANN.bin')
print (kv.most_similar(['dog',], [], 10))
acc_data = kv.accuracy_indexer("./nativedata/questions-words.txt")
for section in acc_data:
  if len(section['correct']) +  len(section['incorrect']) > 0:
    if section['section'] == 'total':
      print (section['section'], len(section['correct'])/(len(section['correct']) +  len(section['incorrect'])), len(section['correct']),  len(section['correct']) + len(section['incorrect']))
    else:
      print (section['section'], len(section['correct'])/(len(section['correct']) +  len(section['incorrect'])))
~~~ 
