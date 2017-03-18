# KeyedVectorsANN
Genism word2vec + Pysparnn ANN + Trimmed GoogleNewsVec - Fast and lightweight  
Extension of gensim's KeyedVectors using pysparnn's approximate nearest neighber indexer. Depends on gensim, numpy, sklearn and scipy.  
Also includes a utility to load Google News' vector and collapse down to a manageable size.  
Copyright (C) 2017 Hiep Huu Nguyen <ontocord@gmail.com>  
Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html  
See additional licenses at https://github.com/facebookresearch/pysparnn and https://radimrehurek.com/gensim/  

Download the the google vector file from here: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit  

TODO: Extend gensim's word2vec to use KeyedVectorsANN. Refactor to use the pysparnn module when compatible.  

You can create the model based on Google News' 3,000,000 word vectors. This will result in a vocab of ~16K words with an additional ~44K "synonyms" of compound words.  

Model creation:

~~~
import gensim, time
from KeyedVectorsANN import *

t = time.time()    
kv = prepareANNModel("./nativedata/GoogleNews-vectors-negative300.bin", "GoogleNewsANN.bin", createSynonyms=True)
print ("finished creating model ...", time.time() - t, "seconds")        

~~~

Testing accuracy:
~~~ 
import gensim, time
from KeyedVectorsANN import *

t = time.time()    
#assumes you have already created the ANN file by calling:
#kv = prepareANNModel("./nativedata/GoogleNews-vectors-negative300.bin", "GoogleNewsANN.bin", createSynonyms=True)
kv = KeyedVectorsANN.load('GoogleNewsANN.bin')
#you can set the k_clusters variable to get higher accuracy at the expense of speed. 
#1 is fastest with lowest accuracy. Default is set to 10.
#kv.indexer.k_clusters = 1 
#simple analogy lookup
#print (kv.most_similar(['dog',], [], 10))
#you can also pass k_clusters=<number> as a parameter to most_similar
#do an accuracy test
acc_data = kv.accuracy_indexer("./nativedata/questions-words.txt")
for section in acc_data:
  if len(section['correct']) +  len(section['incorrect']) > 0:
    if section['section'] == 'total':
      print (section['section'], len(section['correct'])/(len(section['correct']) +  len(section['incorrect'])), len(section['correct']),  len(section['correct']) + len(section['incorrect']))
    else:
      print (section['section'], len(section['correct'])/(len(section['correct']) +  len(section['incorrect'])))
print ("finished ann accuracy ...", time.time() - t, "seconds")        

~~~ 

Results in:
 
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
finished ann accuracy ... 45.34215688705444 seconds
~~~
