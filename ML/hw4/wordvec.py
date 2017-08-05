import word2vec
import numpy as np
import nltk
import os
import sys
import re
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from adjustText import adjust_text

MIN_COUNT = 10
WORDVEC_DIM = 100
WINDOW = 5
NEGATIVE_SAMPLES = 10
ITERATIONS = 5
MODEL = 0
LEARNING_RATE = 0.07

# train model
if not os.path.isfile('all_ta.bin') or sys.argv[1] == "redo":
    
    word2vec.word2vec(
        train='all.txt',
        output='all_ta.bin',
        cbow=MODEL,
        size=WORDVEC_DIM,
        min_count=MIN_COUNT,
        window=WINDOW,
        negative=NEGATIVE_SAMPLES,
        iter_=ITERATIONS,
        alpha=LEARNING_RATE,
        verbose=True)
    
    #word2vec.word2vec('all.txt','all_ta.bin')
    print("Finish training")
    # load model for plotting
model = word2vec.load('all_ta.bin')
vocabs = []                 
vecs = []                   
for vocab in model.vocab:
    vocabs.append(vocab)
    vecs.append(model[vocab])
plot_num = 1000
vecs = np.array(vecs)[:plot_num]
vocabs = vocabs[:plot_num]
#Dimensionality Reduction

tsne = TSNE(n_components=2)
reduced = tsne.fit_transform(vecs)
#Plotting
# filtering
use_tags = set(['JJ', 'NNP', 'NN', 'NNS'])
#puncts = ["'", '.', ':', ";", ',', "?", "!", "’"]
punc = re.compile('[“ , . : ; ’ ! ? ”]')
    
plt.figure()
texts = []
for i, label in enumerate(vocabs):
    pos = nltk.pos_tag([label])
    if (label[0].isupper() and len(label) > 1 and pos[0][1] in use_tags
            and (punc.search(label) == None)):
        x, y = reduced[i, :]
        texts.append(plt.text(x, y, label))
        plt.scatter(x, y)

adjust_text(texts, arrowprops=dict(arrowstyle='-', color='k', lw=0.5))
plt.show()
