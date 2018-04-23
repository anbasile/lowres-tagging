# 0.9258 first result I got, random seed
# 0.9246 with topic, seed 42, loss 0.0036, 20 iters
# 0.9271 with topic, seed 42, loss 0.0001, 100 iters
# 0.9275 w/no topic, seed 42, loss 0.0000, 100 iters
# MTL, w topic, 100 iters, seed 42
# [lr=0.1 clips=3 updates=1000] 8.344590022497144e-05
# 0.9281789638932496
# MTL, w/no topic, 100 iters, seed 42
#[lr=0.1 clips=6 updates=1000] 0.0001348906319961877
#0.929945054945055
# MTL, w topic, 20 iters, seed 42
# [lr=0.1 clips=53 updates=1000] 0.17308817080800945
# 0.9258241758241759
# MTL, w/no topic, 20 iters, seed 42
# [lr=0.1 clips=85 updates=1000] 0.0038225452183890266
# 0.9248430141287284

import dynet_config
dynet_config.set(random_seed=42, autobatch=True)
import dynet as dy
from collections import Counter
import random
import numpy as np


# format of files: each line is "word<TAB>tag<newline>", blank line is new sentence.
train_file= "../../data/nn/ita.train"
test_file= "../../data/nn/ita.test"

class Vocab:
    def __init__(self, w2i):
        self.w2i = dict(w2i)
        self.i2w = {i:w for w,i in w2i.items()}

    @classmethod
    def from_corpus(cls, corpus):
        w2i = {}
        for sent in corpus:
            for word in sent:
                w2i.setdefault(word, len(w2i))
        return Vocab(w2i)

    def size(self):
        return len(self.w2i.keys())

def read(fname):
    sent = []
    for line in open(fname):
        line = line.strip().split()
        if not line:
            if sent: yield sent
            sent = []
        else:
            w,p = line
            sent.append((w,p))

train=list(read(train_file))[:3000]
test=list(read(test_file))
words=[]
tags=[]


TOPIC=True
# read topic
with open('topic.train','r') as f:
    topics = f.readlines()

topics = list(int(x.strip()) for x in topics)[:3000]

assert len(topics) == len(train)

print("N training docs", len(train))
print("N testing docs", len(test))

wc=Counter()
for s in train:
    for w,p in s:
        words.append(w)
        tags.append(p)
        wc[w]+=1
words.append("_UNK_")
#words=[w if wc[w] > 1 else "_UNK_" for w in words]
tags.append("_START_")

for s in test:
    for w,p in s:
        words.append(w)

vw = Vocab.from_corpus([words])
vt = Vocab.from_corpus([tags])
UNK = vw.w2i["_UNK_"]

nwords = vw.size()
ntags  = vt.size()
ntopics = len(set(topics))

print("tagset size: ", ntags)
print(set(tags))

model = dy.Model()
trainer = dy.SimpleSGDTrainer(model)


E = model.add_lookup_parameters((nwords, 128))
# p_t1  = model.add_lookup_parameters((ntags, 30))

pH = model.add_parameters((32, 50*2))
pO = model.add_parameters((ntags, 32))

# pO = model.add_parameters((ntags, 50*2))

builders=[
        dy.LSTMBuilder(1, 128, 50, model),
        dy.LSTMBuilder(1, 128, 50, model),
        ]
if TOPIC:
    # MLP for topic
    topic_hlayer = model.add_parameters((100, 100))
    topic_olayer = model.add_parameters((ntopics, 100))
    
def build_tagging_graph(words, tags, builders, topic):
    dy.renew_cg()
    f_init, b_init = [b.initial_state() for b in builders]

    # embeddings
    wembs = [E[w] for w in words]
    wembs = [dy.noise(we,0.1) for we in wembs]

    # bilstm
    fw = [x.output() for x in f_init.add_inputs(wembs)]
    bw = [x.output() for x in b_init.add_inputs(reversed(wembs))]

    # MLP for tag prediction
    H = dy.parameter(pH)
    O = dy.parameter(pO)
    
    errs = []

    for f,b,t in zip(fw, reversed(bw), tags):
        f_b = dy.concatenate([f,b])
        r_t = O*(dy.tanh(H * f_b))
        # r_t = O * f_b
        err = dy.pickneglogsoftmax(r_t, t)
        errs.append(err)
    
    # add an extra layer with MLP to predict topic
    if TOPIC:
        # aux_layer = dy.reshape(dy.parameter(topic_olayer) * dy.parameter(topic_hlayer),(5000,1)) * f_b[-1]
        aux_layer = dy.parameter(topic_olayer) * (dy.tanh(dy.parameter(topic_hlayer)*f_b))
        aux_loss = dy.pickneglogsoftmax(aux_layer, topic)
        errs.append(aux_loss)
    return dy.esum(errs)

def tag_sent(sent, builders):
    dy.renew_cg()
    f_init, b_init = [b.initial_state() for b in builders]
    wembs = [E[vw.w2i.get(w, UNK)] for w,t in sent]

    fw = [x.output() for x in f_init.add_inputs(wembs)]
    bw = [x.output() for x in b_init.add_inputs(reversed(wembs))]

    H = dy.parameter(pH)
    O = dy.parameter(pO)

    tags=[]
    for f,b,(w,t) in zip(fw,reversed(bw),sent):
        r_t = O*(dy.tanh(H * dy.concatenate([f,b])))
        # r_t = O*dy.concatenate([f,b])
        out = dy.softmax(r_t)
        chosen = np.argmax(out.npvalue())
        tags.append(vt.i2w[chosen])
    return tags

tagged = loss = 0

for ITER in range(20):
    print("Iteration #:", ITER)
    random.seed(42)
    random.shuffle(train)
    random.seed(42)
    random.shuffle(topics)
    for i,s in enumerate(train,1):
        if i % 1000 == 0:
            trainer.status()
            print(loss / tagged)
            loss = 0
            tagged = 0
        if i % 1000 == 0:
            good = bad = 0.0
            for sent in test:
                tags = tag_sent(sent, builders)
                golds = [t for w,t in sent]
                for go,gu in zip(golds,tags):
                    if go == gu: good +=1 
                    else: bad+=1
            print(good/(good+bad))
        ws = [vw.w2i.get(w, UNK) for w,p in s]
        ps = [vt.w2i[p] for w,p in s]
        sum_errs = build_tagging_graph(ws,ps,builders,topics[train.index(s)])
        squared = -sum_errs# * sum_errs
        loss += sum_errs.scalar_value()
        tagged += len(ps)
        sum_errs.backward()
        trainer.update()

good = bad = 0.0
for sent in test:
    tags = tag_sent(sent, builders)
    golds = [t for w,t in sent]
    for go,gu in zip(golds,tags):
        if go == gu: good +=1 
        else: bad+=1
    print(good/(good+bad))
