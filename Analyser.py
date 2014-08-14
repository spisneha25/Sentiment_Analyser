import nltk
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()

poslines= open(r'rt-polarity.pos', 'r').read().splitlines()
neglines= open(r'rt-polarity.neg', 'r').read().splitlines()
stop = open('stopverbs.txt', 'r').read().splitlines()

N= 4000
N1 = N
poslinesTrain= poslines[:N]
neglinesTrain= neglines[:N]
poslinesTest= poslines[N1:]
neglinesTest= neglines[N1:]
trainset= [(x,1) for x in poslinesTrain] + [(x,-1) for x in neglinesTrain]
testset= [(x,1) for x in poslinesTest] + [(x,-1) for x in neglinesTest]

posverbs = {}
negverbs = {}

for l, label in trainset:
    t = nltk.word_tokenize(l)
    tt = nltk.pos_tag(t)
    for pos in tt:
        if pos[0] in stop:
            continue
        for p in pos[1]:
            if(p == 'V'):
                w = pos[0]
                word = wnl.lemmatize(w, 'v')
                if label == 1:
                    if posverbs.has_key(word):
                        posverbs[word] = posverbs[word] + 1
                    else:
                        posverbs[word] = 1
                    ws = wn.synsets(w)
                    for s in ws:
                        for l in s.lemmas:
                            lname = wnl.lemmatize(l.name, 'v')
                            if posverbs.has_key(lname):
                                    posverbs[lname] = posverbs[lname] + 1
                            else:
                                    posverbs[lname] = 1
                            for ant in l.antonyms():
                                aname = wnl.lemmatize(ant.name, 'v')
                                if negverbs.has_key(aname):
                                    negverbs[aname] = negverbs[aname] + 1
                                else:
                                    negverbs[aname] = 1

                else:
                    if negverbs.has_key(word):
                        negverbs[word] = negverbs[word] + 1
                    else:
                        negverbs[word] = 1
                    ws = wn.synsets(w)
                    for s in ws:
                        for l in s.lemmas:
                            lname = wnl.lemmatize(l.name, 'v')
                            if negverbs.has_key(lname):
                                    negverbs[lname] = negverbs[lname] + 1
                            else:
                                    negverbs[lname] = 1                    
                            for ant in l.antonyms():
                                aname = wnl.lemmatize(ant.name, 'v')
                                if posverbs.has_key(aname):
                                    posverbs[aname] = posverbs[aname] + 1
                                else:
                                    posverbs[aname] = 1

print(len(posverbs))
print(len(negverbs))
wrong = 0
for l, label in testset:
    totpos, totneg = 0.0, 0.0
    t = nltk.word_tokenize(l)
    tt = nltk.pos_tag(t)
    for pos in tt:
        if pos[0] in stop:
            print(pos[0])
            continue
        for p in pos[1]:
            if(p == 'V'):
                w = pos[0]
                word = wnl.lemmatize(w, 'v')
                a = posverbs.get(word, 0.0) + 1.0
                b = negverbs.get(word, 0.0) + 1.0
                totpos+=a/(a+b)
                totneg+=b/(a+b)
    prediction = 1
    if totneg > totpos: prediction = -1

    if prediction != label :
       wrong += 1
       print 'ERROR: %s posscore=%.2f negscore=%.2f' % (l, totpos, totneg)
    else:
       print 'CORRECT: %s posscore=%.2f negscore=%.2f' % (l, totpos, totneg)
print 'Stop error rate is %f' % (1.0*wrong/len(testset),)

