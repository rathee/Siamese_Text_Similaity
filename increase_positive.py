import numpy as np
import pandas as pd
from IPython.display import  display
from collections import defaultdict
from itertools import combinations
import sys
pd.set_option('display.max_colwidth',-1)
train_df=pd.read_csv(sys.argv[1])
ddf=train_df[train_df.is_duplicate==1]
negatives = train_df[train_df.is_duplicate == 0]
display(train_df.head(2))
print('Duplicated questions shape:',ddf.shape)

clean_ddf1=ddf[['qid1','question1']].drop_duplicates()
clean_ddf1.columns=['qid','question']
clean_ddf2=ddf[['qid2','question2']].drop_duplicates()
clean_ddf2.columns=['qid','question']
all_dqdf=clean_ddf1.append(clean_ddf2,ignore_index=True)
print 'all_dqdf shape ',(all_dqdf.shape)

dqids12=ddf[['qid1','qid2']]
display(dqids12.head(2))
df12list=dqids12.groupby('qid1', as_index=False)['qid2'].agg({'dlist':(lambda x: list(x))})
print 'df12list length', (len(df12list))
d12list=df12list.values
d12list=[[i]+j for i,j in d12list]
# get all the combinations of id, like (id1,id2)...
d12ids=set()
for ids in d12list:
    ids_len=len(ids)
    for i in range(ids_len):
        for j in range(i+1,ids_len):
            d12ids.add((ids[i],ids[j]))
print(len(d12ids))

dqids21=ddf[['qid2','qid1']]
display(dqids21.head(2))
df21list=dqids21.groupby('qid2', as_index=False)['qid1'].agg({'dlist':(lambda x: list(x))})
print '---------------------------'
print df21list.head(2)
print '---------------------------'
print(len(df21list))
ids2=df21list.qid2.values
d21list=df21list.values
print df21list.head(2)
d21list=[[i]+j for i,j in d21list]
print d21list[0]
d21ids=set()
for ids in d21list:
    ids_len=len(ids)
    for i in range(ids_len):
        for j in range(i+1,ids_len):
            d21ids.add((ids[i],ids[j]))
print len(d21ids)

dids=list(d12ids | d21ids)
print len(dids)

def indices_dict(lis):
    d = defaultdict(list)
    for i,(a,b) in enumerate(lis):
        d[a].append(i)
        d[b].append(i)
    return d
from collections import Counter
def disjoint_indices(lis):
    d = indices_dict(lis)
    print 'counter values',Counter(d).most_common(3)
    sets = []
    while len(d):
        que = set(d.popitem()[1])
        ind = set()
        while len(que):
            ind |= que 
            que = set([y for i in que 
                         for x in lis[i] 
                         for y in d.pop(x, [])]) - ind
        sets += [ind]
    return sets

def disjoint_sets(lis):
    return [set([x for i in s for x in lis[i]]) for s in disjoint_indices(lis)]

did_u=disjoint_sets(dids)
new_dids=[]
for u in did_u:
    new_dids.extend(list(combinations(u,2)))
print len(new_dids)

new_ddf=pd.DataFrame(new_dids,columns=['qid1','qid2'])
print('New duplicated shape:',new_ddf.shape)

new_ddf=new_ddf.merge(all_dqdf,left_on='qid1',right_on='qid',how='left')
new_ddf.drop('qid',inplace=True,axis=1)
new_ddf.columns=['qid1','qid2','question1']
new_ddf.drop_duplicates(inplace=True)
print(new_ddf.shape)

new_ddf=new_ddf.merge(all_dqdf,left_on='qid2',right_on='qid',how='left')
new_ddf.drop('qid',inplace=True,axis=1)
new_ddf.columns=['qid1','qid2','question1','question2']
new_ddf.drop_duplicates(inplace=True)
print(new_ddf.shape)

new_ddf['is_duplicate']=1

print(len(all_dqdf))
# after we generate more data, then the duplicated pairs count:
print(len(new_ddf))
print(len(negatives))
train_data = new_ddf.append(negatives, ignore_index = True)
train_data['question1'] = train_data['question1'].str.replace('\"','') 
train_data['question2'] = train_data['question2'].str.replace('\"','') 
print(len(train_data))
train_data = train_data.sample(frac=1)
train_data.to_csv(sys.argv[2], encoding='utf-8')
