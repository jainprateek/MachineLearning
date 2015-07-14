import numpy as np
import pandas as pd, os
import scipy.sparse as sps
import itertools

def one_hot_column(df, cols, vocabs):
    mats = []; df2 = df.drop(cols,axis=1)
    mats.append(sps.lil_matrix(np.array(df2)))
    for i,col in enumerate(cols):
        mat = sps.lil_matrix((len(df), len(vocabs[i])))
        for j,val in enumerate(np.array(df[col])):
            mat[j,vocabs[i][val]] = 1.
        mats.append(mat)

    res = sps.hstack(mats)   
    return res

data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
        'year': ['2000', '2001', '2002', '2001', '2002'],
        'pop': [1.5, 1.7, 3.6, 2.4, 2.9]}

df = pd.DataFrame(data)
print df

vocabs = []
vals = ['Ohio','Nevada']
vocabs.append(dict(itertools.izip(vals,range(len(vals)))))
vals = ['2000','2001','2002']
vocabs.append(dict(itertools.izip(vals,range(len(vals)))))

print vocabs

print one_hot_column(df, ['state','year'], vocabs).todense()