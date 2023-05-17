#!/usr/bin/env python
# coding: utf-8
# Created by Mario Xerri, Maciej Wlodek, and Justin Wei
# In[38]:


import pandas as pd
import numpy as np

df1=pd.read_excel("data_set1.xlsx")
df2=pd.read_excel("data_set2.xlsx")


# In[39]:


print(df1)


# In[4]:


print(df2)


# In[25]:


products_sum = 0
j = 0
for i in range(df1.shape[0]):
    h = df1.iloc[i,0]
    k = df1.iloc[i,1]
    l = df1.iloc[i,2]
    print('{},{},{}; {}'.format(h,k,l, j))
    j_old = j
    while j < df2.shape[0]:
        h_prime = df2.iloc[j,0]
        k_prime = df2.iloc[j,1]
        l_prime = df2.iloc[j,2]
        if h == h_prime and k == k_prime and l == l_prime :
            products_sum = products_sum + df1.iloc[i,3] * df2.iloc[j,3]
            j = j+1
            break
        j = j+1
    if j == df2.shape[0]:
        j = j_old
print(products_sum)
    
#for i in range(3):
#    match = df.DataFrame()
#    for j in range(df1.shape[0]):
#        match[i].where(df2.iloc[])


# In[ ]:


match_lst


# In[44]:


hkl_1 = dict()
hkl_2 = dict()
for i in range(df1.shape[0]):
    h = df1.iloc[i,0]
    k = df1.iloc[i,1]
    l = df1.iloc[i,2]
    indices = '{},{},{}'.format(h,k,l)
    if indices not in hkl_1:
        hkl_1[indices] = (df1.iloc[i,3])
    #if indices in hkl_1:
        #hkl_1[indices] = [hkl_1[indices], df1.iloc[i,3]]
    #    hkl_1[indices].append(df1.iloc[i,3])
    #else :
    #    hkl_1[indices] = (df1.iloc[i,3])
for i in range(df2.shape[0]):
    h = df2.iloc[i,0]
    k = df2.iloc[i,1]
    l = df2.iloc[i,2]
    indices = '{},{},{}'.format(h,k,l)
    if indices not in hkl_2:
        hkl_2[indices] = (df2.iloc[i,3])
#    if indices in hkl_2:
#        #hkl_2[indices] = [hkl_2[indices], df2.iloc[i,3]]
#        hkl_2[indices].append(df2.iloc[i,3])
#    else :
#        hkl_2[indices] = numpy.linspace(df2.iloc[i,3])
sum = 0
num_matches = 0
norm_a = 0
norm_b = 0
for key in hkl_1:
    if key in hkl_2:
        num_matches = num_matches+1
        sum = sum + hkl_1[key] * hkl_2[key]
        norm_a = norm_a + hkl_1[key]*hkl_1[key]
        norm_b = norm_b + hkl_2[key]*hkl_2[key]
norm_a = np.sqrt(norm_a)
norm_b = np.sqrt(norm_b)
print('norm_a = {}'.format(norm_a))
print('norm_b = {}'.format(norm_b))
print('sum = {}'.format(sum))
correlation_coefficient = sum/(norm_a*norm_b)
print('cc = {}'.format(correlation_coefficient))


penalty_a = 1/num_matches*(df1.shape[0] - num_matches)
penalty_b = 1/num_matches*(df2.shape[0] - num_matches)
distance = 1-correlation_coefficient + penalty_a + penalty_b
print('penalty_a = {}'.format(penalty_a))
print('penalty_b = {}'.format(penalty_b))
print('distance = {}'.format(distance))


# In[ ]:




