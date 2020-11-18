import os
import email
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import EnglishStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import numpy as np
import math
import sqlite3

path = '/Users/apple/Desktop/data'

def getfilelist(mail_dir, filelist):
    if os.path.isfile(mail_dir):
        filelist.append(mail_dir)
    elif os.path.isdir(mail_dir):
        for s in os.listdir(mail_dir):
            newdir = os.path.join(mail_dir, s)
            getfilelist(newdir, filelist)
    return filelist

file_list = getfilelist(path, [])

tokenizer = RegexpTokenizer(r'[A-Za-z][A-Za-z][A-Za-z]+')

es=EnglishStemmer()

forwardIndex = dict()
for i in range(0, len(file_list)):
    files = open(file_list[i], 'r')
    tokendict = dict()
    tempEmail = email.message_from_string(files.read())
    tempMessage = tempEmail.get_payload()
    tokens = list()
    tokens+=(tokenizer.tokenize(str(tempEmail['Subject'].lower())+str(tempMessage.lower())))
    tokens_withoout_Stopword = [w for w in tokens if not w in stopwords.words('english')]
    tokens_stemmed = [es.stem(word) for word in tokens_withoout_Stopword]
    for token in tokens_stemmed:
        if token in tokendict:
            tokendict[token] += 1
        else:
            tokendict[token] = 1
    forwardIndex[file_list[i]]=tokendict
    files.close()

# for i in range(len(forwardIndex)):
#     print(list(forwardIndex.keys())[i])
#     for j in range(len(list(forwardIndex.values())[i])):
#         print(list(list(forwardIndex.values())[i])[j])

invertIndex = dict()

word_tf_Index = dict()

for i in range(len(forwardIndex)):
    path = list(forwardIndex.keys())[i]
    dicts = list(forwardIndex.values())[i]
    for j in range(len(dicts)):
        words = list(dicts.keys())[j]
        times = list(dicts.values())[j]
        if words not in invertIndex:
            invertIndex[words] = dict()
        invertIndex[words][path] = times

for i in range(len(invertIndex)):
    times = 0
    for j in range(len(list(invertIndex.values())[i])):
        times += list(list(invertIndex.values())[i].values())[j]
    word_tf_Index[list(invertIndex.keys())[i]] = times

# print(word_tf_Index)
top_50 = list()
top_50 = sorted(word_tf_Index.items(), key = lambda kv :(kv[1], kv[0]))
# print(top_50)
# print(len(top_50))
top_50_word = list()
top_50_len = len(top_50)

for i in range(0, 50):
    top_50_word.append(list(top_50[top_50_len-i-1])[0])

# print(top_50_word)

invertIndex_top50 = dict()
for i in top_50_word:
    invertIndex_top50[i] = invertIndex.get(i)

top_50_lexorder = list()
top_50_lexorder = sorted(invertIndex_top50.items(), key = lambda kv :(kv[0], kv[1]))

# print(top_50_lexorder)
top_50_word_lexorder = list()
top_50_lexorder_len = len(top_50_lexorder)

for i in range(0, 50):
    top_50_word_lexorder.append(list(top_50_lexorder[i])[0])

# print(top_50_word_lexorder)

invertIndex_top50_lexorder = dict()
for i in top_50_word_lexorder:
    invertIndex_top50_lexorder[i] = invertIndex_top50.get(i)

# forwardIndex_top50 = dict()

# print(invertIndex_top50_lexorder)

path_column_Index = dict()

word_line_Index = dict()

word_df_Index = dict()

for i in range(len(forwardIndex)):
    path_column_Index[list(forwardIndex.keys())[i]] = i

for i in range(len(invertIndex_top50_lexorder)):
    word_line_Index[list(invertIndex_top50_lexorder.keys())[i]] = i

N = len(forwardIndex)

for word in invertIndex_top50_lexorder.keys():
    word_df_Index[word] = N / len(invertIndex_top50_lexorder.get(word))

# print(invertIndex_top50)

# print(sorted(word_tf_Index.items(), key = lambda kv :(kv[1], kv[0]))) #频率

# print(sorted(word_tf_Index.items(), key = lambda kv :(kv[0], kv[1]))) #字典序

# for i in range(len(invertIndex)):
#     print(list(invertIndex.keys())[i])
#     for j in range(len(list(invertIndex.values())[i])):
#         print(list(list(invertIndex.values())[i])[j])

line = len(invertIndex_top50_lexorder)
# 50
column = len(forwardIndex)
# 4
tf_idf = np.zeros((line, column))

for column_it in range(column):
    path = list(forwardIndex.keys())[column_it]
    word_tf = list(forwardIndex.values())[column_it]
    word_tf_list_in_top50 = list()
    for i in word_tf:
        word_tf_list_in_top50.append(i)

    word_tf_list_in_top50 = list(set(word_tf_list_in_top50).intersection(set(top_50_word_lexorder)))
    # print(word_tf_list_in_top50)
    # print(len(word_tf_list_in_top50))

    for word in word_tf_list_in_top50:
        lineinmatrix = word_line_Index.get(word)
        columninmatrix = column_it
        tf_idf[lineinmatrix][columninmatrix] = (1+math.log(word_tf.get(word), 10))*math.log(word_df_Index.get(word), 10)

# print(tf_idf)

con = sqlite3.connect("test_tf_idf_50.db")

cur = con.cursor()

sql = "CREATE TABLE IF NOT EXISTS test(path text, word text, line integer, td_idf real , primary key (path, word))"

cur.execute(sql)

# for i in range(len(invertIndex)):
#    for j in range(len(list(invertIndex.values())[i])):
#       print(tf_idf[word_line_Index.get(list(invertIndex.keys())[i])][path_column_Index.get(list(list(invertIndex.values())[i])[j])])


for i in range(len(invertIndex_top50_lexorder)):
   for j in range(len(list(invertIndex_top50_lexorder.values())[i])):
      cur.execute("INSERT OR IGNORE INTO test(path, word, line, td_idf) values(?, ?, ?, ?)",
                  (list(list(invertIndex_top50_lexorder.values())[i])[j], list(invertIndex_top50_lexorder.keys())[i], word_line_Index.get(list(invertIndex_top50_lexorder.keys())[i]), tf_idf[word_line_Index.get(list(invertIndex_top50_lexorder.keys())[i])][path_column_Index.get(list(list(invertIndex_top50_lexorder.values())[i])[j])]))

con.commit()

cur.close()
con.close()

con = sqlite3.connect("test_tf_idf_50.db")

cur = con.cursor()

# cur.execute("select * from test")
# print(cur.fetchall())

len_invertIndex_top50_lexorder = len(invertIndex_top50_lexorder)
path = "/Users/apple/Desktop/data/maildir/arnold-j/2000_conference/1"

def tf_idf_path(path, len_invertIndex_top50_lexorder):
    templist = list()
    line_last = 0
    for word in invertIndex_top50_lexorder.keys():
        for row in cur.execute("select * from test where test.path = '" + path + "' and test.word = '" + word + "'"):
            for i in range(line_last, row[2]-1):
                templist.append(0)
            templist.append(row[3])
            line_last = row[2]
    for i in range(line_last, len_invertIndex_top50_lexorder):
        templist.append(0)
    return templist

for i in forwardIndex:
    print(tf_idf_path(i, len_invertIndex_top50_lexorder))

cur.close()
con.close()