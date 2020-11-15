import os
# import sys
import email
# import nltk
from nltk.stem.snowball import EnglishStemmer
# from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import numpy as np
import math
import sqlite3


# path = '/Users/apple/Desktop/data/maildir/arnold-j/active_international'
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

forwardIndex = dict()
i = 0

for route in file_list:
    dictionary = dict()
    myFile = open(route, 'r', encoding='utf-8')
    emailData = myFile.read()
    tempEmail = email.message_from_string(emailData)

    tempSubject = tempEmail['subject']
    tempSubject = str(tempSubject)

    tempMessage = tempEmail.get_payload(decode=True)
    tempMessage = str(tempMessage)

    if tempEmail.is_multipart():
        for part in tempEmail.walk():
            ctype = part.get_content_type()
            cdispo = str(part.get('Content-Disposition'))

            if ctype == 'text/plain' and 'attachment' not in cdispo:
                tempMessage = part.get_payload(decode=False)
                break
    else:
        tempMessage = tempEmail.get_payload(decode=False)

    tempSubject = tempSubject.lower()
    tempMessage = tempMessage.lower()

    es = EnglishStemmer()
    dist = re.sub(r'[^a-zA-Z]', " ", tempMessage)
    dis = word_tokenize(dist)

    for token in dis:

        if token in stopwords.words('english'):
            continue

        else:
            if token in dictionary:
                dictionary[token] += 1
            else:
                dictionary[token] = 1

    forwardIndex[route] = dictionary
    i = i + 1

invertIndex = dict()

path_column_Index = dict()

word_line_Index = dict()

word_df_Index = dict()


for i in range(len(forwardIndex)):
    path_column_Index[list(forwardIndex.keys())[i]] = i

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
    word_line_Index[list(invertIndex.keys())[i]] = i

N = len(forwardIndex)

for word in invertIndex.keys():
    word_df_Index[word] = N / len(invertIndex.get(word))

line = len(invertIndex)
# 238
column = len(forwardIndex)
# 4
tf_idf = np.zeros((line, column))

for column_it in range(column):
    path = list(forwardIndex.keys())[column_it]
    word_tf = list(forwardIndex.values())[column_it]
    for word in word_tf.keys():
        lineinmatrix = word_line_Index.get(word)
        columninmatrix = column_it
        tf_idf[lineinmatrix][columninmatrix] = (1+math.log(word_tf.get(word), 2))*math.log(word_df_Index.get(word), 10)

# print(tf_idf)

# path1 = '/Users/apple/Desktop/data/maildir/arnold-j/2000_conference/1'
# word1 = 'jennifer'
# print(tf_idf[word_line_Index.get(word1)][path_column_Index.get(path1)])

# for i in range(len(invertIndex)):
#    for j in range(len(list(invertIndex.values())[i])):
#       print(list(invertIndex.keys())[i])
#       print(list(list(invertIndex.values())[i])[j])

con = sqlite3.connect("invertIndex.db")

cur = con.cursor()

sql = "CREATE TABLE IF NOT EXISTS test(word text, id integer, path text , primary key (word, id))"

cur.execute(sql)

for i in range(len(invertIndex)):
   for j in range(len(list(invertIndex.values())[i])):
      cur.execute("INSERT OR IGNORE INTO test(word, id, path) values(?, ?, ?)", (list(invertIndex.keys())[i], j, list(list(invertIndex.values())[i])[j]))

con.commit()

cur.close()
con.close()

con = sqlite3.connect("invertIndex.db")

cur = con.cursor()

for row in cur.execute("select * from test where test.word = 'jennifer'"):
   print(row[2])

# cur.execute("select * from test")
# print(cur.fetchall())

cur.close()
con.close()

con = sqlite3.connect("tf_idf.db")

cur = con.cursor()

sql = "CREATE TABLE IF NOT EXISTS test(path text, word text, td_idf real , primary key (path, word))"

cur.execute(sql)

# for i in range(len(invertIndex)):
#    for j in range(len(list(invertIndex.values())[i])):
#       print(tf_idf[word_line_Index.get(list(invertIndex.keys())[i])][path_column_Index.get(list(list(invertIndex.values())[i])[j])])


for i in range(len(invertIndex)):
   for j in range(len(list(invertIndex.values())[i])):
      cur.execute("INSERT OR IGNORE INTO test(path, word, td_idf) values(?, ?, ?)",
                  (list(list(invertIndex.values())[i])[j], list(invertIndex.keys())[i], tf_idf[word_line_Index.get(list(invertIndex.keys())[i])][path_column_Index.get(list(list(invertIndex.values())[i])[j])]))

con.commit()

cur.close()
con.close()

con = sqlite3.connect("tf_idf.db")

cur = con.cursor()

cur.execute("select * from test")
print(cur.fetchall())

cur.close()
con.close()