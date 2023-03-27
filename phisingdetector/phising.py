from flask import Flask, request, jsonify
import lightgbm as lgb
from lightgbm import *
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as metrics
from sklearn.preprocessing import LabelEncoder
import os.path
from tld import get_tld
from googlesearch import search
# import googlesearch
from urllib.parse import urlparse
import re
import io
import pandas as pd
import itertools
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from lightgbm import LGBMClassifier
import os
import seaborn as sns
from wordcloud import WordCloud
import pickle
# df = pd.read_csv('malicious_phish.csv')

# df.head()
# urls=eval(input("Enter the links : "))
# urls="asaadasf"
# urls=["www.google.com"]
df = pd.read_csv('malicious_phish.csv')
# print(df.shape)

# Use of IP or not in domain


def having_ip_address(url):
    match = re.search(
        '(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.'
        '([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|'  # IPv4
        # IPv4 in hexadecimal
        '((0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\/)'
        '(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}', url)  # Ipv6
    if match:
        # print match.group()
        return 1
    else:
        # print 'No matching pattern found'
        return 0


df['use_of_ip'] = df['url'].apply(lambda i: having_ip_address(i))


def abnormal_url(url):
    hostname = urlparse(url).hostname
    hostname = str(hostname)
    match = re.search(hostname, url)
    if match:
        # print match.group()
        return 1
    else:
        # print 'No matching pattern found'
        return 0


df['abnormal_url'] = df['url'].apply(lambda i: abnormal_url(i))

# pip install googlesearch-python


def google_index(url):
    site = search(url, 5)
    return 1 if site else 0


df['google_index'] = df['url'].apply(lambda i: google_index(i))


def count_dot(url):
    count_dot = url.count('.')
    return count_dot


df['count.'] = df['url'].apply(lambda i: count_dot(i))


def count_www(url):
    url.count('www')
    return url.count('www')


df['count-www'] = df['url'].apply(lambda i: count_www(i))


def count_atrate(url):

    return url.count('@')


df['count@'] = df['url'].apply(lambda i: count_atrate(i))


def no_of_dir(url):
    urldir = urlparse(url).path
    return urldir.count('/')


df['count_dir'] = df['url'].apply(lambda i: no_of_dir(i))


def no_of_embed(url):
    urldir = urlparse(url).path
    return urldir.count('//')


df['count_embed_domian'] = df['url'].apply(lambda i: no_of_embed(i))


def shortening_service(url):
    match = re.search('bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
                      'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
                      'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
                      'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|'
                      'db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|'
                      'q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|'
                      'x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|'
                      'tr\.im|link\.zip\.net',
                      url)
    if match:
        return 1
    else:
        return 0


df['short_url'] = df['url'].apply(lambda i: shortening_service(i))


def count_https(url):
    return url.count('https')


df['count-https'] = df['url'].apply(lambda i: count_https(i))


def count_http(url):
    return url.count('http')


df['count-http'] = df['url'].apply(lambda i: count_http(i))


def count_per(url):
    return url.count('%')


df['count%'] = df['url'].apply(lambda i: count_per(i))


def count_ques(url):
    return url.count('?')


df['count?'] = df['url'].apply(lambda i: count_ques(i))


def count_hyphen(url):
    return url.count('-')


df['count-'] = df['url'].apply(lambda i: count_hyphen(i))


def count_equal(url):
    return url.count('=')


df['count='] = df['url'].apply(lambda i: count_equal(i))


def url_length(url):
    return len(str(url))


# Length of URL
df['url_length'] = df['url'].apply(lambda i: url_length(i))
# Hostname Length


def hostname_length(url):
    return len(urlparse(url).netloc)


df['hostname_length'] = df['url'].apply(lambda i: hostname_length(i))

df.head()


def suspicious_words(url):
    match = re.search('PayPal|login|signin|bank|account|update|free|lucky|service|bonus|ebayisapi|webscr',
                      url)
    if match:
        return 1
    else:
        return 0


df['sus_url'] = df['url'].apply(lambda i: suspicious_words(i))


def digit_count(url):
    digits = 0
    for i in url:
        if i.isnumeric():
            digits = digits + 1
    return digits


df['count-digits'] = df['url'].apply(lambda i: digit_count(i))


def letter_count(url):
    letters = 0
    for i in url:
        if i.isalpha():
            letters = letters + 1
    return letters


df['count-letters'] = df['url'].apply(lambda i: letter_count(i))

# pip install tld


# First Directory Length


def fd_length(url):
    urlpath = urlparse(url).path
    try:
        return len(urlpath.split('/')[1])
    except:
        return 0


df['fd_length'] = df['url'].apply(lambda i: fd_length(i))

# Length of Top Level Domain
df['tld'] = df['url'].apply(lambda i: get_tld(i, fail_silently=True))


def tld_length(tld):
    try:
        return len(tld)
    except:
        return -1

df['tld_length'] = df['tld'].apply(lambda i: tld_length(i))


# lb_make = LabelEncoder()
# df["type_code"] = lb_make.fit_transform(df["type"])
# X = df[['use_of_ip', 'abnormal_url', 'count.', 'count-www', 'count@',
#        'count_dir', 'count_embed_domian', 'short_url', 'count-https',
#         'count-http', 'count%', 'count?', 'count-', 'count=', 'url_length',
#         'hostname_length', 'sus_url', 'fd_length', 'tld_length', 'count-digits',
#         'count-letters']]

# # Target Variable
# y = df['type_code']
# X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, shuffle=True, random_state=5)
# rf = RandomForestClassifier(n_estimators=100, max_features='sqrt')
# rf.fit(X_train, y_train.values.ravel())
# y_pred_rf = rf.predict(X_test)
# print(classification_report(y_test, y_pred_rf, target_names=[
#       'benign', 'defacement', 'phishing', 'malware']))
# score = metrics.accuracy_score(y_test, y_pred_rf)
# print("accuracy:   %0.3f" % score)
# knn = rf.fit(X_train, y_train)

# feat_importances = pd.Series(rf.feature_importances_, index=X_train.columns)
# feat_importances.sort_values().plot(kind="barh", figsize=(10, 6))


def main(url):

    status = []

    status.append(having_ip_address(url))
    status.append(abnormal_url(url))
    status.append(count_dot(url))
    status.append(count_www(url))
    status.append(count_atrate(url))
    status.append(no_of_dir(url))
    status.append(no_of_embed(url))

    status.append(shortening_service(url))
    status.append(count_https(url))
    status.append(count_http(url))

    status.append(count_per(url))
    status.append(count_ques(url))
    status.append(count_hyphen(url))
    status.append(count_equal(url))

    status.append(url_length(url))
    status.append(hostname_length(url))
    status.append(suspicious_words(url))
    status.append(digit_count(url))
    status.append(letter_count(url))
    status.append(fd_length(url))
    tld = get_tld(url, fail_silently=True)

    status.append(tld_length(tld))
    # print(status)
    return status

def get_prediction_from_url(test_url):
    features_test = main(test_url)
    # clf = lgb.LGBMClassifier()
    # clf.fit(X_train, y_train)
    # knn = clf.fit(X_train, y_train)
    # print(clf)
    # features_test = np.array(features_test).reshape((1, -1))
    
    features_test = np.reshape(features_test, (1, -1))
    # print(features_test)
    # features_test= clf.predict(features_test)
    
    # print(features_test)


# Save the trained model as a pickle string.
# saved_model = pickle.dumps(knn)
    file = open('./model_pickle', 'rb')
    # response = pickle.load(file)
# file.close()
# Load the pickled model
    knn_from_pickle = pickle.load(file)

    features_test = knn_from_pickle.predict(
    features_test)
    if int(features_test[0]) == 0:

        res = "SAFE"
        return res
    elif int(features_test[0]) == 1.0:

        res = "DEFACEMENT"
        return res
    elif int(features_test[0]) == 2.0:
        res = "PHISHING"
        return res

    elif int(features_test[0]) == 3.0:

        res = "MALWARE"
        return res



# y_pred = clf.predict([[0, 1, 4, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 46, 23, 1, 8, 30, 14, 4]

#                       ])
# print(y_pred)
urls = ["https://www.google.com"]

for url in urls:
    print(get_prediction_from_url(str(url)))

# with open('model_pickle1','wb') as f:
    # pickle.dump(clf,f)
    
#     # Save the trained model as a pickle string.
    # saved_model = pickle.dumps(knn,f)


