#@title Process URL content
import numpy as np
import pandas as pd
import urllib.request 
import ssl
from bs4 import BeautifulSoup
from urllib.error import URLError, HTTPError
from socket import error as SocketError
import re
from http.client import IncompleteRead


def load_data(train_filepath):
    x_y_train = pd.read_csv(train_filepath).as_matrix()
#     print(type(x_y_train))

    return x_y_train
  

def read_url(fetch_url):
    
    ###################### read content of the page for each URL ##########################
    
    new_text = None
    text = None
    context = ssl._create_unverified_context()
  
    try: 
        content = urllib.request.urlopen(fetch_url, context=context, timeout=10)
        html = content.read()
        soup = BeautifulSoup(html, 'html.parser', from_encoding="iso-8859-1")
        text = ''
        for tag in soup.findAll('p'):
            text += tag.getText()

        if text:
            new_text = text.strip().replace("\n", " ")
            new_text = new_text.strip().replace("\t", " ")
            new_text = new_text.strip().replace("\r", " ")
            new_text = re.sub(r'[^a-zA-Z0-9 ]',r'',new_text)
            new_text = re.sub(r'[-()\"#/@;:<>{}`+=~|.!?,''$\]]', ' ', new_text)

            list_item = [str(x) for x in new_text.split() if len(x) > 30]
#             print(list_item)
            for item in list_item:
                new_text = new_text.strip().replace(item, " ")
            
    except URLError:
        error = 'URLError'
    except HTTPError:
        error = "HTTPError"
    except SocketError:
        error = "SocketError"
    except IncompleteRead:
        error = "IncompleteRead"
#     print(new_text)
    return new_text

def pre_process(X):
    
    urls = X[:,1:3]
    ############################# process documents ###########################################
    ######### get content from webpage if successfully fetched. Else use title ################
    
    for i in range(len(urls)):
        url = urls[i]
#         print("process URL:",url)
        content = read_url(url[1])
        title = re.sub(r'[-()\"#/@;:<>{}`+=~|.!?,''$\]]', ' ', url[0].lower())
#         print(content)
        if content:
            if content.strip():
                X[i,2] = title +' '+ content.strip().lower()
            else:
                X[i,2] = title
        else:
            X[i,2] = title  
        
    return X


if __name__=='__main__':
    
    ################# save files ##################
    
    x_train = load_data('../data/train_v2.csv')
    x_content = pre_process(x_train)
    print(x_content.shape)
    df = pd.DataFrame(x_content, columns=['article_id', 'title', 'content','publisher','hostname','timestamp','category'])

    df.to_csv('train_v2_content.csv', index=False)
    
    x_train = load_data('../data/test_v2.csv')
    x_content = pre_process(x_train)
    print(x_content.shape)
    df = pd.DataFrame(x_content, columns=['article_id', 'title', 'content','publisher','hostname','timestamp'])

    df.to_csv('test_v2_content.csv', index=False)
