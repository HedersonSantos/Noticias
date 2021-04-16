# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 20:12:08 2021

@author: User
"""
import requests
import urllib.parse
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
from datetime import datetime
import csv, re

response = requests.get('https://g1.globo.com/')
content = response.content
site = BeautifulSoup(content,'html.parser')
#print(site.prettify())
noticias = site.findAll('div',attrs={'class':'feed-post-body'}) ## o método find traz para uma noticia
titulo,subtitulo, url, reportagem, categoria=[],[],[],[],[]
for index, noticia in enumerate(noticias):
    tit = noticia.find('a',attrs={'class':'feed-post-link'})
    titulo.append(tit.text)
    url.append(tit['href'])
    subtit=noticia.find('div',attrs={'class':'feed-post-body-resumo'})
    if subtit:
        subtitulo.append(subtit.text)
    else:
        subtitulo.append('')
    tmp_classe = urllib.parse.urlparse(tit['href']).path.split('/')[1:-1]
    classe=[]
    for cl in tmp_classe:
        padrao = re.match(r'\b(\d{4})\b|\b(\d{2})\b|\b(globo)\b|\b(g1)\b\
                          |\b(noticia)\b|globonews',cl )
        if padrao==None:
            classe.append(cl)
    
    categoria.append(classe)
    responseRep = requests.get(tit['href'])
    contentRep = responseRep.content
    siteRep = BeautifulSoup(contentRep,'html.parser')
    texto = siteRep.findAll('p')
    if texto:
        news=texto[0].text+'\n'
        for par in texto:
            news+=par.text + '\n'
        reportagem.append(news)
    else:
        reportagem.append('NA')

df = pd.DataFrame({'titulo':titulo,
                    'subtitulo':subtitulo,
                    'url':url,
                    'categoria':categoria,
                    'texto':reportagem})

arquivo = "e:/Hederson/MBA/tcc/g1/g1"+datetime.now().strftime('%Y%m%d_%H%M%S')+'.csv'
df.to_csv(arquivo, mode='w', sep = ';', encoding='utf-8',header=True, index=False, quotechar='"', quoting = csv.QUOTE_MINIMAL   )


