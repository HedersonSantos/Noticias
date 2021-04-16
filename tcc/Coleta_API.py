# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 18:51:51 2021

@author: User
"""
import pandas as pd
import urllib.parse
import requests,  re, time
#from urllib3 import Request, urlopen
import json,csv, glob, os
from datetime import datetime
from bs4 import BeautifulSoup
pd.set_option('display.max_colwidth', None)

def extrairNoticiaporApi():

    lsturl = ["https://newsapi.org/v2/top-headlines?sources=google-news-br&apiKey=e9c4accee8ab4ea8a0266a9f81edf869",
          'https://gnews.io/api/v4/search?q=covid&token=9456eed34bcc0251a3b4752f95e1643d&country=br',
          'https://gnews.io/api/v4/search?q=bolsonaro&token=9456eed34bcc0251a3b4752f95e1643d&country=br',
          'https://gnews.io/api/v4/search?q=congresso&token=9456eed34bcc0251a3b4752f95e1643d&country=br',
          'https://gnews.io/api/v4/search?q=stf&token=9456eed34bcc0251a3b4752f95e1643d&country=br',
          'https://gnews.io/api/v4/search?q=futebol&token=9456eed34bcc0251a3b4752f95e1643d&country=br',
          'https://gnews.io/api/v4/search?q=esporte&token=9456eed34bcc0251a3b4752f95e1643d&country=br',
          'https://gnews.io/api/v4/search?q=tokyo&token=9456eed34bcc0251a3b4752f95e1643d&country=br',
          'https://gnews.io/api/v4/search?q=olimpíada&token=9456eed34bcc0251a3b4752f95e1643d&country=br',
          'https://gnews.io/api/v4/search?q=politica&token=9456eed34bcc0251a3b4752f95e1643d&country=br',
          'https://gnews.io/api/v4/search?q=bbb&token=9456eed34bcc0251a3b4752f95e1643d&country=br',
          'https://gnews.io/api/v4/search?q=novela&token=9456eed34bcc0251a3b4752f95e1643d&country=br'
          'https://gnews.io/api/v4/search?q=ciência&token=9456eed34bcc0251a3b4752f95e1643d&country=br',
          'https://gnews.io/api/v4/search?q=tecnologia&token=9456eed34bcc0251a3b4752f95e1643d&country=br'
           ]


    url = 'https://gnews.io/api/v4/search?q=covid&token=9456eed34bcc0251a3b4752f95e1643d&country=br'
    for index,url in enumerate(lsturl): 
        r = requests.get(url)
        if r.status_code ==200:
            dados = r.json()
            if not 'error' in dados.keys():
                dfDados = pd.DataFrame(dados)
                frame = []
                for art in dfDados['articles']:
                    frame.append(pd.DataFrame(art))
                dfNews = pd.concat(frame, sort=False)
                dfNews = dfNews.reset_index()
                del(dfNews['index'])
                dfNews = dfNews.drop_duplicates(subset='title', keep="first")
                dfNewsTexto = extraiUrlHtml(dfNews.copy())
                categoria=[]
                for link in dfNewsTexto['url'].tolist(): 
                    tmp_classe = urllib.parse.urlparse(link).path.split('/')[1:-1]
                    classe=[]
                    for cl in tmp_classe:
                        padrao = re.match(r'\b(\d{4})\b|\b(\d{2})\b|\b(globo)\b|\b(g1)\b\
                                          |\b(noticia)\b|globonews',cl )
                        if padrao==None:
                            classe.append(cl)
                    categoria.append(classe)
                dfNewsTexto['categoria']=categoria
                arquivo = "e:/Hederson/MBA/tcc/News"+datetime.now().strftime('%Y%m%d_%H%M%S')+str(index)+'.csv'
                dfNewsTexto.to_csv(arquivo, mode='w', sep = ';', encoding='utf-8',header=True, index=False, quotechar='"', quoting = csv.QUOTE_MINIMAL   )
                #return dfNewsTexto

def extraiUrlHtml(dfNews):
    reportagem=[]
    for index in dfNews.index:
        url = dfNews['url'][index]
        print(url)
        responseRep = requests.get(url,allow_redirects=False)
        if responseRep.status_code == 200:
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
        else:
            reportagem.append('NA')
    dfNews['Texto']=reportagem    
    return dfNews

def extraiUrlHtmlArquivos():
    diretorio = "e:/Hederson/MBA/tcc/"
    arquivos = os.listdir(diretorio)
    arquivoscsv = [file for file in arquivos if file.endswith('.csv')]
    #arquivo = arquivoscsv[0]
    for arquivo in arquivoscsv:
        print('Arquivo:',arquivo)
        dfNews = pd.read_csv(arquivo, sep=';').reset_index()
        reportagem=[]
        #index=1
        for index in dfNews.index:
            #url='https://www.poder360.com.br/governo/ouca-e-leia-a-integra-do-que-disse-bolsonaro-a-kajuru-sobre-a-cpi-da-covid/'
            url = dfNews['url'][index]
            print(url)
            responseRep = requests.get(url,allow_redirects=False)
            if responseRep.status_code == 200:
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
            else:
                reportagem.append('NA')
        dfNews['Texto']=reportagem    
        dfNews.to_csv(diretorio+'/textos/'+arquivo, mode='w', sep = ';', encoding='utf-8',header=True, index=False, quotechar='"', quoting = csv.QUOTE_MINIMAL   )
        os.renames(arquivo, arquivo+'_ok')
def extraiUrlHtmlSelenium():
    from selenium import webdriver
    search_query = 'https://g1.globo.com/politica/noticia/2021/04/12/rosa-weber-suspende-trechos-dos-decretos-de-armas-de-bolsonaro-que-entram-em-vigor-nesta-terca.ghtml'
    driver = webdriver.Chrome(executable_path='e:/webdriver/chromedriver.exe')
    job_details = []
    
    driver.get(search_query)
    time.sleep(10)
    job_list = driver.find_elements_by_xpath(".//p.content-text_container")
    
    
df = extrairNoticiaporApi()
#extraiUrlHtml()
#extraiUrlHtmlArquivos()