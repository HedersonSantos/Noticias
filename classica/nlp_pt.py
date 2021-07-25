#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 10:46:47 2021

@author: hederson
"""

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

import nltk, re
from nltk import word_tokenize
from nltk.corpus import stopwords



from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import precision_score, recall_score, classification_report, accuracy_score, f1_score
from sklearn.metrics import confusion_matrix

class nlp_pt():
    
    def download_pt_stopWords():
        nltk.download('stopwords')
        
    def removeStopWords(self, texto, excluirWords:list=None):
        naoQueridas = nltk.corpus.stopwords.words('portuguese')
        naoQueridas.extend(excluirWords)
        naoQueridas = list(set(naoQueridas))
        palavras = [i for i in texto.split() if not i.lower() in naoQueridas]
        return (" ".join(palavras))
    def aplicaStemming(self, texto):
        stemmer = nltk.stem.RSLPStemmer()
        palavras = []
        for w in texto.split():
            palavras.append(stemmer.stem(w))
        return (" ".join(palavras))
    
    def removeCaracteresNaoDesejados(self,texto):
        textoLimpo = re.sub(r"http\S+", "", texto)
        textoLimpo = re.sub(r"www\..+\..+", "", texto)
        textoLimpo = re.sub(r"[^a-zA-ZáÁéÉíÍóÓúÚãÃàÀôâÂêÊôÔçÇ!,:.; ]", "", texto)
        
        return textoLimpo
    
    def retornaVetorizacao(self,X,pct_min=1, pct_max=1, excluirSW:list=None):
        count_vect = CountVectorizer(min_df=pct_min, max_df=pct_max, lowercase=True,stop_words=stopwords) 
        matriz_sparsa = count_vect.fit_transform(X)
        vocabulario = count_vect.fit(X)
        return [matriz_sparsa,count_vect]
    
    def limpaMaisEmenosFrequentes(self, matriz_sparsa, objetoVetor):
        dfBoW = pd.SparseDataFrame(matriz_sparsa,columns=objetoVetor.get_feature_names())
        frequency = dfBoW.sum(axis=0) # conta o número de vezes que cada palavra aparece no corpus
        smaller = 1.0               # palavras que aparecem "smaller" ou menos vezes serão removidas
        larger = 0.7*dfBoW.shape[0]   # palavras que aparecem "larger" ou mais vezes serão removidas

        s_mask = (frequency <= smaller) | (frequency >= larger)  # mascara boleana indicando as palavras
        words_to_remove = frequency[s_mask].index.values         # que serão removidas

        m = dfBoW.shape[1]
        dfBoW = dfBoW.drop(words_to_remove, axis=1)  # remove as colunas do DataFrame correspondentes 
                                         # as palavras que devem ser removidas
        print('Foram removidas',m-dfBoW.shape[1],'palavras')
        V = dfBoW.values
        return V
        
    
    def retornaMatrizItfIdf(self, V):
        tfidf_transformer = TfidfTransformer()
        matriz_tfidf = tfidf_transformer.fit_transform(V)
        return matriz_tfidf
    
    def imprimeGraficosFrequenciaPCA(self, X, nro_dim):
        X_norm = StandardScaler(with_mean=False).fit_transform(X)
        data_pca= TruncatedSVD(nro_dim)
        data_transformed = data_pca.fit(X_norm)
        plt.rcParams['figure.figsize'] = [12, 6]
        f, (ax1, ax2)  = plt.subplots(1,2)
        ax1.plot(data_pca.explained_variance_ratio_, '*') 
        ax2.plot(data_pca.explained_variance_ratio_[:20], '*')
        plt.show()
        return data_pca.explained_variance_ratio_
    
    def reduzDimensionalidadecomPCA(self, X, nro_dimensao,UT=None):
        X_norm = StandardScaler(with_mean=False).fit_transform(X)
        data_pca= TruncatedSVD(nro_dimensao)
        if UT==None:
            UT = data_pca.fit(X_norm)
        X_pca =  UT.fit_transform(X_norm)
        
        return [UT,X_pca]
    
    def imprimeGraficosdosGrupos(self,Xpca, c=None):
        fig, ax = plt.subplots()
        ax.set_xlim(-5,5)
        ax.set_ylim(-10,10)
        plt.scatter(Xpca[:,0],Xpca[:,1], c=c) 
        plt.show()


    
    def treinaNaiveBayes(self, X_train, y_train):
        model = GaussianNB().fit(X_train, y_train) #usar o Gaussiano, pois o multinomial não aceita valores negativos gerados pelo pca.
        return model
        
    def treinaSVM(self, X_train, y_train):
        SVM = svm.SVC(C=1.0, kernel='linear', gamma='auto', max_iter=-1)
        return SVM.fit(X_train,y_train)
        
    
    def treinaRandomForest(self, X_train, y_train):
        model = RandomForestClassifier(n_estimators= 100,
                             #min_samples_split = 2,
                             #min_samples_leaf = 6,
                             max_features = 'auto',
                             #max_depth = 0,
                             bootstrap= True)
        return model.fit(X_train, y_train);
        
    def treinaKnn(self, X_train, y_train,k=5):
        model = KNeighborsClassifier(n_neighbors=k, metric = 'euclidean')
        return model.fit(X_train,y_train)
    
    def avaliaModeloClassificacao(self, model, X_test,y_test):
        y_pred = model.predict(X_test)
        
        confusion_matrix(y_pred, y_test)
        print('matriz confusão: \n', pd.crosstab(y_pred, y_test, rownames=['True'], colnames=['Predicted'], margins=True))
        
        print('Accuracy:', accuracy_score(y_pred, y_test))
        print('F1 score:', f1_score(y_pred, y_test, average="macro"))
        print('Precision:', precision_score(y_pred, y_test, average="macro"))
        print('Recall:', recall_score(y_pred, y_test, average="macro"))
        print('\n clasification report:\n', classification_report(y_pred, y_pred))
        return 'fim'
    
    def retornaDFcomMatrizSparsa_Vocabulario(self,vetores:list):
        ''' vetores é a lista de saída do método retornaVetorizacao que contém
          matriz_sparsa e vocabulário'''
        df = pd.SparseDataFrame(vetores[0],columns=vetores[1].get_feature_names())
        vocabulario = vetores[1].vocabulary_
        dfV = pd.DataFrame(list(vocabulario.items()),columns=['termo','frequencia']) 
        dfV = dfV.sort_values("frequencia", ascending=False)
        return [df, dfV]
    
    def fluxoProcessamento(self):
        inicio = datetime.now()
        dfDados = self.obterAmostra('2021-01-01 00:00:00','2021-01-30 00:00:00',True,2000)
        dfDados = dfDados[~dfDados['TEXTO_LIMPO'].isnull()]
        #dfG = dfDados.groupby('TIPO')['CODIGO'].count().reset_index()
        #dfG.hist()
        dfDados['TEXTO_LIMPO'] = dfDados['TEXTO_LIMPO'].astype('unicode')
        dfDados['TEXTO_LIMPO'] = dfDados.loc[:,['TEXTO_LIMPO']].apply(lambda x: self.removeCaracteresNaoDesejados(x['TEXTO_LIMPO']),axis=1)
        dfDados['TEXTO_LIMPO'] = dfDados.loc[:,['TEXTO_LIMPO']].apply(lambda x: self.removeStopWords(x['TEXTO_LIMPO']),axis=1)
        dfDados['TEXTO_LIMPO'] = dfDados.loc[:,['TEXTO_LIMPO']].apply(lambda x: self.aplicaStemming(x['TEXTO_LIMPO']),axis=1)
        X = dfDados['TEXTO_LIMPO']
        y = dfDados['TIPO']
        X_train, X_test, y_train, y_test = train_test_split(X,y , test_size=0.2)
        print('X_train:',X_train.shape,'y_train:',y_train.shape,'X_test:',X_test.shape,'y_test:',y_test.shape)
        vetores = self.retornaVetorizacao(X_train)
        V = vetores[0]
        
        #df = pd.SparseDataFrame(V,columns=vetores[1].get_feature_names())
        vocabulario = vetores[1].vocabulary_
        dfV = pd.DataFrame(list(vocabulario.items()),columns=['termo','frequencia']) 
        dfV = dfV.sort_values("frequencia", ascending=False)
        X_train_tfidf = self.retornaMatrizItfIdf(V)
        #self.imprimeGraficosFrequenciaPCA(X_train_tfidf)
        pca = self.reduzDimensionalidadecomPCA(X_train_tfidf,2)
        X_train_pca = pca[1]
        UT = pca[0]
        vet_test = self.retornaVetorizacao(X_test)[0]
        X_test_tfidf = self.retornaMatrizItfIdf(vet_test)
        X_test_pca = self.reduzDimensionalidadecomPCA(X_test_tfidf,2,UT)[1]
        #self.imprimeGraficosdosGrupos(X_train_pca)
        modelNB = self.treinaNaiveBayes(X_train_pca, y_train)
        
        
        scoreNB = self.avaliaModeloClassificacao(modelNB,X_test_pca,y_test)
        print('Acurácia Naive-Bayes:', scoreNB)
        modelSVM = self.treinaSVM(X_train_pca, y_train)
        scoreSVM = self.avaliaModeloClassificacao(modelSVM,X_test_pca,y_test)
        print('Acurácia SVM:', scoreSVM)
        modelRF = self.treinaRandomForest(X_train_pca, y_train)
        scoreRF = self.avaliaModeloClassificacao(modelRF,X_test_pca,y_test)
        print('Acurácia RF:', scoreRF)
        modelKnn = self.treinaKnn(X_train_pca, y_train,2)
        scoreKnn = self.avaliaModeloClassificacao(modelKnn,X_test_pca,y_test)
        print('Acurácia knn:', scoreKnn)
        print('duracao:',datetime.now()-inicio)

if __name__ == '__main__':
    nf = nlp_fc()
    nf.fluxoProcessamento()
    #df = nf.obterAmostra()
    