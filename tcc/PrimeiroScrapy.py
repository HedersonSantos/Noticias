# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 19:59:01 2021

@author: User
"""
import scrapy


class PrimeiroscrapyItem(scrapy.Item):
    url         = scrapy.Field()
    categoria   = scrapy.Field()
    conteudo    = scrapy.Field()
    