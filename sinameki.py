# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 21:41:35 2021
@author: Hakan Dagli
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as karar
#import pandas as pd




"""
  Bu yapay zeka algoritması, son 5 yılın farklı seviye ortasaha oyuncularının 
  verileri göz önüne alınarak hazırlanmıştır.
  Kriterler ve değerler özenle seçilmiştir.
"""

# Oyuncunun son 20 maç verisine göre sınıfını belirleme

sutDeneme = karar.Antecedent(np.arange(0, 46, 1), 'Rakip Ceza sahasında şut sayisi')
dikinePas = karar.Antecedent(np.arange(0, 601, 1), 'Dikine oynamaya sayisi')
topKazanma = karar.Antecedent(np.arange(0, 131, 1), 'Kendi sahasında top kazanma')
oyuncu = karar.Consequent(np.arange(1, 101, 1), 'ortasaha oyuncusu sınıfı')


sutDeneme['çok yetersiz'] = fuzz.trimf(sutDeneme.universe, [0, 0, 8])
sutDeneme['yetersiz'] = fuzz.trimf(sutDeneme.universe, [4,10,20])
sutDeneme['normal'] = fuzz.trimf(sutDeneme.universe, [10, 20, 30])
sutDeneme['iyi'] = fuzz.trimf(sutDeneme.universe, [20, 30, 45])
sutDeneme['çok iyi'] = fuzz.trimf(sutDeneme.universe, [30, 45, 45])

dikinePas['çok yetersiz'] = fuzz.trimf(dikinePas.universe, [0, 0, 200])
dikinePas['yetersiz'] = fuzz.trimf(dikinePas.universe, [100, 200,300])
dikinePas['normal'] = fuzz.trimf(dikinePas.universe, [200,300,450])
dikinePas['iyi'] = fuzz.trimf(dikinePas.universe, [300, 450, 600])
dikinePas['çok iyi'] = fuzz.trimf(dikinePas.universe, [450, 600, 600])

topKazanma['çok yetersiz'] = fuzz.trimf(topKazanma.universe, [0, 0, 50])
topKazanma['yetersiz'] = fuzz.trimf(topKazanma.universe, [25,50,75 ])
topKazanma['normal'] = fuzz.trimf(topKazanma.universe, [50, 75, 100])
topKazanma['iyi'] = fuzz.trimf(topKazanma.universe, [75, 100, 130])
topKazanma['çok iyi'] = fuzz.trimf(topKazanma.universe, [100, 130, 130])

oyuncu['berbat'] = fuzz.trimf(oyuncu.universe, [1, 1, 40])
oyuncu['sinameki'] = fuzz.trimf(oyuncu.universe, [20, 40,60 ])
oyuncu['normal'] = fuzz.trimf(oyuncu.universe, [40, 60, 80])
oyuncu['iyi'] = fuzz.trimf(oyuncu.universe, [60, 80, 100])
oyuncu['superstar'] = fuzz.trimf(oyuncu.universe, [80, 100, 100])


##berbat filtresi
kural1 = karar.Rule(sutDeneme['çok yetersiz'] & dikinePas['çok yetersiz'], oyuncu['berbat'])
kural2 = karar.Rule(sutDeneme['çok yetersiz'] & topKazanma['çok yetersiz'] , oyuncu['berbat'])
kural3 = karar.Rule(dikinePas['çok yetersiz'] & topKazanma['çok yetersiz'], oyuncu['berbat'])

##sinameki oyuncu filtresi
kural4 = karar.Rule(sutDeneme['çok yetersiz'] | dikinePas['çok yetersiz'] | topKazanma['çok yetersiz'], oyuncu['sinameki'])
kural5 = karar.Rule(sutDeneme['yetersiz'] & dikinePas['yetersiz'], oyuncu['sinameki'])
kural6 = karar.Rule(sutDeneme['yetersiz'] & topKazanma['yetersiz'], oyuncu['sinameki'])
kural7 = karar.Rule(topKazanma['yetersiz'] & dikinePas['yetersiz'], oyuncu['sinameki'])

##normal oyuncu filtresi
kural8 = karar.Rule(sutDeneme['yetersiz'] | dikinePas['yetersiz'] | topKazanma['yetersiz'], oyuncu['normal'])
kural9 = karar.Rule(sutDeneme['normal'] & dikinePas['normal'], oyuncu['normal'])
kural10 = karar.Rule(sutDeneme['normal'] & topKazanma['normal'], oyuncu['normal'])
kural11 = karar.Rule(topKazanma['normal'] & dikinePas['normal'], oyuncu['normal'])
kural12= karar.Rule(sutDeneme['iyi'] | dikinePas['iyi'] | topKazanma['iyi'], oyuncu['normal'])

##iyi oyuncu filtresi
kural13 = karar.Rule(sutDeneme['iyi'] & dikinePas['iyi'], oyuncu['iyi'])
kural14 = karar.Rule(sutDeneme['iyi'] & topKazanma['iyi'], oyuncu['iyi'])
kural15 = karar.Rule(topKazanma['iyi'] & dikinePas['iyi'], oyuncu['iyi'])
kural16= karar.Rule(sutDeneme['çok iyi'] | dikinePas['çok iyi'] | topKazanma['çok iyi'], oyuncu['iyi'])

##çok iyi oyuncu filtresi
kural17 = karar.Rule(sutDeneme['çok iyi'] & dikinePas['çok iyi'] & topKazanma['normal'],oyuncu['superstar'])
kural18 = karar.Rule(sutDeneme['çok iyi'] & dikinePas['çok iyi'] & topKazanma['iyi'],oyuncu['superstar'])
kural19 = karar.Rule(sutDeneme['çok iyi'] & dikinePas['normal'] & topKazanma['çok iyi'],oyuncu['superstar'])
kural20 = karar.Rule(sutDeneme['çok iyi'] & dikinePas['iyi'] & topKazanma['çok iyi'],oyuncu['superstar'])
kural21 = karar.Rule(sutDeneme['normal'] & dikinePas['çok iyi'] & topKazanma['çok iyi'],oyuncu['superstar'])
kural22 = karar.Rule(sutDeneme['iyi'] & dikinePas['çok iyi'] & topKazanma['çok iyi'],oyuncu['superstar'])
kural23 = karar.Rule(sutDeneme['çok iyi'] & dikinePas['çok iyi'] & topKazanma['çok iyi'],oyuncu['superstar'])
oyuncu_karar = karar.ControlSystem([kural1, kural2, kural3, kural4, kural5, kural6, kural7, kural8, kural9, kural10, kural11, kural12, kural13, kural14, kural15, kural16, kural17, kural18, kural19, kural20, kural21, kural22, kural23])
oyuncu_ = karar.ControlSystemSimulation(oyuncu_karar)

oyuncu_.input['Rakip Ceza sahasında şut sayisi'] =45
oyuncu_.input['Dikine oynamaya sayisi']=600
oyuncu_.input['Kendi sahasında top kazanma'] = 25


oyuncu_.compute()

print(oyuncu_.output['ortasaha oyuncusu sınıfı'])
oyuncu.view(sim=oyuncu_)
