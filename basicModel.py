import pandas as pd
import numpy as np
from pulp import *


class CashLogWLP:
    # MIP for the CashLog warehouse location problem
    def __init__(self):
        self.warehouses = pd.read_csv('https://raw.githubusercontent.com/NikoStein/CashLog/master/data/warehouses.csv', index_col='warehouseID')
        self.W = self.warehouses.index.values
        self.regions = pd.read_csv('https://raw.githubusercontent.com/NikoStein/CashLog/master/data/regions.csv', index_col='regionID')
        self.R = self.regions.index.values
        self.shifts = pd.read_csv('https://raw.githubusercontent.com/NikoStein/CashLog/master/data/shifts.csv', index_col=['warehouseID', 'regionID'])
        self.S = self.shifts.index.values
                
    def solve(self, n_warehouses=-1, force_open = []):
        prob = LpProblem('CashLog_BasicAnalysis', LpMinimize)
        x = LpVariable.dicts(name='x', indexs=self.S, cat=LpBinary)
        y = LpVariable.dicts(name='y', indexs=self.W, cat=LpBinary)
        
        fixedCosts = lpSum([y[w] * self.warehouses.loc[w].fixedCosts for w in self.W])
        variableCosts = lpSum([x[w,r] * self.shifts.loc[w,r].transportationCosts for w,r in self.S]) 

        prob += fixedCosts + variableCosts


        for r in self.R:
            prob += lpSum([x[w,r] for w in self.W]) == 1

        for w,r in self.S:
            prob += x[w,r] <= y[w]
        
        for w in force_open:
            prob += y[w] == 1
            
        if n_warehouses != -1:
            prob += lpSum([y[w] for w in self.W]) == n_warehouses

        status = prob.solve()
        
        self.totalCosts = prob.objective.value()
        self.fixedCosts = sum([y[w].varValue * self.warehouses.loc[w].fixedCosts for w in self.W]) 
        self.variableCosts = sum([x[w,r].varValue * self.shifts.loc[w,r].transportationCosts 
                                  for w,r in self.S])
        
        
        self.region_results = []
        
        for w,r in self.S:
            v = x[w,r].varValue
            if v >= 0.1:
                self.region_results.append({'regionID': r, 
                                            'warehouseID':w, 
                                            'serviced': v, 
                                            'zipCode': self.regions.loc[r].zipCode, 
                                            'lat': self.regions.loc[r].lat,
                                            'lon': self.regions.loc[r].lon,
                                           'city': self.regions.loc[r].city})
        
        self.warehouse_results = [{'warehouseID': w, 'city': self.warehouses.loc[w].city, 
                                   'open': y[w].varValue, 
                                   'lat': self.warehouses.loc[w].lat,
                                   'lon': self.warehouses.loc[w].lon} for w in self.W]