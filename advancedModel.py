import pandas as pd
import numpy as np
from pulp import *


class CashLogWLPadvanced:
    # MIP for the CashLog warehouse location problem
    def __init__(self):
        self.warehouses = pd.read_csv('data/warehouses.csv', index_col='warehouseID')
        self.W = self.warehouses.index.values
        self.regions = pd.read_csv('data/regions_advanced_analysis.csv', index_col='regionID')
        self.R = self.regions.index.values
        self.shifts = pd.read_csv('data/shifts.csv', index_col=['warehouseID', 'regionID'])
        self.S = self.shifts.index.values
        
        self.fixCostFunctions = {1: [
                {'lower_bound': 0, 'upper_bound': 19348, 'fix_fix': 61165, 'var_fix': 4.14}, 
                {'lower_bound': 19349, 'upper_bound': 45415, 'fix_fix': 86071, 'var_fix': 2.85},
                {'lower_bound': 45416, 'upper_bound': 107327, 'fix_fix': 145100, 'var_fix': 1.55}, 
                {'lower_bound': 107328, 'upper_bound': 199999, 'fix_fix': 145100, 'var_fix': 1.55}, 
                {'lower_bound': 200000, 'upper_bound': 99999999, 'fix_fix': 145100, 'var_fix': 1.55}],
            2: [
                {'lower_bound': 0, 'upper_bound': 19348, 'fix_fix': 61165, 'var_fix': 5.20}, 
                {'lower_bound': 19349, 'upper_bound': 45415, 'fix_fix': 86071, 'var_fix': 3.91},
                {'lower_bound': 45416, 'upper_bound': 107327, 'fix_fix': 228337, 'var_fix': 0.78}, 
                {'lower_bound': 107328, 'upper_bound': 199999, 'fix_fix': 145100, 'var_fix': 1.55}, 
                {'lower_bound': 200000, 'upper_bound': 99999999, 'fix_fix': 145100, 'var_fix': 1.55}],
            3: [
                {'lower_bound': 0, 'upper_bound': 19348, 'fix_fix': 61165, 'var_fix': 8.50}, 
                {'lower_bound': 19349, 'upper_bound': 45415, 'fix_fix': 154384, 'var_fix': 3.68},
                {'lower_bound': 45416, 'upper_bound': 107327, 'fix_fix': 290789, 'var_fix': 0.68}, 
                {'lower_bound': 107328, 'upper_bound': 199999, 'fix_fix': 197059, 'var_fix': 1.55}, 
                {'lower_bound': 200000, 'upper_bound': 99999999, 'fix_fix': 197059, 'var_fix': 1.55}]
        }
                
    def solve(self, n_warehouses=-1, force_open = [], fixCostFunction = 1):
        cFix = self.fixCostFunctions[fixCostFunction]
        self.C = range(len(cFix))
        
        prob = LpProblem('CashLog_AdvancedAnalysis', LpMinimize)
        x = LpVariable.dicts(name='x', indexs=self.S, cat=LpBinary)
        y = LpVariable.dicts(name='y', indexs=(self.W, self.C), cat=LpBinary)
        z = LpVariable.dicts(name='z', indexs=(self.W, self.C), lowBound=0, upBound=9999999, cat=LpContinuous)
        
        fixedCosts = lpSum([z[w][c] * cFix[c]['var_fix'] for w in self.W for c in self.C]) + lpSum([y[w][c] * cFix[c]['fix_fix'] for w in self.W for c in self.C])
        variableCosts = lpSum([x[w,r] * self.shifts.loc[w,r].transportationCosts for w,r in self.S])

        prob += fixedCosts + variableCosts
        
        for r in self.R:
            prob += lpSum([x[w,r] for w in self.W]) == 1

        for w,r in self.S:
            prob += x[w,r] <= lpSum([y[w][c] for c in self.C])

        for w in self.W:
            prob += lpSum([y[w][c] for c in self.C]) <= 1

        for w in self.W:
            for c in self.C:
                prob += z[w][c] <= cFix[c]['upper_bound'] * y[w][c]
                prob += z[w][c] >= cFix[c]['lower_bound'] * y[w][c]

        for w in self.W:
            prob += lpSum([z[w][c] for c in self.C]) == lpSum([x[w,r] * self.regions.loc[r].sumDeliveries for r in self.R])
            
        for w in force_open:
            prob += lpSum([y[w][c] for c in self.C]) == 1
            
        if n_warehouses != -1:
            prob += lpSum([lpSum([y[w][c] for c in self.C]) for w in self.W]) == n_warehouses

            
        status = prob.solve(PULP_CBC_CMD())
        
        
        self.totalCosts = prob.objective.value()
        self.variableCosts = sum([x[w,r].varValue * self.shifts.loc[w,r].transportationCosts for w,r in self.S])
        self.cash_center_fixed = sum([y[w][c].varValue * cFix[c]['fix_fix'] for w in self.W for c in self.C])
        self.cash_center_var = sum([z[w][c].varValue * cFix[c]['var_fix'] for w in self.W for c in self.C])
        

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
        
        self.warehouse_results = []
        
        for w in self.W:
            self.warehouse_results.append(
                {'warehouseID': w,
                 'city': self.warehouses.loc[w].city, 
                 'open': sum([y[w][c].varValue for c in self.C]),
                 'lat': self.warehouses.loc[w].lat,
                                       'lon': self.warehouses.loc[w].lon
                }
            )