#!usr/bin/env python  
#-*- coding:utf-8 _*-  
""" 
@author:JiaYiHe 
@file: shengao.py 
@time: 2019/02/{DAY} 
"""

import sys
import pandas as pd
import random
import math
import numpy as np
import xlrd
import copy
import matplotlib.pyplot as plt
from geopy.distance import geodesic
import time

origin_population_size=1000
best_population_size=500
city_num=0
x, y_min, z_mean, m_max, chromo = [], [], [], [], []

#订单
so=[]

#route
route=[]

#ou
ou=[]


#route_qty_cost {(ou,cus):{vender:[[minq1,percost1],[min2,percost2]]}} 路线运费
route_qty_cost={}

#cus_sku_ou={(cus,sku):rz_ou} 客户认证工厂约束
cus_sku_ou={}

#sku_ou={sku:rz_ou} 工厂生产sku约束
sku_ou={}



class InitialNodes(object):
    def __init__(self):
        xlsfile = r'C:\Users\JiaYiHe\Documents\mine\document\shengao\shengao.xlsx'

        #读入订单信息 so_info
        global so_init_df, so_info
        so_init_df = pd.read_excel(xlsfile,sheet_name='so_info',header=0,names=['so_id','cus_id','sku_id','dem'])
        so_info=np.array(so_init_df)


        #读入客户信息cus_info，且构建【客户，sku，认证工厂】约束
        global cus_info
        cus_info_df = pd.read_excel(xlsfile, sheet_name='cus_info', header=0,
                                    names=['cus_id', 'cus_nm', 'bs', 'nwm', 'rz_ou', 'sku_id'])
        cus_info = np.c_[cus_info_df['cus_id'],cus_info_df['cus_nm'],cus_info_df['rz_ou'],cus_info_df['sku_id']]

        #cus_sku_ou={(cus,sku):rz_ou} 客户认证工厂约束
        for i in range(0, len(cus_info_df)):
            k1 = cus_info_df.iloc[i]['cus_id']
            k2 = cus_info_df.iloc[i]['sku_id']
            v = cus_info_df.iloc[i]['rz_ou']
            cus_sku_ou.setdefault((k1, k2), set())
            cus_sku_ou[(k1, k2)].add(v)


        #读入产品信息prod_info，并且构建【sku，认证工厂】约束
        global prod_info
        prod_info_df = pd.read_excel(xlsfile,sheet_name='prod_info',header=0,names=['prod_id','prod_nm','sku_id','rz_ou'])
        # prod_info=np.c_[prod_info_df['sku_id'],prod_info_df['rz_ou']]
        prod_info=np.array(prod_info_df)

        #sku_ou {sku:rz_ou}
        for i in range(0,len(prod_info_df)):
            k=prod_info_df.iloc[i]['sku_id']
            v=prod_info_df.iloc[i]['rz_ou']
            sku_ou.setdefault(k,set())
            sku_ou[k].add(v)

        #ou_sku_cost
        global ou_sku_cost_df, ou_sku_cost
        ou_sku_cost_df = pd.read_excel(xlsfile,sheet_name='ou_sku_cost',header=0,names=['ou_id','sku_id','per_pcost'])
        ou_sku_cost = np.array(ou_sku_cost_df)

        #读入工厂产能ou_sku_qty
        global ou_sku_qty, ou_sku_qty_df
        ou_sku_qty_df = pd.read_excel(xlsfile, sheet_name='ou_sku_qty', header=0,
                                      names=['ou_id', 'ou_nm', 'sku_id', 'min_q', 'max_q', 'safe_q'])
        ou_sku_qty = np.array(ou_sku_qty_df)


        #读入承运商单位运输成本表route_qty_cost
        route_qty_cost_df = pd.read_excel(xlsfile, sheet_name='route_qty_cost', header=0,
                                          names=['route', 'ou_id', 'cus_id', 'min_q', 'percost', 'vender_id'])

        #{(ou,cus):{vender:[[minq1,percost1],[min2,percost2]]}}
        for i in range(0, len(route_qty_cost_df)):
            k1 = route_qty_cost_df.iloc[i]['ou_id']
            k2 = route_qty_cost_df.iloc[i]['cus_id']
            vender = route_qty_cost_df.iloc[i]['vender_id']
            minq = route_qty_cost_df.iloc[i]['min_q']
            percost = route_qty_cost_df.iloc[i]['percost']
            route_qty_cost.setdefault((k1, k2), {})
            route_qty_cost[(k1, k2)].setdefault(vender, [])
            route_qty_cost[(k1, k2)][vender].append([minq, percost])

        #将每条路线的阶级价格按升序排列，保证即使输入时不是按阶梯价升序输入也不影响
        for k, v in route_qty_cost.items():
            for nm, cost in v.items():
                cost = sorted(cost, key=lambda d: d[0], reverse=False)
                route_qty_cost[k][nm] = cost


    def addOUBackup(self, so_info_df):
        ou_list = []
        error_list1 = []
        error_list2 = []
        index_list = []
        for index, r in so_info_df.iterrows():
            cus_id = r['cus_id']
            sku_id = r['sku_id']

            if sku_id not in sku_ou:
                print('%s没有能生产的工厂' % (sku_id))
                ou_list.append([])
                error_list1.append([so_info_df.iloc[index]['so_id'],sku_id])
                index_list.append(index)
            elif (cus_id, sku_id) not in cus_sku_ou:
                # print('客户%s的%s不存在认证工厂' % (cus_id, sku_id))
                a = list(sku_ou[sku_id])
                ou_list.append(a)
                error_list2.append([cus_id, sku_id])
            else:
                a = sku_ou[sku_id]
                b = cus_sku_ou[(cus_id, sku_id)]
                ou=[]
                ou = list(a.union(b))
                ou_list.append(ou)

        ou = np.array(ou_list)
        so_info_df['ou_backup'] = ou

        #删除不能提供生产的订单，允许非客户认证工厂生产的sku
        index_list.sort(reverse=True)
        for i in index_list:
            so_info_df=so_info_df.drop(i)

        if len(error_list1)!=0:
            error_list1 = pd.DataFrame(error_list1)
            errorfile1 = r'C:\Users\JiaYiHe\Documents\mine\document\shengao\no_sku_ou&deleted_so_id.csv'
            error_list1.to_csv(errorfile1, index=False, header=['so_id','sku'])
        if len(error_list2)!=0:
            error_list2 = pd.DataFrame(error_list2)
            errorfile2 = r'C:\Users\JiaYiHe\Documents\mine\document\shengao\no_cus_sku_ou.csv'
            error_list2.to_csv(errorfile2, index=False, header=['cus_id','sku'])

        return so_info_df

    def addOU(self, so_info_df):
        global ou_sku_cost_df
        ou = [random.sample(x, 1)[0] for x in so_info_df['ou_backup']]
        so_info_df['ou_id'] = ou
        so_info_df = pd.merge(so_info_df, ou_sku_cost_df, how='left', on=(['ou_id','sku_id']))
        return so_info_df


    def addVender(self, so_info_df):
        line1, line2 = 'qty too small', 'no route exist'

        a = so_info_df.groupby(['ou_id', 'cus_id'])['dem'].sum().reset_index()
        a.rename(columns={'dem': 'route_dem_sum'}, inplace=True)

        vender_list = []
        for index, r in a.iterrows():
            ouid, cusid, dem = r['ou_id'], r['cus_id'], r['route_dem_sum']
            if (ouid, cusid) in route_qty_cost:
                venders = route_qty_cost[(ouid, cusid)]
                length = len(venders)
                best_lcost = []
                for vender, value in venders.items():
                    i = 0
                    for k in value:
                        if dem >= k[0] and i == len(value) - 1:
                            if len(best_lcost) == 0:
                                best_lcost.append([vender, value[i][1]])
                                break
                            elif value[i][1] < best_lcost[0][1]:
                                best_lcost = []
                                best_lcost.append([vender, value[i][1]])
                                break
                        elif dem >= k[0]:
                            i += 1
                            continue
                        elif i == 0:
                            break
                        else:
                            if len(best_lcost) == 0:
                                best_lcost.append([vender, value[i - 1][1]])
                                break
                            elif value[i - 1][1] < best_lcost[0][1]:
                                best_lcost = []
                                best_lcost.append([vender, value[i - 1][1]])
                                break
                if len(best_lcost) == 0:
                    best_lcost.append([line1, 0])
                vender_list.append(best_lcost[0])
            else:
                print('from ou %d to cus %s does not exixt route' % (ouid, cusid))
                vender_list.append([line2, 0])

        vl = pd.DataFrame(vender_list)
        a['vender'], a['per_lcost'] = vl[:][0], vl[:][1]

        result = pd.merge(so_info_df, a, how='left', on=(['ou_id', 'cus_id']))

        return result

    def outputAbnomalRoute(self, so_info_df):
        line1, line2 = 'qty too small', 'no route exist'
        # 删除不存在运输路线跟运输路线运量最小值不支持的订单
        index_list, index, error_list1, error_list2 = [], 0, [], []
        for i in so_info_df['vender']:
            #from ou %s to cus %s has a small demand qty %s, cannot find a vender to carry
            if i == line1:
                index_list.append(index)
                error_list1.append([so_info_df.iloc[index]['so_id'], so_info_df.iloc[index]['ou_id'], so_info_df.iloc[index]['cus_id'],
                                    so_info_df.iloc[index]['dem'], so_info_df.iloc[index]['route_dem_sum']])
            #from ou %d to cus %s does not exixt route
            if i == line2:
                index_list.append(index)
                error_list2.append([so_info_df.iloc[index]['so_id'], so_info_df.iloc[index]['ou_id'], so_info_df.iloc[index]['cus_id']])
            index += 1

        #执行删除操作
        # index_list.sort(reverse=True)
        # for i in index_list:
        #     so_info_df = so_info_df.drop(i)

        if len(error_list1) != 0:
            error_list1 = pd.DataFrame(error_list1)
            errorfile1 = r'C:\Users\JiaYiHe\Documents\mine\document\shengao\route_qty_too_small.csv'
            error_list1.to_csv(errorfile1, index=False, header=['so_id', 'ou_id', 'cus_id', 'dem', 'route_dem_sum'])
        if len(error_list2) != 0:
            error_list1 = pd.DataFrame(error_list2)
            errorfile1 = r'C:\Users\JiaYiHe\Documents\mine\document\shengao\no_route_exist.csv'
            error_list1.to_csv(errorfile1, index=False, header=['so_id', 'ou_id', 'cus_id'])

        return so_info_df



    def addCost(self, so_info_df):
        so_info_df['lcost_sum'] = so_info_df.apply(lambda x: x['dem'] * x['per_lcost'], axis=1)
        so_info_df['pcost_sum'] = so_info_df.apply(lambda x: x['dem'] * x['per_pcost'], axis=1)
        so_info_df['total_cost'] = so_info_df.apply(lambda x: x['lcost_sum'] + x['pcost_sum'], axis=1)
        return so_info_df

class Restriction(object):
    def productive(self, so_info_df):
        if so_info_df['total_cost'].isnull().any()==True or so_info_df['per_lcost'].isin([0]).any() or so_info_df['per_pcost'].isin([0]).any():
            return False
        else:
            return True

    def ouCapacity(self, so_info_df):
        global ou_sku_qty_df
        a = so_info_df.groupby(['ou_id', 'sku_id'])['dem'].sum().reset_index()
        a.rename(columns={'dem': 'ou_sku_sum'}, inplace=True)
        a = pd.merge(a, ou_sku_qty_df, how='left', on=(['ou_id','sku_id']))
        a['flag_minq'] = a.apply(lambda x : True if x['ou_sku_sum']>=x['min_q'] else False, axis=1)
        a['flag_maxq'] = a.apply(lambda x : True if x['ou_sku_sum']<=x['max_q'] else False, axis=1)
        if a['flag_minq'].isin(['False']).any() or a['flag_maxq'].isin(['False']).any():
            return False
        else:
            return True


class Selection(object):
    def calAndSort(self, population, desc= False):
        #计算每个chromosome路线的总长度
        global so_init_dfm, ou_sku_cost_df
        population_cost, i = [], 0

        while i < len(population):
            new = so_init_df
            new['ou_id']=np.array(population[i])
            new = pd.merge(new, ou_sku_cost_df, how='left', on=(['ou_id', 'sku_id']))
            new = init.addVender(new)
            # new = init.outputAbnomalRoute(new)
            new = init.addCost(new)

            if restriction.productive(new) and restriction.ouCapacity(new):
                # new_population.append(list(new['ou_id'].values))
                population_cost.append([list(new['ou_id'].values), new['total_cost'].sum()])
                # print(new)
            else:
                print('error')
            i += 1


        # 升序排列，取距离最小的几个，如果要降序则在后面加reverse=TRUE
        population_cost = sorted(population_cost, key=lambda d: d[1], reverse=desc)

        new_population = []
        for k, v in population_cost:
            new_population.append(k)

        return new_population, population_cost

    def linearRanking(self, population, population_distance, desc= False):
        if 1==2 :#best_population_size>len(population):
            print('best population size is lager than population size')
        else:
            #取出best_population_size数量的种群作为最优种群,当只取population list的部分的时候，新list与旧list不是一个list
            best_population=population[0:best_population_size]
            population_distance=population_distance[0:best_population_size]
            random.shuffle(best_population)

            # print('linear ranking population distance',population_distance)
            return best_population, population_distance

    def rouletteWheel(self, population, population_distance, desc=False):
        # 此轮盘赌算法中，能选中重复个体，输出的子代存在重复,输入的population是否排序不重要
        if 1 == 2:  # best_population_size>len(population):
            print('best population size is lager than population size')
        else:
            # population_distance = sorted(population_distance, key=lambda d: d[1], reverse=False)
            ar = np.array(population_distance)

            # 处理值，归一化
            # ar[:, 1] = ar[:, 1] / sum(ar[:, 1])
            # ar[:, 1] =( 1 / ar[:, 1] )
            a = sum(ar[:, 1])
            ar[:, 1] = (a - ar[:, 1])
            temp, d, F = [], 0, sum(ar[:, 1])
            for k, v in ar:
                d += v
                temp.append([k, d / F])

            # 发射n个飞镖
            # target = list( np.random.rand(best_population_size) )
            target, i = [], 0
            while i < best_population_size:
                target.append(random.betavariate(1, 3))
                i += 1

            temp1, best_population, best_population_distance = [], [], []

            while len(target) > 0:
                t = target.pop()
                index = 0
                for k, v in temp:
                    if index > len(temp) - 1:
                        break
                    elif t < v:
                        best_population_distance.append(population_distance[index])
                        best_population.append(population_distance[index][0])
                        break
                    elif t < temp[index + 1][1]:
                        best_population_distance.append(population_distance[index + 1])
                        best_population.append(population_distance[index + 1][0])
                        break
                    else:
                        index += 1
                        continue

            random.shuffle(best_population)
            return best_population, best_population_distance

class Evolution(object):
    # 单点交叉，输入父代种群，交叉基因占比
    def onePointCrossOver(self, population_ori, pg=0.3):
        pg_num = int(city_num * pg)
        population = []
        population = copy.deepcopy(population_ori)
        begin_index = random.randint(0, city_num - pg_num)
        for i in range(0, len(population), 2):
            population[i][begin_index:begin_index + pg_num], population[i + 1][begin_index:begin_index + pg_num] = \
                population[i + 1][begin_index:begin_index + pg_num], population[i][begin_index:begin_index + pg_num]

        return population

    def multation(self, population_ori, pm=0.5, pl=0.2):
        if pl >= 0.5:
            print('变异的基因占比太大，请重新设置pl')
        elif pm > 1 or pm < 0:
            print('变异染色体占比在0-1之间，请重新设置pm')
        else:
            population = []
            population = copy.deepcopy(population_ori)
            pl_num = int(city_num * pl)
            index1 = random.randint(0 + 1, int(city_num / 2) - pl_num)
            index2 = random.randint(int(city_num / 2), city_num - pl_num)
            temp = set()
            while len(temp) < len(population) * pm:
                chro_index = random.randint(0, len(population))
                temp.add(chro_index)
            for index in range(0, len(population)):
                if index in temp:
                    population[index][index1:index1 + pl_num], population[index][index2:index2 + pl_num] = \
                    population[
                        index][
                    index2:index2 + pl_num], \
                    population[
                        index][
                    index1:index1 + pl_num]
                    temp.remove(index)
            return population

class Result(object):
    def plottingOptimal(self, i, child_distance_list):
        plt.ion()
        plt.figure(1)
        # fig,ax = plt.subplots()
        # 画图
        # global x, y_min, z_mean, m_max

        c = child_distance_list
        p = np.mean(c[:, 1])
        q = np.max(c[:, 1])
        k = np.min(c[:, 1])

        print('time',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),'i=', i, 'optimal=', k, len(father_best_s), len(father), len(child))

        x.append(i)
        y_min.append(k)
        z_mean.append(p)
        m_max.append(q)

        plt.clf()
        plt.title('best population size=%s, origin population size=%s, cross over=%s' % (
            best_population_size, origin_population_size, pg))
        plt.plot(x, y_min, x, z_mean, x, m_max)
        plt.pause(0.1)  # 暂停一秒
        plt.ioff()  # 关闭画图的窗口


        def outputtingExcel(self):
            f = open('result.txt', 'w', encoding='utf-8')
            f['distance'] = data.apply(lambda x: geodesic((x[4], x[3]), (x[9], x[8])).miles, axis=1)

if __name__ == '__main__':
    init=InitialNodes() #读入6张基本表
    so_init_df = init.addOUBackup(so_init_df)
    print('so_init_df length',len(so_init_df))
    city_num = len(so_init_df)
    population=[]
    population_cost=[]
    father=[]
    restriction=Restriction()
    selection = Selection()
    evolution = Evolution()
    result = Result()

    #生成初始种群
    while len(population) < origin_population_size:
        new = init.addOU(so_init_df)
        new = init.addVender(new)
        new =init.outputAbnomalRoute(new)
        new =init.addCost(new)

        if restriction.productive(new) and restriction.ouCapacity(new):
            population.append(list(new['ou_id'].values))
            # population_cost.append([list(new['ou_id'].values),new['total_cost'].sum()])
            # print(new)
        else:
            print('error')

    print('length of population',len(population))
    y_min, fitness, i = [], 0, 0

    father=population
    while y_min.count(fitness) < 70:
        times = 100
        if i <= times:
            (father_s, father_d_s) = selection.calAndSort(father)
            (father_best_s, father_d_best_s) = selection.linearRanking(father_s, father_d_s)

            pg = 0.618
            child = evolution.onePointCrossOver(father_best_s, pg)
            (child_s, child_d_s) = selection.calAndSort(child)

            print('child length',len(child_s))
            father = father_best_s + child_s
            print('new father length',len(father))
            father_d = father_d_best_s + child_d_s

        i += 1
        child_d_list = np.array(father_d)
        fitness = np.min(child_d_list[:, 1])
        result.plottingOptimal(i, child_d_list)










