#!usr/bin/env python  
#-*- coding:utf-8 _*-  
""" 
@author:JiaYiHe 
@file: gavrp.py 
@time: 2018/12/{DAY} 
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

origin_population_size=100 #初始化种群数量
best_population_size=50     #最优的种群数量
sta_dict = {}
sta_distance={}
x, y_min, z_mean, m_max, chromo = [], [], [], [], []

#global city_num, chromosome_legth 定义在初始化节点

#读入城市与经纬度
xlsfile=r'C:\Users\JiaYiHe\Documents\mine\document\basic\abnormal_all.txt'

data = pd.read_csv(xlsfile,delimiter=',')
data['distance'] = data.apply(lambda x: geodesic((x[4],x[3]),(x[9],x[8])).miles,axis=1)

class InitialNodes(object):
    def __init__(self):
        #读入 奶站编码sta_id-奶站名称sta_nm-纬度lat-经度lon,构建sta_dict
        xlsfile = r'C:\Users\JiaYiHe\Documents\mine\document\basic\test.xlsx'
        book = xlrd.open_workbook(xlsfile)
        sta = book.sheet_by_name("Sheet1")
        # sta = pd.read_csv(xlsfile,delimiter=',')
        n = 1
        for r in range(1, sta.nrows):
            sta_id = str(int(sta.cell(r, 0).value))
            sta_nm = sta.cell(r, 1).value
            lat = sta.cell(r, 2).value
            lon = sta.cell(r, 3).value
            sta_dict.setdefault(n, {'sta_id': sta_id})
            sta_dict[n]['sta_nm'] = sta_nm
            sta_dict[n]['lat'] = lat
            sta_dict[n]['lon'] = lon
            global city_num, chromosome_length
            city_num = n    #城市数量为路径长度-2（刨除起点与终点）
            chromosome_length = city_num + 2    #每条路径的长度
            n += 1
        #初始化工厂
        factory_id=81
        factory_nm='230017_OU_内蒙古伊利实业集团股份有限公司乌兰察布乳品厂'
        lat = 40.957
        lon = 113.11033
        sta_dict.setdefault(0,{'factory_id':factory_id,'factory_nm':factory_nm,'lat':lat,'lon':lon})

class Distance(object):
    # 初始化奶站与工厂距离字典
    def __init__(self):
        for i in sta_dict:
            for j in sta_dict:
                sta_distance.setdefault(i,{j:{}})
                if i==j:
                    sta_distance[i][j]=999999
                else:
                    distance=geodesic((sta_dict[i]['lat'],sta_dict[i]['lon']),(sta_dict[j]['lat'],sta_dict[j]['lon'])).miles
                    sta_distance[i][j]=distance

class GaEngine(object):
    #产生种群，一共选取population_size条路线，每条路线含起点终点（同一个），共长chromosome_length
    def __init__(self, city_num, origin_population_size, best_population_size, chromosome_length):
        self.city_num=city_num
        self.origin_population_size=origin_population_size
        self.best_population_size=best_population_size
        self.chromosome_length=chromosome_length

    def spicesOrigin(self):
        population = []
        for i in range(origin_population_size):
            temp = []
            s1 = [n for n in range(1,chromosome_length-1,1)]
            for j in range(chromosome_length):
                if j == 0: temp.append(0)
                elif j == chromosome_length-1: temp.append(0)
                else :
                    index = random.randint(0,len(s1)-1)
                    temp.append(s1[index])
                    s1.remove(s1[index])
            population.append(temp)
        return population

class Selection(object):
    #计算每条染色体的长度，并且将染色体按顺序进行排序（默认升序），输出排序后的population/list与带有路程的population_distance/dict
    def calAndSort(self, population, desc= False):
        #计算每个chromosome路线的总长度
        population_distance=[]
        for chromosome in population:
            sum1 = 0
            index = 0
            for gene in chromosome:
                if index >= len(chromosome) - 1:
                    break
                else:
                    sum1 += sta_distance[gene][chromosome[index + 1]]
                    index += 1
            population_distance.append([chromosome, sum1])

        #升序排列，取距离最小的几个，如果要降序则在后面加reverse=TRUE
        population_distance = sorted(population_distance, key=lambda d: d[1], reverse=desc)

        new_population=[]
        for k, v in population_distance:
            new_population.append(k)

        return new_population, population_distance

    #输入排序后的population/list，输入population_distance/dict，输出top population_size的population
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


    def rouletteWheel_distinct(self, population, population_distance, desc= False):
        #此轮盘赌算法中，不能选中重复个体，输出的子代不存在重复
        if 1==2:#best_population_size>len(population):
            print('best population size is lager than population size')
        else:
            best_population = []
            dict_pd = {}
            new_population_distance = {}
            temp = {}
            # 计算轮盘的分母
            F = 0
            for k, v in population_distance:
                dict_pd.setdefault(tuple(k), v)
                v=(1/v)**4
                F+=v
                new_population_distance.setdefault(tuple(k), v)
            v1=0
            for k, v in new_population_distance.items():
                v1+=v
                temp.setdefault(tuple(k), v1 / F)

            new_population_distance = list(temp.items())
            dict_pd = list(dict_pd.items())
            temp = {}
            temp1 = {}
            n = 0
            best_population = []
            pop = set()
            while len(pop) < best_population_size:
                r = random.random()
                index = 0
                for k, v in new_population_distance:
                    if index >= len(new_population_distance) - 1 or len(pop) >= best_population_size:
                        break
                    elif r < v and k not in pop:
                        pop.add(k)
                        temp.setdefault(k, v)
                        temp1.setdefault(dict_pd[index][0], dict_pd[index][1])
                        index += 1
                        break
                    elif r < new_population_distance[index + 1][1] and r > v:
                        pop.add(new_population_distance[index + 1][0])
                        temp.setdefault(new_population_distance[index + 1][0], new_population_distance[index + 1][1])
                        temp1.setdefault(dict_pd[index + 1][0], dict_pd[index + 1][1])
                        index += 1
                        break
                    else:
                        index += 1
                        continue

            new_population_distance = []
            for k, v in temp1.items():
                new_population_distance.append([list(k), v])

            new_population_distance=sorted(new_population_distance, key=lambda d: d[1], reverse=desc)
            for i in pop:
                best_population.append(list(i))
            # best_population = list(pop)
            # random.shuffle(best_population)
            return best_population, new_population_distance

    def rouletteWheel(self,population,population_distance,desc = False):
        # 此轮盘赌算法中，能选中重复个体，输出的子代存在重复,输入的population是否排序不重要
        if 1 == 2:  # best_population_size>len(population):
            print('best population size is lager than population size')
        else:
            # population_distance = sorted(population_distance, key=lambda d: d[1], reverse=False)
            ar = np.array(population_distance)

            # 处理值，归一化
            # ar[:, 1] = ar[:, 1] / sum(ar[:, 1])
            # ar[:, 1] =( 1 / ar[:, 1] )
            a=sum(ar[:, 1])
            ar[:,1]=(a-ar[:, 1])
            temp, d, F = [], 0, sum(ar[:, 1])
            for k, v in ar:
                d += v
                temp.append([k, d / F])

            # 发射n个飞镖
            # target = list( np.random.rand(best_population_size) )
            target, i=[], 0
            while i < best_population_size:
                target.append(random.betavariate(1, 3))
                i+=1

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
    #单点交叉，输入父代种群，交叉基因占比
    def onePointCrossOver(self, population_ori, pg=0.3):
        pg_num = int( city_num * pg )
        population=[]
        population=copy.deepcopy(population_ori)
        begin_index= random.randint( 0 + 1, city_num - pg_num + 1)
        for i in range(0, len(population), 2):
            population[i][begin_index:begin_index + pg_num], population[i + 1][begin_index:begin_index + pg_num] = \
            population[i + 1][begin_index:begin_index + pg_num], population[i][begin_index:begin_index + pg_num]
            set1 = set([x for x in population[i] if population[i].count(x) > 1])
            set1.remove(0)
            set1.discard(999)
            temp1 = copy.deepcopy(population[i])
            temp2 = copy.deepcopy(population[i + 1])
            while len(set1) > 0:
                set2 = set([x for x in temp2 if temp2.count(x) > 1])
                set2.remove(0)
                set2.discard(999)
                b = set2.pop()
                a = set1.pop()
                if temp1.index(a) >= begin_index and temp1.index(a) < begin_index + pg_num:
                    temp1[temp1.index(a)] = 9999
                    temp1[temp1.index(a)] = b
                else:
                    temp1[temp1.index(a)] = b
                if temp2.index(b) >= begin_index and temp2.index(b) < begin_index + pg_num:
                    temp2[temp2.index(b)] = 999
                    temp2[temp2.index(b)] = a
                else:
                    temp2[temp2.index(b)] = a
            population[i][0:begin_index], population[i][begin_index + pg_num:] = temp1[0:begin_index], temp1[
                                                                                                       begin_index + pg_num:]
            population[i + 1][0:begin_index], population[i + 1][begin_index + pg_num:] = temp2[0:begin_index], temp2[
                                                                                                       begin_index + pg_num:]
        return population

    def multation(self, population_ori, pm = 0.5, pl = 0.2):
        if pl >= 0.5:
            print('变异的基因占比太大，请重新设置pl')
        elif pm > 1 or pm < 0 :
            print('变异染色体占比在0-1之间，请重新设置pm')
        else:
            population=[]
            population=copy.deepcopy(population_ori)
            pl_num = int(city_num * pl)
            index1 = random.randint(0 + 1, int(city_num/2) - pl_num)
            index2 = random.randint(int(city_num/2), city_num - pl_num)
            temp=set()
            while len(temp) < len(population) * pm:
                chro_index = random.randint(0, len(population))
                temp.add(chro_index)
            for index in range(0, len(population)):
                if index in temp:
                    population[index][index1:index1 + pl_num], population[index][index2:index2 + pl_num] = population[
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

        print('i=', i, 'optimal=', k, len(father_best_s), len(father), len(child))

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
        # plt.show()

    def outputtingExcel(self):
        f = open('result.txt', 'w', encoding='utf-8')
        f['distance'] = data.apply(lambda x: geodesic((x[4], x[3]), (x[9], x[8])).miles, axis=1)

if __name__ == '__main__':
    init=InitialNodes() #读取奶站与工厂信息，写入字段sta_dict
    Distance()  # 初始化工厂与奶站距离字典

    a=GaEngine(city_num,origin_population_size, best_population_size, chromosome_length)   #初始化种群,取10个
    father=a.spicesOrigin()
    print('fatehr',father)
    # print('sta_distance',sta_distance)

    selection=Selection()
    evolution = Evolution()
    result = Result()

    y_min ,fitness, i= [], 0, 0


    while y_min.count(fitness) < 400 :
        times=200
        if i <=times:
            (father_s, father_d_s) = selection.calAndSort(father)  #排序不删减
            (father_best_s, father_d_best_s) = selection.linearRanking(father_s, father_d_s)

            pg=0.618
            child=evolution.onePointCrossOver(father_best_s,pg)
            child = evolution.multation(child, 0.8, 0.2)
            (child_s, child_d_s) = selection.calAndSort(child)

            father = father_best_s + child_s
            father_d = father_d_best_s + child_d_s


        if i >times:
            (father_s, father_d_s) = selection.calAndSort(father)  # 排序不删减
            (father_best_s, father_d_best_s) = selection.rouletteWheel(father_s, father_d_s)

            pg = 0.5
            child = evolution.onePointCrossOver(father_best_s,pg)
            child = evolution.multation(child,0.2,0.3)
            (child_s, child_d_s) = selection.calAndSort(child)

            father = father_best_s + child_s
            father_d = father_d_best_s + child_d_s

        i+=1
        child_d_list = np.array(father_d)
        fitness = np.min(child_d_list[:,1])
        result.plottingOptimal(i, child_d_list)






