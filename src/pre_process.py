# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import xgboost as xgb


def creat_feature_dict(data_series, feature_list=None, start=1,step=1):
    if feature_list == None:
        feature_list = list(data_series.unique())
    length = len(feature_list)
    feature_dict = dict(zip(feature_list,range(start,start+length+1,step)))
    return feature_dict




def pre_process(data_frame):
    try :
        data_frame.drop('Unnamed: 0',axis=1,inplace=True)
    except:
        pass
    # 无意义的特征列
    col_of_nonsense = [
            #'Unnamed: 0',
            'member_id',
            #'zip_code',
            #'earliest_cr_line',
            'verification_status_joint',
            ]
    data_frame.drop(col_of_nonsense, axis = 1, inplace=True)
    
    # 空值填充均值的列========================================================
    col_na_to_mean = [
    
        ]
    
    # 空值填零的列===========================================================
    col_na_to_zero = [
        "mths_since_last_major_derog",
        "mths_since_last_record",
        "revol_util",
        
        "annual_inc_joint",
        "dti_joint",
        "all_util",
        "il_util",
        "inq_fi",
        "inq_last_12m",
        "max_bal_bc",
        "mths_since_rcnt_il",
        "open_acc_6m",
        "open_il_12m",
        "open_il_24m",
        "open_il_6m",
        "open_rv_12m",
        "open_rv_24m",
        "total_bal_il",
        "total_cu_tl",
    
        "tot_coll_amt",
        "total_rev_hi_lim",
        "tot_cur_bal",
        ]
    data_frame[col_na_to_zero] = data_frame[col_na_to_zero].fillna(-1)
    
    # 空值 检查 以及填充均值 或 0
    col_check = [
            'annual_inc',
            'pub_rec',
            'total_acc',
            'collections_12_mths_ex_med',
            ]
    for i in col_check[:3]:
        data_frame[i] = data_frame[i].fillna(data_frame[i].median())
    data_frame[col_check[3]] = data_frame[col_check[3]].fillna(-1)
    
    
    
    # 可以正则匹配提取数据列=================================================
    # 提取数字，其余填充0
    col_re = [
        "term",
        "zip_code",
        "emp_length",
        "sub_grade",
        ]
    for i in col_re:
        col_series = eval("data_frame."+i)
        data_frame[i] = col_series.str.extract("(\d+)", expand=False)
    data_frame[col_re] = data_frame[col_re].fillna(0).astype(np.int64)
    
    # 提取字符串
    data_frame.issue_d = data_frame.issue_d.str.extract("([A-Za-z]+)", expand = False)
    
    data_frame.earliest_cr_line = data_frame.earliest_cr_line.str.extract("(\d+)", expand = False)
    data_frame.earliest_cr_line = data_frame.earliest_cr_line.fillna('18')
    data_frame.earliest_cr_line = (((data_frame.earliest_cr_line.astype(np.int64)<=20)*20 + \
                        (data_frame.earliest_cr_line.astype(np.int64)>20)*19).astype(str)+ \
                                   data_frame.earliest_cr_line).astype(np.int64)
    
    # 全字符型数据且适合分类列===============================================
    col_str_cls = [
        "grade",
        "home_ownership",
        "pymnt_plan",
        "verification_status",
        "initial_list_status",
        "application_type",
        
        "issue_d",
        #"earliest_cr_line",
        "loan_status",
        #"addr_state",
        #"verification_status_joint",
        ]
    
    # 有明显意义的
    grade_dict = {'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7}
    home_ownership_dict = {"OWN":1,"RENT":2,"MORTGAGE":3}
    pymnt_plan_dict = {'y':1,'n':2}
    verification_status_dict = {'Verified':1,'Source Verified':1,'Not Verified':2,}
    initial_list_status_dict = {'f':1,'w':2}
    application_type_dict = {'INDIVIDUAL':2,'JOINT':1}
    loan_status_dict = {
        "Does not meet the credit policy. Status:Charged Off":1,
        "Fully Paid" :2,
        "Charged Off":3,
        "Does not meet the credit policy. Status:Fully Paid":4,
        "Default":5,
        "Issued":6,
        "In Grace Period":7,
        "Late (16-30 days)":8,
        "Late (31-120 days)":9,
        "Current":10,
        }
    
    # 无明显意义的 自动生成
    month = ('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')
    issue_d_dict = creat_feature_dict(data_frame.issue_d,feature_list=month,start=1)
    #earliest_cr_line_dict = creat_feature_dict(data_frame.earliest_cr_line,feature_list=month,start=1)
    #loan_status_dict = creat_feature_dict(data_frame.loan_status)
    #addr_state_dict = creat_feature_dict(data_frame.addr_state)
    
    # 修改frame
    for i in col_str_cls:
        col_series = eval("data_frame."+i)
        data_frame[i] = col_series.map(eval(i+'_dict'))
        data_frame[i] = data_frame[i].fillna(0)
        data_frame[i].astype(np.int64)
        
    # 把sub_grade 和 grade 结合起来
    data_frame.sub_grade = data_frame.grade*5+data_frame.sub_grade*1
    #data_frame.sub_grade = pd.factorize(data_frame.sub_grade.values, sort=True)[0]+1
    
    # 修改最乱字符串列================================================================
    col_str_mess = [
        'desc',
        #'emp_title',
        'purpose',
        'title',
        
        'addr_state',
        ]
    '''
    for i in col_str_mess:
        i_dict = creat_feature_dict(eval("data_frame."+i))
        col_series = eval("data_frame."+i)
        data_frame[i] = col_series.map(i_dict).astype(np.int64)
    '''
    # 用pandas factorize 自动转化成分类
    for i in col_str_mess:
        data_frame[i] = pd.factorize(data_frame[i].values, sort=True)[0]+1
    # 有工作为0 无工作为1
    data_frame.emp_title = data_frame.emp_title.isnull()*1
    
    # 试着丢掉一些特征
    col_drop = ['tot_cur_bal',
                'zip_code',
                'revol_bal',
                'addr_state',
                'grade',
                'revol_util',
                
                ]
    #data_frame.drop(col_drop, axis = 1, inplace=True)
    
    return data_frame


def retain_features(data_frame, feature_retain_list):
    if not 'acc_now_delinq' in feature_retain_list:
        feature_retain_list.append('acc_now_delinq')
    data_frame = data_frame[feature_retain_list]
    return data_frame

def normalize_feature(data_frame, feature_nor_list = None):
    if not feature_nor_list == None:
        columns = feature_nor_list
    else :
        columns = list(data_frame.columns)
    if "acc_now_delinq" in columns:
        columns.remove("acc_now_delinq")
    for col in columns:
        mean = data_frame[col].mean()
        std = data_frame[col].std()
        if std == 0 :
            data_frame[col] = 0
        else:
            data_frame[col] = (data_frame[col]-mean)/std
    return data_frame

def main():
    #data_train = pd.read_csv("../data/train.csv", low_memory=False)
    data_train = pd.read_csv("../data/train_sample.csv", low_memory=False,
                             encoding='ascii')
    data_train = pre_process(data_train)
    data_train.to_csv("../data/train_sample_mod.csv", index = False)

if __name__ == "__main__":
    main()



















