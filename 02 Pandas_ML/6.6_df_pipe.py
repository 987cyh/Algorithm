# -*- coding: utf-8 -*-
"""
출처: 파이썬 머신러닝 판다스 데이터 분석
목적: 독학으로 학습한 파이썬의 개념을 다시 한번 정리(학습)
"""
#%%
# 라이브러리 불러오기
import seaborn as sns

# titanic 데이터셋에서 age, fare 2개 열을 선택하여 데이터프레임 만들기
titanic = sns.load_dataset('titanic')
df = titanic.loc[:, ['age','fare']]

# 각 열의 NaN 찾기 - 데이터프레임 전달하면 데이터프레임을 반환
def missing_value(x):    
    return x.isnull()    

# 각 열의 NaN 개수 반환 - 데이터프레임 전달하면 시리즈 반환
def missing_count(x):    # 
    return missing_value(x).sum()

# 데이터프레임의 총 NaN 개수 - 데이터프레임 전달하면 값을 반환
def totoal_number_missing(x):    
    return missing_count(x).sum()
    
# 데이터프레임에 pipe() 메소드로 함수 매핑
result_df = df.pipe(missing_value)   
print(result_df.head())
print(type(result_df))

result_series = df.pipe(missing_count)   
print(result_series)
print(type(result_series))

result_value = df.pipe(totoal_number_missing)   
print(result_value)
print(type(result_value))
#%%
"""
참고: https://wikidocs.net/152677
"""
import pandas as pd
org_data = pd.DataFrame({'info':['삼성전자/3/70000','SK하이닉스/2/100000']})

# def1
def code_name(data):
    result=pd.DataFrame(columns=['name','count','price'])
    df = pd.DataFrame(list(data['info'].str.split('/'))) # '/ ' 로 구분하여 문자열을 나누어 리스트에 넣음
    result['name'] = df[0] # 여기엔 첫번째 값인 이름이 입력
    result['count']= df[1] # 여기엔 두번째 값인 수량이 입력
    result['price']= df[2] # 여기엔 세번째 값인 가격이 입력
    result = result.astype({'count':int,'price':int}) # count와 price를 int로 바꿈(기존str)
    return result
print(code_name(org_data))

# def2
def value_cal(data,unit=''):
    result = pd.DataFrame(columns=['name','value'])
    result['name'] =data['name'] # 이름은 기존거를 가져옴
    result['value']=data['count']*data['price'] # value는 count * price를 입력함
    result = result.astype({'value':str}) # value를 str로 변경(단위를 붙이기 위함)
    result['value']=result['value']+unit # 단위를 붙임
    return(result)

input=code_name(org_data)
print(value_cal(input,'원'))

# pipe 메서드를 사용하지 않는경우
print(value_cal(code_name(org_data),'원'))

# pipe 메서드를 사용하는 경우 ★★★
print(org_data.pipe(code_name).pipe(value_cal,'원')) # pipe를 통해서, 함수내 함수 연속적용
#%%
