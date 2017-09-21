import pandas as pd
from sklearn.utils import shuffle

labeled_tasks = shuffle(pd.read_excel('./1.xls').rename(
    index=str, columns={
        '任务号码': 'id',
        '任务gps 纬度': 'latitude',
        '任务gps经度': 'longitude',
        '任务标价': 'price',
        '任务执行情况': 'status'
    }
), random_state=42)
labeled_tasks['longitude'] = (labeled_tasks['longitude'] - 113.537538) / 0.372860
labeled_tasks['latitude'] = (labeled_tasks['latitude'] - 22.982542) / 0.245252
labeled_tasks['price'] = (labeled_tasks['price'] - 69.110778) / 4.512772

users = pd.read_excel('./2.xlsx').rename(
    index=str,
    columns={
        '会员编号': 'id',
        '会员位置(GPS)': 'location',
        '预订任务限额': 'task_capacity',
        '预订任务开始时间': 'start_time',
        '信誉值': 'credit'
    }
)
users['latitude'], users['longitude'] = users['location'].str.split(' ', 1).str
users['longitude'] = users['longitude'].astype('float32')
users['latitude'] = users['latitude'].astype('float32')

users['task_capacity'] = (users['task_capacity'] - 6.835376) / 14.166843
users['credit'] = (users['credit'] - 278.134376) / 2328.754979
users['latitude'] = (users['latitude'] - 22.983044) / 2.120936
users['longitude'] = (users['longitude'] - 113.591042) / 2.140695


unlabeled_tasks = shuffle(pd.read_excel('./3.xls').rename(
    index=str,
    columns={
        '任务号码': 'id',
        '任务GPS纬度': 'latitude',
        '任务GPS经度': 'longitude'
    }
), random_state=84)
unlabeled_tasks['longitude'] = (unlabeled_tasks['longitude'] - 113.537538) / 0.372860
unlabeled_tasks['latitude'] = (unlabeled_tasks['latitude'] - 22.982542) / 0.245252
