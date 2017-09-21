from torch import FloatTensor
from sklearn.preprocessing import PolynomialFeatures


class Fit2Dataset(object):
    def __init__(self, user_num, labeled_tasks, users, topk_mapper=None):
        self.user_num = user_num
        self.labeled_tasks = labeled_tasks[['longitude', 'latitude', 'price']].astype('float')
        self.users = users[['credit', 'task_capacity', 'longitude', 'latitude']].astype('float')
        self.cache = {}
        self.poly = PolynomialFeatures(3)
        self.topk_mapper = topk_mapper

    def __len__(self):
        return len(self.labeled_tasks)

    # (20 | 10)
    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]
        task = self.labeled_tasks.iloc[idx]
        self.users.loc[:, 'd_longitude2'] = (
                                         self.users.loc[:, 'longitude'] -
                                         task['longitude'])\
                                     ** 2
        self.users.loc[:, 'd_latitude2'] = (self.users.loc[:, 'latitude'] - task['latitude']) ** 2
        self.users.loc[:, 'dists'] = self.users.loc[:, 'd_longitude2'] + self.users['d_latitude2']
        topk = self.users.nsmallest(self.user_num, 'dists')
        if self.topk_mapper is not None:
            topk = self.topk_mapper(topk)
        # print(self.users)
        # assert(False)
        self.cache[idx] = (FloatTensor(
            self.poly.fit_transform(topk[['dists', 'credit', 'task_capacity']].as_matrix())
        ), FloatTensor(self.poly.fit_transform([[task['longitude'], task['latitude']]]).flatten())), FloatTensor([task['price']])
        return self.cache[idx]
