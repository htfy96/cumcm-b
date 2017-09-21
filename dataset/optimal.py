from torch import FloatTensor
from sklearn.preprocessing import PolynomialFeatures


class OptimalPriceDataset(object):
    def __init__(self, user_num, labeled_tasks, users):
        self.user_num = user_num
        self.labeled_tasks = labeled_tasks[['longitude', 'latitude', 'price', 'status']].astype('float')
        self.users = users[['credit', 'task_capacity', 'longitude', 'latitude']].astype('float')
        self.cache = {}
        self.poly = PolynomialFeatures(3)


    def __len__(self):
        return len(self.labeled_tasks)

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]
        task = self.labeled_tasks.iloc[idx]
        self.users.loc[:, 'd_longitude2'] = (
                                                self.users.loc[:, 'longitude'] -
                                                task['longitude']) \
                                            ** 2
        self.users.loc[:, 'd_latitude2'] = (self.users.loc[:, 'latitude'] - task['latitude']) ** 2
        self.users.loc[:, 'dists'] = self.users.loc[:, 'd_longitude2'] + self.users['d_latitude2']
        topk = self.users.nsmallest(self.user_num, 'dists')
        self.cache[idx] = (FloatTensor(self.poly.fit_transform(topk[['dists', 'credit', 'task_capacity']].as_matrix())),
                FloatTensor(self.poly.fit_transform([[task['longitude'], task['latitude']]]).flatten()),
                FloatTensor([task['price']])
               ),FloatTensor([task['status']])
        return self.cache[idx]

