from torch import FloatTensor


class FitDataset(object):
    def __init__(self, user_num, labeled_tasks, users):
        self.user_num = user_num
        self.labeled_tasks = labeled_tasks[['longitude', 'latitude', 'price']].astype('float')
        self.users = users[['credit', 'task_capacity', 'longitude', 'latitude']].astype('float')
        self.cache = {}

    def __len__(self):
        return len(self.labeled_tasks)

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
        # print(self.users)
        # assert(False)
        self.cache[idx] = (FloatTensor(topk[['dists', 'credit', 'task_capacity']].as_matrix()), FloatTensor([task['longitude'], task['latitude']])), FloatTensor([task['price']])
        return self.cache[idx]
