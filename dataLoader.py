class train_loader(object):
    def __init__(self, train_list, train_path, **kwargs):
        # Load data & labels
        self.data_list = []
        self.data_label = []
        with open(train_path + '/' + train_list, 'r', encoding='utf-8') as file:
            for line in file:
                label = line.strip().split('\t')[1]
                fileName = line.strip().split('\t')[0]
                self.data_list.append(fileName)
                self.data_label.append(label)

    def __getitem__(self, item):
        return

    def __len__(self):
        return len(self.data_list)

    def getList(self):
        return self.data_list
