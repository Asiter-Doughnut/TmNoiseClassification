from dataLoader import train_loader

train_Loader = train_loader('train_list.txt', './data', 2)


def execute_my_test_func():
    for i in range(20):
        audio, label = train_Loader.__getitem__(i)
        print(f'audio.shape:{audio.shape},label: {label}')


if __name__ == '__main__':
    execute_my_test_func()
