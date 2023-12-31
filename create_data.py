import os


# create data list
def get_U8K_data_list(audio_path, metadata_path, list_path):
    sound_sum = 0
    # get train test label
    file_train = open(os.path.join(list_path, 'train_list.txt'),
                      'w', encoding='utf-8')
    file_test = open(os.path.join(list_path, 'test_list.txt'),
                     'w', encoding='utf-8')
    file_label = open(os.path.join(list_path, 'label_list.txt'),
                      'w', encoding='utf-8')

    with open(metadata_path) as f:
        lines = f.readlines()

    labels = {}

    for i, line in enumerate(lines):
        if i == 0:
            continue
        data = line.replace("\n", '').split(',')
        class_id = int(data[6])
        if class_id not in labels.keys():
            labels[class_id] = data[-1]
        sound_path = os.path.join(audio_path, f'fold{data[5]}', data[0]).replace('\\', '/')
        # train:test 80:1
        if sound_sum % 80 == 0:
            file_test.write(f'{sound_path}\t{data[6]}\n')
        else:
            file_train.write(f'{sound_path}\t{data[6]}\n')
        sound_sum += 1

    for i in range(len(labels)):
        file_label.write(f'{labels[i]}\n')

    file_label.close()
    file_test.close()
    file_train.close()


def get_ESC50_data_list(audio_path, metadata_path, list_path):
    sound_sum = 0
    # get train test label
    file_train = open(os.path.join(list_path, 'train_list.txt'),
                      'w', encoding='utf-8')
    file_test = open(os.path.join(list_path, 'test_list.txt'),
                     'w', encoding='utf-8')
    file_label = open(os.path.join(list_path, 'label_list.txt'),
                      'w', encoding='utf-8')
    with open(metadata_path) as f:
        lines = f.readlines()

    labels = {}

    for i, line in enumerate(lines):
        if i == 0:
            continue

        file_name = line.split(',')[0]
        label_name = line.split(',')[3]
        label_num = line.split(',')[2]
        print('file:%s label:%s' % (file_name, label_name))
        if label_name not in labels.keys():
            labels[int(label_num)] = label_name
        # train:test = 40:1
        sound_path = os.path.join(audio_path, file_name).replace('\\', '/')
        if sound_sum % 40 == 0:
            file_test.write(f'{sound_path}\t{label_num}\n')
        else:
            file_train.write(f'{sound_path}\t{label_num}\n')
        sound_sum += 1

    for i in range(len(labels)):
        file_label.write(f'{labels[i]}\n')

    file_label.close()
    file_test.close()
    file_train.close()


# get_U8K_data_list('./data/UrbanSound8K/audio', './data/UrbanSound8K/metadata/UrbanSound8K.csv',
#                   './data/UrbanSound8K')

get_ESC50_data_list('./data/ESC50/audio', './data/ESC50/esc50.csv',
                    './data/ESC50')
