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

        if data[-1] != "street_music":
            continue

        if data[-1] not in labels.values():
            labels[len(labels)] = data[-1]
        sound_path = os.path.join(audio_path, f'fold{data[5]}', data[0]).replace('\\', '/')
        # train:test 80:1
        if sound_sum % 80 == 0:
            file_test.write(f'{sound_path}\t{data[6]}\n')
        else:
            file_train.write(f'{sound_path}\t{data[6]}\n')
        sound_sum += 1

    print(labels)
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

    add_U8K_data_list('./data/UrbanSound8K/audio', './data/UrbanSound8K/metadata/UrbanSound8K.csv', file_train,
                      file_test, labels, "street_music")

    add_custom_data_list('./data/CustomClass', file_train, file_test, labels)

    for i in range(len(labels)):
        file_label.write(f'{labels[i]}\n')

    file_label.close()
    file_test.close()
    file_train.close()


def add_U8K_data_list(audio_path, metadata_path, file_train, file_test, labels, label_name):
    with open(metadata_path) as u8f:
        lines2 = u8f.readlines()

    sound_sum = 0

    for i, line in enumerate(lines2):
        if i == 0:
            continue
        data = line.replace("\n", '').split(',')
        # Pick label
        if data[-1] != label_name:
            continue
        # ESC50 Single category only 50
        if sound_sum >= 70:
            break

        if data[-1] not in labels.values():
            labels[len(labels)] = data[-1]

        labelsIndex = list(labels.values()).index(data[-1])

        sound_path = os.path.join(audio_path, f'fold{data[5]}', data[0]).replace('\\', '/')
        if sound_sum % 40 == 0:
            file_test.write(f'{sound_path}\t{labelsIndex}\n')
        else:
            file_train.write(f'{sound_path}\t{labelsIndex}\n')
        sound_sum += 1


def add_custom_data_list(audio_path, file_train, file_test, labels):
    dir_list = os.listdir(audio_path)
    sound_sum = 0

    for base_path in dir_list:
        print(base_path)
        labels[len(labels)] = base_path
        labelsIndex = list(labels.values()).index(base_path)
        file_list = os.listdir(os.path.join(audio_path, base_path))

        for file_path in file_list:
            sound_path = os.path.join(audio_path, base_path, file_path).replace('\\', '/')
            if sound_sum % 19 == 0:
                file_test.write(f'{sound_path}\t{labelsIndex}\n')
            else:
                file_train.write(f'{sound_path}\t{labelsIndex}\n')
            sound_sum += 1


# get_U8K_data_list('./data/UrbanSound8K/audio', './data/UrbanSound8K/metadata/UrbanSound8K.csv',
#                   './data/UrbanSound8K')

get_ESC50_data_list('./data/ESC50/audio', './data/ESC50/esc50.csv',
                    './data/')

# add_custom_data_list('./data/CustomClass', './data/CustomClass', './data/CustomClass', './data/CustomClass',
#                      './data/CustomClass')
