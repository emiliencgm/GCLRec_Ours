'''
将数据集分为7:1:2
LightGCN原文的数据集为8:2
因此将train划分为7:1两部分(train + valid)
设置超参数, 在7上train, 在1上test用于early stop
(在上述的test中选取最好的超参数, 再在2上test ??)
都调试好了, 再将train和valid合并回8, 在8上训练再在2上测试
'''

import os
from os.path import join
import numpy as np

def read_data(datasetpath):
    train_origin = join(datasetpath, 'train.txt')
    ui_dict_train = {}
    ui_dict_valid = {}
    with open(train_origin) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    num_items = len(items)
                    num_split = int(num_items/8)
                    if num_split<1:
                        num_split = 1
                    valid_index = np.random.choice(num_items, num_split, replace=False)
                    uid = int(l[0])
                    ui_dict_valid[uid] = []
                    ui_dict_train[uid] = []
                    for i in range(num_items):
                        if i in valid_index:
                            ui_dict_valid[uid].append(items[i])
                        else:
                            ui_dict_train[uid].append(items[i])

    return ui_dict_train, ui_dict_valid

def write_data(filename, ui_dict, datasetpath):
    file = join(datasetpath, filename)

    # if not os.path.exists(file):
    #     os.makedirs(file, exist_ok=True)
    # else:
    #     return 

    with open(file, mode='a') as f:
        for uid, items in ui_dict.items():
            f.write(str(uid))
            f.write(' ')
            for item in items[:-1]:
                f.write(str(item))
                f.write(' ')
            f.write(str(items[-1]))
            f.write('\n')

def main():
    ROOT_PATH = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    DATA_PATH = join(ROOT_PATH, 'data')

    datasetpath = join(DATA_PATH, 'yelp2018')
    ui_dict_train, ui_dict_valid = read_data(datasetpath)
    write_data('train_7.txt', ui_dict_train, datasetpath)
    write_data('valid_1.txt', ui_dict_valid, datasetpath)

    # datasetpath = join(DATA_PATH, 'gowalla')
    # ui_dict_train, ui_dict_valid = read_data(datasetpath)
    # write_data('train_7.txt', ui_dict_train, datasetpath)
    # write_data('valid_1.txt', ui_dict_valid, datasetpath)

main()

def check_all_items_appeared_in_train():
    ROOT_PATH = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    DATA_PATH = join(ROOT_PATH, 'data')
    datasetpath = join(DATA_PATH, 'yelp2018')
    train_origin = join(datasetpath, 'train.txt')
    test_origin = join(datasetpath, 'test.txt')
    train_item_set = set([])
    test_item_set = set([])

    with open(train_origin) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    for item in items:
                        train_item_set.add(item)

    with open(test_origin) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    for item in items:
                        if item not in train_item_set:
                            print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print('done')