import os
import csv
import random
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='../../../dataset/Music',
                        help="path to output index files")
    parser.add_argument('--dum_time', default=10, type=int,
                        help="dummy times")
    args = parser.parse_args()
    filename = 'test.csv'

    info = []
    with open(os.path.join(args.path, filename), 'r') as f:
        for item in f:
            info.append(item)
    num = len(info)
    print(num)
    N = 4
    with open(os.path.join(args.path, 'test_sep.csv'), 'w') as sep:
        for k in range(args.dum_time):
            for i in range(num):
                class_list = []
                infos = [[] for n in range(N)]

                infos[0] = info[i]
                cls = infos[0].split(',')[0].split('/')[1]
                class_list.append(cls)
                for n in range(1, N):
                    indexN = random.randint(0, (num) - 1)
                    sample = info[indexN]
                    while sample.split(',')[0].split('/')[1] in class_list:
                        indexN = random.randint(0, num - 1)
                        sample = info[indexN]
                    infos[n] = sample
                    class_list.append(sample.split(',')[0].split('/')[1])

                print(class_list)
                s = infos[0].split(',')[0] + ', ' + infos[1].split(',')[0] + ', ' + infos[2].split(',')[0]+ ', ' + infos[3].split(',')[0]
                print(s)
                sep.write(s+'\n')


                    #f.write(item + '\n')




