#!/usr/bin/python3

import numpy as np





def main():

    train_lst = []
    with open("./NLI2FOLI/SICK/SICK_train.txt", "r") as file:

        for line in file:
            line = line.strip()
            lst = line.split()
            train_lst.append(lst)
    print(train_lst)

if __name__ == '__main__':
    main()
