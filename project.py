#!/usr/bin/python3

import numpy as np



def file_to_matrix(file_path):
    matrix = []
    with open(file_path, "r") as file:

        for line in file:
            line = line.strip()
            lst = line.split()
            matrix.append(lst)
    return np.array(matrix)



def main():

    train_matrix = file_to_matrix("./NLI2FOLI/SICK/SICK_train.txt")


    trial_matrix = file_to_matrix("./NLI2FOLI/SICK/SICK_trial.txt")



if __name__ == '__main__':
    main()
