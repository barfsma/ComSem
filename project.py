#!/usr/bin/python3

import numpy as np



def file_to_matrix(file_path):
    matrix = []
    with open(file_path, "r") as file:

        for line in file:
            line = line.strip()
            lst = line.split()
            matrix.append(lst)
    return matrix



def main():

    train_matrix = file_to_matrix("./NLI2FOLI/SICK/SICK_train.txt")
    train_matrix = np.array(train_matrix)

    trial_matrix = file_to_matrix("./NLI2FOLI/SICK/SICK_trial.txt")
    trial_matrix = np.array(trial_matrix)

    
    print(trial_matrix)



if __name__ == '__main__':
    main()
