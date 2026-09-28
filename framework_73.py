import argparse
import numpy as np
import pandas as pd
import operator 

def framework(pairs, arr):
    targets = []
    for i in range(arr.shape[0]):
        row = arr[i]
        for cond, calc in pairs:
            if cond_eval(cond, row):
                targets.append(calc(row))
                break
    return targets

def cond_eval(condition, arr):
    ops = {">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le, "==": operator.eq}
    if condition is None: return True
    return ops[condition[1]](arr[condition[0]], condition[2])


# R1: feat_130 <= 0.20
def calc1(arr): 
    return -0.8500 * arr[149] + 0.0500 * arr[267] + 0.0500 * arr[50] 

# R2: 0.20 < feat_130 <= 0.50
def calc2(arr): 
    return -1.6500 * arr[149] + 1.6500 * arr[267] - 0.0500 * arr[50] 

# R3: 0.50 < feat_130 <= 0.70
def calc3(arr): 
    return 0.1505 * arr[149] - 0.3495 * arr[267] + 0.3500 * arr[50] - 0.0006

# R4: feat_130 > 0.70
def calc4(arr): 
    return -0.7500 * arr[149] - 1.1500 * arr[267] + 0.2500 * arr[50] 

def main(args):
    # Rule List ordered to implement the piecewise logic sequentially
    pair_list = [
        ((130, "<=", 0.20), calc1), 
        ((130, "<=", 0.50), calc2),
        ((130, "<=", 0.70), calc3),
        (None, calc4)  # Default for feat_130 > 0.70
    ]
    
    data_array = pd.read_csv(args.eval_file_path).values
    return framework(pair_list, data_array)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Framework Task 2")
    parser.add_argument("--eval_file_path", required=True, help="Path to evaluation CSV")
    args = parser.parse_args()
    target02 = main(args)