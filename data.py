import csv
from args import parser
import numpy as np
args = parser.parse_args()
n, k = args.n, args.k
def create_sentences(sentence, n, k):
    tmp = []
    for i in range(len(sentence) -n -k + 1 ):
        tmp.append(sentence[i:i + n + k])
        
    return tmp
    

def create_dataset():
    train_, val_, test_ = [], [], []
    with open("data/steam.csv", "r") as f:
        for sentence in f.readlines():
            sentence = sentence.strip().split(",")
            sentence = [int(i) for i in sentence]  
            if len(sentence) >=  n + k:
                tmp = create_sentences(sentence, n, k)
                if len(tmp) >= 3:
                    train_.append(tmp[:-2])
                    val_.append(tmp[-2])
                    test_.append(tmp[-1])
                
    train_ = np.concatenate(train_, axis=0)
    # val_ = np.concatenate(val_, axis=0)
    # test_ = np.concatenate(test_, axis=0)
    with open("data/train.csv", "w", newline="") as f:  
        writer = csv.writer(f)
        writer.writerows(train_)  
    with open("data/val.csv", "w", newline="") as f:  
        writer = csv.writer(f)
        writer.writerows(val_) 
    with open("data/test.csv", "w", newline="") as f:  
        writer = csv.writer(f)
        writer.writerows(test_) 
        
if __name__ == "__main__":
    create_dataset()