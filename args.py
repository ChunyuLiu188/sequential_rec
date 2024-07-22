import argparse


    
parser = argparse.ArgumentParser(description='这是一个示例程序，用于演示 argparse 的使用。')
parser.add_argument('-n', type=int, default=20)
parser.add_argument('-k', type=int, default=1)
parser.add_argument('-batch_size', type=int, default=64)
parser.add_argument('-dim', type=int, default=64)
parser.add_argument('-item_num', type=int, default=13028)
parser.add_argument('-lr', type=float, default=1e-2)
parser.add_argument('-epoch', type=float, default=100)


    

    