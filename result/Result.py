import os 
import sys

ls = [i for i in os.walk(os.getcwd())]
# print(ls)
print(len(ls[0]))
