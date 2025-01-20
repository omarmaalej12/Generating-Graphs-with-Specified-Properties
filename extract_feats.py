# extract_feats.py
import random
import re

random.seed(32)

def extract_feats(file):
    stats = []
    fread = open(file,"r")
    line = fread.read()
    line = line.strip()
    fread.close()
    return line