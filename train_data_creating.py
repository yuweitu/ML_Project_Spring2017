#!/usr/bin/env python3.5

import sys
import string

print('{0},{1},{2}'.format('userID', 'itemID', 'rating'))

for line in sys.stdin:

    #Remove leading and trailing whitespace
    line = line.strip()
    
    if "|" in line:
        userID = line.split("|")[0]
        num_rating = line.split("|")[1]


    else:
        itemID= line.split("\t")[0]
        rating = line.split("\t")[1]
        print('{0},{1},{2}'.format(userID, itemID, rating))







