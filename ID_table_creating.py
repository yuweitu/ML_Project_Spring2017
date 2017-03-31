#!/usr/bin/env python3.5

# save track album, artist, genre ID as different integer lists.



import sys
import string
import numpy as np
import pandas as pd

print('{0},{1}'.format('itemID','label'))
# read track ID
with open('trackData2.txt') as f:
    for line in f:
        line = line.strip()
    	trackID = line.split('|')[0]
    	label = 'track'
    	print('{0},{1}'.format(trackID,label))

# read album ID
with open('albumData2.txt') as f:
    for line in f:
        line = line.strip()
    	albumID = line.split('|')[0]
    	label = 'album'
    	print('{0},{1}'.format(albumID,label))

# read artist ID
with open('artistData2.txt') as f:
    for line in f:
        line = line.strip()
    	artistID = line.split('|')[0]
    	label = 'artist'
    	print('{0},{1}'.format(artistID,label))

# read genre ID
with open('genreData2.txt') as f:
    for line in f:
        line = line.strip()
    	genreID = line.split('|')[0]
    	label = 'genre'
    	print('{0},{1}'.format(genreID,label))


    


