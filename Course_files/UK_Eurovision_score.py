# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 20:15:24 2016

@author: David Lloyd
"""
# plotting exercise with matplotlib
# did middle east wars affect eurovision score?
# Afghanistan no. Iraq, possibly.
import matplotlib.pyplot as plt
import numpy as np

# Better to be able scrape data online.
#Score =np.array ([76,77,227,166,38,28,28,111,0,29,18,25,19,14,173,10,100,12,23,40,5]);
Score = np.loadtxt("UK_Eurovision_score.txt", delimiter=",");
Year = np.array(range(1995,2016))+5./12; # Eurovision occurs middle of the year

# Iraq ivasion 20th March 2003 or 2003.22
# Afghan invasioon 7th October 2001 or 2001.77

ax1, = plt.plot(np.ones(28)*2003.22,range(0,280,10),'.k',label='Invasion of Iraq')
ax2, = plt.plot(np.ones(28)*2001.77,range(0,280,10),'.b',label='Invasion of Afghanistan')    
ax3 = plt.plot(Year,Score,'-or')

plt.legend(handles=[ax1, ax2])
plt.xlabel('Year')
# Make the y-axis label and tick labels match the line color.
plt.ylabel('Score')
plt.xlim([1995,2016])
plt.ylim([0, 250])
plt.title('UK Eurovision Score')
plt.xticks(np.arange(1995, 2020, step=5))
plt.show()

#%%
scores = np.loadtxt("UK_Eurovision_score.txt", delimiter=",")