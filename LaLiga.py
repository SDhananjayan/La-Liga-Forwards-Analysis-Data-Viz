# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 11:43:12 2021

@author: DHANANJAYAN
"""
import pandas as pd
import matplotlib.pyplot as plt
DATADIR ='/Users/Dhananjayan/Onedrive/Documents'
home_team = 'real'
away_team = 'barca'
neutral_team = 'atleti'
home_shotfile = '/%s_elclasico.csv' % (home_team)
away_shotfile = '/%s_elclasico.csv' % (away_team)
neutral_shotfile = '/%s_elclasico.csv' % (neutral_team)
neutral_shot_data = pd.read_csv('{}/{}'.format(DATADIR, neutral_shotfile))
neutral_forward_shot_data = neutral_shot_data[neutral_shot_data['Pos'].str.contains('FW')]
home_shot_data = pd.read_csv('{}/{}'.format(DATADIR, home_shotfile))
away_shot_data = pd.read_csv('{}/{}'.format(DATADIR, away_shotfile))
home_forward_shot_data = home_shot_data[home_shot_data['Pos'].str.contains('FW')]
away_forward_shot_data = away_shot_data[away_shot_data['Pos'].str.contains('FW')]
away_forward_xGperShot = away_forward_shot_data[['Player','npxG/Sh']]
home_forward_xGperShot = home_forward_shot_data[['Player','npxG/Sh']]
home_forward_Shots_per90 = home_forward_shot_data[['Player','Sh/90']]
away_forward_Shots_per90 = away_forward_shot_data[['Player','Sh/90']]
neutral_forward_xGperShot = neutral_forward_shot_data[['Player','npxG/Sh']]
neutral_forward_Shots_per90 = neutral_forward_shot_data[['Player','Sh/90']]
away_forward_shot_data['plotsize'] = away_forward_shot_data['npxG']*400
home_forward_shot_data['plotsize'] = home_forward_shot_data['npxG']*400
neutral_forward_shot_data['plotsize'] = neutral_forward_shot_data['npxG']*400
league = 'laliga'
league_shotfile = '/%s_elclasico.csv' % (league)
league_shot_data = pd.read_csv('{}/{}'.format(DATADIR, league_shotfile))
league_forward_shot_data = league_shot_data[league_shot_data['Pos'].str.contains('FW')]
league_forward_shot_data['plotsize'] = league_forward_shot_data['npxG']*400
league_forward_median_Sh90 = league_forward_shot_data['Sh/90'].median()
league_forward_median_xG90 = league_forward_shot_data['xG'].median()
league_forward_median_npxG_perShot = league_forward_shot_data['npxG/Sh'].median()
x_away = away_forward_shot_data['Sh/90']
y_away = away_forward_shot_data['npxG/Sh']
s_away = away_forward_shot_data['plotsize']
n_away = away_forward_shot_data['Player']
x_neutral = neutral_forward_shot_data['Sh/90']
y_neutral = neutral_forward_shot_data['npxG/Sh']
s_neutral = neutral_forward_shot_data['plotsize']
n_neutral = neutral_forward_shot_data['Player']
fig, ax = plt.subplots(1, figsize=(10,6))
plt.ylabel("xG per Shot")
plt.xlabel("Shots per 90")
fig.suptitle("QUALITY VS QUANTITY of Shots in La Liga: Barcelona", fontsize ='17')
ax.scatter(x_away, y_away, s_away, c = 'red')
for x_pos, y_pos, label in zip(x_away, y_away, n_away):
    ax.annotate(label, xy=(x_pos, y_pos), xytext=(-7,14), textcoords='offset points', ha='left', va='top', fontsize = '10')
plt.show()
x_home = home_forward_shot_data['Sh/90']
y_home = home_forward_shot_data['npxG/Sh']
s_home = home_forward_shot_data['plotsize']
n_home = home_forward_shot_data['Player']
fig, ax = plt.subplots(1, figsize=(10,6))
plt.ylabel("xG per Shot")
plt.xlabel("Shots per 90")
fig.suptitle("QUALITY VS QUANTITY of Shots in La Liga: Real Madrid", fontsize ='17')
ax.scatter(x_home, y_home, s_home, c = 'cyan')
for x_pos, y_pos, label in zip(x_home, y_home, n_home):
    ax.annotate(label, xy=(x_pos, y_pos), xytext=(-7,14), textcoords='offset points', ha='left', va='top', fontsize = '10')
plt.show()
x_league = league_forward_shot_data['Sh/90']
y_league = league_forward_shot_data['npxG/Sh']
s_league = league_forward_shot_data['plotsize']
n_league = league_forward_shot_data['Player']
fig, ax = plt.subplots(1, figsize=(10,6))
plt.ylabel("xG per Shot")
plt.xlabel("Shots per 90")
fig.suptitle("QUALITY VS QUANTITY of Shots in La Liga", fontsize ='17')
ax.scatter(x_league, y_league, s_league, c = 'black')
for x_pos, y_pos, label in zip(x_league, y_league, n_league):
    ax.annotate(label, xy=(x_pos, y_pos), xytext=(-7,14), textcoords='offset points', ha='left', va='top', fontsize = '10', zorder = 's_league')
plt.show()
fig, ax = plt.subplots(1, figsize=(10,6))
plt.ylabel("xG per Shot")
plt.xlabel("Shots per 90")
fig.suptitle("QUALITY VS QUANTITY of Shots in La Liga: Barcelona vs Real Madrid", fontsize ='17')
ax.scatter(x_away, y_away, s_away, c = 'red')
for x_pos, y_pos, label in zip(x_away, y_away, n_away):
    ax.annotate(label, xy=(x_pos, y_pos), xytext=(-10,18), textcoords='offset points', ha='left', va='top', fontsize = '10')
ax.scatter(x_home, y_home, s_home, c = 'cyan')
for x_pos, y_pos, label in zip(x_home, y_home, n_home):
    ax.annotate(label, xy=(x_pos, y_pos), xytext=(-10,18), textcoords='offset points', ha='left', va='top', fontsize = '10')
plt.show()
fig, ax = plt.subplots(1, figsize=(10,6))
plt.ylabel("xG per Shot")
plt.xlabel("Shots per 90")
fig.suptitle("La Liga: Shot QUALITY VS QUANTITY", fontsize ='17')
ax.set_facecolor("#313332")
ax.scatter(x_away, y_away, s_away, c = 'red', alpha=1)
for x_pos, y_pos, label in zip(x_away, y_away, n_away):
    ax.annotate(label, xy=(x_pos, y_pos), xytext=(-10,18), textcoords='offset points', ha='left', va='top', fontsize = '10', color = 'red')
ax.scatter(x_home, y_home, s_home, c = 'white', alpha=1)
for x_pos, y_pos, label in zip(x_home, y_home, n_home):
    ax.annotate(label, xy=(x_pos, y_pos), xytext=(-10,18), textcoords='offset points', ha='left', va='top', fontsize = '10', color = 'white')
ax.scatter(x_league, y_league, s_league, c = 'black', alpha=0.2)
plt.axvline(league_forward_median_Sh90, linestyle = 'dashed')
plt.axhline(league_forward_median_npxG_perShot, linestyle = 'dashed')
plt.show()
fig, ax = plt.subplots(1, figsize=(10,6))
plt.ylabel("xG per Shot")
plt.xlabel("Shots per 90")
fig.suptitle("La Liga: Shot QUALITY VS QUANTITY", fontsize ='17')
ax.set_facecolor("#313332")
import numpy as np
def f(x):
    return league_forward_median_xG90/x
x=np.setdiff1d(np.linspace(0.5,5.75,100),[0])    
y=f(x)
plt.plot(x, y)    
ax.scatter(x_away, y_away, s_away, c = 'red', alpha=1)
for x_pos, y_pos, label in zip(x_away, y_away, n_away):
    ax.annotate(label, xy=(x_pos, y_pos), xytext=(-10,18), textcoords='offset points', ha='left', va='top', fontsize = '10', color = 'red')
ax.scatter(x_home, y_home, s_home, c = 'white', alpha=1)
for x_pos, y_pos, label in zip(x_home, y_home, n_home):
    ax.annotate(label, xy=(x_pos, y_pos), xytext=(-10,18), textcoords='offset points', ha='left', va='top', fontsize = '10', color = 'white')
ax.scatter(x_league, y_league, s_league, c = 'black', alpha=0.2)
plt.axvline(league_forward_median_Sh90, linestyle = 'dashed')
plt.axhline(league_forward_median_npxG_perShot, linestyle = 'dashed')
plt.show()

import matplotlib as mpl
import matplotlib.font_manager as fm
prop = fm.FontProperties(fname='/Users/DHANANJAYAN/Downloads/Lato/Lato-Regular.ttf')
fig, ax = plt.subplots(1, figsize=(10,6))
plt.ylabel("xG per Shot", color='w', fontsize='18', fontproperties=prop)
plt.xlabel("Shots per 90", color='w', fontsize='18', fontproperties=prop)
fig.suptitle("      La Liga: Forwards of the BIG 3", fontsize ='22',fontproperties=prop, color='w')
ax.set_title("Shot Quantity vs Shot Quality", fontsize ='19',style='italic',fontproperties=prop, color='w')
fig.set_facecolor("#313332")
ax.set_facecolor("#313332") 
ax.tick_params(axis="both", length=0)
mpl.rcParams['xtick.color'] = "white"
mpl.rcParams['ytick.color'] = "white"
ax.grid(ls="dotted", lw=0.4, c="white", zorder=1)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color('white')
ax.spines["bottom"].set_color('white')
ax.scatter(x_away, y_away, s_away, c = '#004d98', alpha=1, zorder=7)
for x_pos, y_pos, label in zip(x_away, y_away, n_away):
    ax.annotate(label, xy=(x_pos, y_pos), xytext=(-10,18), textcoords='offset points', ha='left', va='top', fontsize = '10', color = '#a50044', zorder=11)
ax.scatter(x_home, y_home, s_home, c = 'white', alpha=1, zorder=7)
for x_pos, y_pos, label in zip(x_home, y_home, n_home):
    ax.annotate(label, xy=(x_pos, y_pos), xytext=(-10,18), textcoords='offset points', ha='left', va='top', fontsize = '10', color = 'white', zorder=11)
ax.scatter(x_neutral, y_neutral, s_neutral, c = '#262f61', alpha=1, zorder=7)
for x_pos, y_pos, label in zip(x_neutral, y_neutral, n_neutral):
    ax.annotate(label, xy=(x_pos, y_pos), xytext=(-10,18), textcoords='offset points', ha='left', va='top', fontsize = '10', color = '#ce3524', zorder=11)    
ax.scatter(x_league, y_league, s_league, c = 'black', alpha=0.2, zorder=3)
plt.axvline(league_forward_median_Sh90, linestyle = 'dashed')
plt.axhline(league_forward_median_npxG_perShot, linestyle = 'dashed')
from PIL import Image
import requests
from io import BytesIO
ax2 = fig.add_axes([0.08,0.88,0.14,0.14])
ax2.axis("off")
url = "https://www.logofootball.net/wp-content/uploads/Real-Madrid-CF-HD-Logo.png"
response = requests.get(url)
img = Image.open(BytesIO(response.content))
ax2.imshow(img)
plt.show()

fig, ax = plt.subplots(1, figsize=(10,6))
plt.ylabel("xG per Shot", color='w', fontsize='18', fontproperties=prop)
plt.xlabel("Shots per 90", color='w', fontsize='18', fontproperties=prop)
fig.suptitle("                                La Liga: Forwards of the BIG 3", fontsize ='26',fontproperties=prop, color='w', va='bottom')
ax.set_title("                             Shot Quantity vs Shot Quality", fontsize ='23',style='italic',fontproperties=prop, color='w', va='bottom')
fig.set_facecolor("#313332")
ax.set_facecolor("#313332") 
ax.tick_params(axis="both", length=0)
mpl.rcParams['xtick.color'] = "white"
mpl.rcParams['ytick.color'] = "white"
ax.grid(ls="dotted", lw=0.4, c="white", zorder=1)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color('white')
ax.spines["bottom"].set_color('white')
ax.scatter(x_away, y_away, s_away, c = '#004d98', alpha=1, zorder=7)
for x_pos, y_pos, label in zip(x_away, y_away, n_away):
    ax.annotate(label, xy=(x_pos, y_pos), xytext=(-10,18), textcoords='offset points', ha='left', va='top', fontsize = '10', color = '#a50044', zorder=11)
ax.scatter(x_home, y_home, s_home, c = 'white', alpha=1, zorder=7)
for x_pos, y_pos, label in zip(x_home, y_home, n_home):
    ax.annotate(label, xy=(x_pos, y_pos), xytext=(-10,18), textcoords='offset points', ha='left', va='top', fontsize = '10', color = 'white', zorder=11)
ax.scatter(x_neutral, y_neutral, s_neutral, c = '#262f61', alpha=1, zorder=7)
for x_pos, y_pos, label in zip(x_neutral, y_neutral, n_neutral):
    ax.annotate(label, xy=(x_pos, y_pos), xytext=(-10,18), textcoords='offset points', ha='left', va='top', fontsize = '10', color = '#ce3524', zorder=11)    
ax.scatter(x_league, y_league, s_league, c = 'black', alpha=0.2, zorder=3)
plt.axvline(league_forward_median_Sh90, linestyle = 'dashed')
plt.axhline(league_forward_median_npxG_perShot, linestyle = 'dashed')
ax2 = fig.add_axes([0.07,0.90,0.14,0.14])
ax2.axis("off")
url = "https://www.logofootball.net/wp-content/uploads/Real-Madrid-CF-HD-Logo.png"
response = requests.get(url)
img = Image.open(BytesIO(response.content))
ax2.imshow(img)
ax3 = fig.add_axes([0.15, 0.90, 0.14,0.14])
ax3.axis("off")
url1 = "http://pngimg.com/uploads/fcb_logo/fcb_logo_PNG5.png"
response1 = requests.get(url1)
img1 = Image.open(BytesIO(response1.content))
ax3.imshow(img1)
ax4 = fig.add_axes([0.23, 0.90, 0.14,0.14])
ax4.axis("off")
url2 = "https://logodownload.org/wp-content/uploads/2017/02/atletico-madrid-logo.png"
response2 = requests.get(url2)
img2 = Image.open(BytesIO(response2.content))
ax4.imshow(img2)
plt.show()



import pandas as pd
import matplotlib.pyplot as plt
DATADIR ='/Users/Dhananjayan/Onedrive/Documents'
home_team = 'real1'
away_team = 'barca1'
neutral_team = 'atleti1'
home_shotfile = '/%s_elclasico.csv' % (home_team)
away_shotfile = '/%s_elclasico.csv' % (away_team)
neutral_shotfile = '/%s_elclasico.csv' % (neutral_team)
neutral_shot_data = pd.read_csv('{}/{}'.format(DATADIR, neutral_shotfile))
neutral_forward_shot_data = neutral_shot_data[neutral_shot_data['Pos'].str.contains('FW')]
home_shot_data = pd.read_csv('{}/{}'.format(DATADIR, home_shotfile))
away_shot_data = pd.read_csv('{}/{}'.format(DATADIR, away_shotfile))
home_forward_shot_data = home_shot_data[home_shot_data['Pos'].str.contains('FW')]
away_forward_shot_data = away_shot_data[away_shot_data['Pos'].str.contains('FW')]
away_forward_xGperShot = away_forward_shot_data[['Player','npxG/Sh']]
home_forward_xGperShot = home_forward_shot_data[['Player','npxG/Sh']]
home_forward_Shots_per90 = home_forward_shot_data[['Player','Sh/90']]
away_forward_Shots_per90 = away_forward_shot_data[['Player','Sh/90']]
neutral_forward_xGperShot = neutral_forward_shot_data[['Player','npxG/Sh']]
neutral_forward_Shots_per90 = neutral_forward_shot_data[['Player','Sh/90']]
away_forward_shot_data['plotsize'] = away_forward_shot_data['npxG']*100
home_forward_shot_data['plotsize'] = home_forward_shot_data['npxG']*100
neutral_forward_shot_data['plotsize'] = neutral_forward_shot_data['npxG']*100
league = 'laliga2'
league_shotfile = '/%s_elclasico.csv' % (league)
league_shot_data = pd.read_csv('{}/{}'.format(DATADIR, league_shotfile))
league_forward_shot_data = league_shot_data[league_shot_data['Pos'].str.contains('FW')]
league_forward_shot_data['plotsize'] = league_forward_shot_data['npxG']*100
league_forward_median_Sh90 = league_forward_shot_data['Sh/90'].median()
league_forward_median_xG90 = league_forward_shot_data['xG'].median()
league_forward_median_npxG_perShot = league_forward_shot_data['npxG/Sh'].median()
x_away = away_forward_shot_data['Sh/90']
y_away = away_forward_shot_data['npxG/Sh']
s_away = away_forward_shot_data['plotsize']
n_away = away_forward_shot_data['Player']
x_neutral = neutral_forward_shot_data['Sh/90']
y_neutral = neutral_forward_shot_data['npxG/Sh']
s_neutral = neutral_forward_shot_data['plotsize']
n_neutral = neutral_forward_shot_data['Player']
fig, ax = plt.subplots(1, figsize=(10,6))
plt.ylabel("xG per Shot")
plt.xlabel("Shots per 90")
fig.suptitle("QUALITY VS QUANTITY of Shots in La Liga: Barcelona", fontsize ='17')
ax.scatter(x_away, y_away, s_away, c = 'red')
for x_pos, y_pos, label in zip(x_away, y_away, n_away):
    ax.annotate(label, xy=(x_pos, y_pos), xytext=(-7,14), textcoords='offset points', ha='left', va='top', fontsize = '10')
plt.show()
x_home = home_forward_shot_data['Sh/90']
y_home = home_forward_shot_data['npxG/Sh']
s_home = home_forward_shot_data['plotsize']
n_home = home_forward_shot_data['Player']
fig, ax = plt.subplots(1, figsize=(10,6))
plt.ylabel("xG per Shot")
plt.xlabel("Shots per 90")
fig.suptitle("QUALITY VS QUANTITY of Shots in La Liga: Real Madrid", fontsize ='17')
ax.scatter(x_home, y_home, s_home, c = 'cyan')
for x_pos, y_pos, label in zip(x_home, y_home, n_home):
    ax.annotate(label, xy=(x_pos, y_pos), xytext=(-7,14), textcoords='offset points', ha='left', va='top', fontsize = '10')
plt.show()
x_league = league_forward_shot_data['Sh/90']
y_league = league_forward_shot_data['npxG/Sh']
s_league = league_forward_shot_data['plotsize']
n_league = league_forward_shot_data['Player']
fig, ax = plt.subplots(1, figsize=(10,6))
plt.ylabel("xG per Shot")
plt.xlabel("Shots per 90")
fig.suptitle("QUALITY VS QUANTITY of Shots in La Liga", fontsize ='17')
ax.scatter(x_league, y_league, s_league, c = 'black', alpha = 0.4)
for x_pos, y_pos, label in zip(x_league, y_league, n_league):
    ax.annotate(label, xy=(x_pos, y_pos), xytext=(-7,14), textcoords='offset points', ha='left', va='top', fontsize = '10', alpha = 0.4 , zorder = 's_league')
plt.show()
fig, ax = plt.subplots(1, figsize=(10,6))
plt.ylabel("xG per Shot")
plt.xlabel("Shots per 90")
fig.suptitle("QUALITY VS QUANTITY of Shots in La Liga: Barcelona vs Real Madrid", fontsize ='17')
ax.scatter(x_away, y_away, s_away, c = 'red')
for x_pos, y_pos, label in zip(x_away, y_away, n_away):
    ax.annotate(label, xy=(x_pos, y_pos), xytext=(-10,18), textcoords='offset points', ha='left', va='top', fontsize = '10')
ax.scatter(x_home, y_home, s_home, c = 'cyan')
for x_pos, y_pos, label in zip(x_home, y_home, n_home):
    ax.annotate(label, xy=(x_pos, y_pos), xytext=(-10,18), textcoords='offset points', ha='left', va='top', fontsize = '10')
plt.show()
fig, ax = plt.subplots(1, figsize=(10,6))
plt.ylabel("xG per Shot")
plt.xlabel("Shots per 90")
fig.suptitle("La Liga: Shot QUALITY VS QUANTITY", fontsize ='17')
ax.set_facecolor("#313332")
ax.scatter(x_away, y_away, s_away, c = 'red', alpha=1)
for x_pos, y_pos, label in zip(x_away, y_away, n_away):
    ax.annotate(label, xy=(x_pos, y_pos), xytext=(-10,18), textcoords='offset points', ha='left', va='top', fontsize = '10', color = 'red')
ax.scatter(x_home, y_home, s_home, c = 'white', alpha=1)
for x_pos, y_pos, label in zip(x_home, y_home, n_home):
    ax.annotate(label, xy=(x_pos, y_pos), xytext=(-10,18), textcoords='offset points', ha='left', va='top', fontsize = '10', color = 'white')
ax.scatter(x_league, y_league, s_league, c = 'black', alpha=0.4)
plt.axvline(league_forward_median_Sh90, linestyle = 'dashed')
plt.axhline(league_forward_median_npxG_perShot, linestyle = 'dashed')
plt.show()
fig, ax = plt.subplots(1, figsize=(10,6))
plt.ylabel("xG per Shot")
plt.xlabel("Shots per 90")
fig.suptitle("La Liga: Shot QUALITY VS QUANTITY", fontsize ='17')
ax.set_facecolor("#313332")
import numpy as np
def f(x):
    return league_forward_median_xG90/x
x=np.setdiff1d(np.linspace(0.5,5.75,100),[0])    
y=f(x)
plt.plot(x, y)    
ax.scatter(x_away, y_away, s_away, c = 'red', alpha=1)
for x_pos, y_pos, label in zip(x_away, y_away, n_away):
    ax.annotate(label, xy=(x_pos, y_pos), xytext=(-10,18), textcoords='offset points', ha='left', va='top', fontsize = '10', color = 'red')
ax.scatter(x_home, y_home, s_home, c = 'white', alpha=1)
for x_pos, y_pos, label in zip(x_home, y_home, n_home):
    ax.annotate(label, xy=(x_pos, y_pos), xytext=(-10,18), textcoords='offset points', ha='left', va='top', fontsize = '10', color = 'white')
ax.scatter(x_league, y_league, s_league, c = 'black', alpha=0.45)
plt.axvline(league_forward_median_Sh90, linestyle = 'dashed')
plt.axhline(league_forward_median_npxG_perShot, linestyle = 'dashed')
plt.show()

import matplotlib as mpl
import matplotlib.font_manager as fm
prop = fm.FontProperties(fname='/Users/DHANANJAYAN/Downloads/Lato/Lato-Regular.ttf')
fig, ax = plt.subplots(1, figsize=(10,6))
plt.ylabel("xG per Shot", color='w', fontsize='18', fontproperties=prop)
plt.xlabel("Shots per 90", color='w', fontsize='18', fontproperties=prop)
fig.suptitle("      La Liga: Forwards of the BIG 3", fontsize ='22',fontproperties=prop, color='w')
ax.set_title("Shot Quantity vs Shot Quality", fontsize ='19',style='italic',fontproperties=prop, color='w')
fig.set_facecolor("#313332")
ax.set_facecolor("#313332") 
ax.tick_params(axis="both", length=0)
mpl.rcParams['xtick.color'] = "white"
mpl.rcParams['ytick.color'] = "white"
ax.grid(ls="dotted", lw=0.4, c="white", zorder=1)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color('white')
ax.spines["bottom"].set_color('white')
ax.scatter(x_away, y_away, s_away, c = '#004d98', alpha=1, zorder=7)
for x_pos, y_pos, label in zip(x_away, y_away, n_away):
    ax.annotate(label, xy=(x_pos, y_pos), xytext=(-10,18), textcoords='offset points', ha='left', va='top', fontsize = '10', color = '#a50044', zorder=11)
ax.scatter(x_home, y_home, s_home, c = 'white', alpha=1, zorder=7)
for x_pos, y_pos, label in zip(x_home, y_home, n_home):
    ax.annotate(label, xy=(x_pos, y_pos), xytext=(-10,18), textcoords='offset points', ha='left', va='top', fontsize = '10', color = 'white', zorder=11)
ax.scatter(x_neutral, y_neutral, s_neutral, c = '#262f61', alpha=1, zorder=7)
for x_pos, y_pos, label in zip(x_neutral, y_neutral, n_neutral):
    ax.annotate(label, xy=(x_pos, y_pos), xytext=(-10,18), textcoords='offset points', ha='left', va='top', fontsize = '10', color = '#ce3524', zorder=11)    
ax.scatter(x_league, y_league, s_league, c = 'black', alpha=0.2, zorder=3)
plt.axvline(league_forward_median_Sh90, linestyle = 'dashed')
plt.axhline(league_forward_median_npxG_perShot, linestyle = 'dashed')
from PIL import Image
import requests
from io import BytesIO
ax2 = fig.add_axes([0.08,0.88,0.14,0.14])
ax2.axis("off")
url = "https://www.logofootball.net/wp-content/uploads/Real-Madrid-CF-HD-Logo.png"
response = requests.get(url)
img = Image.open(BytesIO(response.content))
ax2.imshow(img)
plt.show()

fig, ax = plt.subplots(1, figsize=(10,6))
plt.ylabel("xG per Shot", color='w', fontsize='18', fontproperties=prop)
plt.xlabel("Shots per 90", color='w', fontsize='18', fontproperties=prop)
fig.suptitle("                                La Liga: Forwards of the BIG 3", fontsize ='26',fontproperties=prop, color='w', va='bottom')
ax.set_title("                             Shot Quantity vs Shot Quality", fontsize ='23',style='italic',fontproperties=prop, color='w', va='bottom')
fig.set_facecolor("#313332")
ax.set_facecolor("#313332") 
ax.tick_params(axis="both", length=0)
mpl.rcParams['xtick.color'] = "white"
mpl.rcParams['ytick.color'] = "white"
ax.grid(ls="dotted", lw=0.4, c="white", zorder=1)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color('white')
ax.spines["bottom"].set_color('white')
ax.scatter(x_away, y_away, s_away, c = '#FF00FF', alpha=1, zorder=7)
for x_pos, y_pos, label in zip(x_away, y_away, n_away):
    ax.annotate(label, xy=(x_pos, y_pos), xytext=(-10,18), textcoords='offset points', ha='left', va='top', fontsize = '10', color = '#FFFF00', zorder=11)
ax.scatter(x_home, y_home, s_home, c = 'white', alpha=1, zorder=7)
for x_pos, y_pos, label in zip(x_home, y_home, n_home):
    ax.annotate(label, xy=(x_pos, y_pos), xytext=(-10,18), textcoords='offset points', ha='left', va='top', fontsize = '10', color = 'white', zorder=11)
ax.scatter(x_neutral, y_neutral, s_neutral, c = '#00FF00', alpha=1, zorder=7)
for x_pos, y_pos, label in zip(x_neutral, y_neutral, n_neutral):
    ax.annotate(label, xy=(x_pos, y_pos), xytext=(-10,18), textcoords='offset points', ha='left', va='top', fontsize = '10', color = '#00FFFF', zorder=11)    
ax.scatter(x_league, y_league, s_league, c = 'black', alpha=0.4 , zorder=3)
plt.axvline(league_forward_median_Sh90, linestyle = 'dashed')
plt.axhline(league_forward_median_npxG_perShot, linestyle = 'dashed')
ax2 = fig.add_axes([0.07,0.90,0.14,0.14])
ax2.axis("off")
url = "https://www.logofootball.net/wp-content/uploads/Real-Madrid-CF-HD-Logo.png"
response = requests.get(url)
img = Image.open(BytesIO(response.content))
ax2.imshow(img)
ax3 = fig.add_axes([0.15, 0.90, 0.14,0.14])
ax3.axis("off")
url1 = "http://pngimg.com/uploads/fcb_logo/fcb_logo_PNG5.png"
response1 = requests.get(url1)
img1 = Image.open(BytesIO(response1.content))
ax3.imshow(img1)
ax4 = fig.add_axes([0.23, 0.90, 0.14,0.14])
ax4.axis("off")
url2 = "https://logodownload.org/wp-content/uploads/2017/02/atletico-madrid-logo.png"
response2 = requests.get(url2)
img2 = Image.open(BytesIO(response2.content))
ax4.imshow(img2)
plt.show()
