#!/usr/bin/env python
# coding: utf-8

# In[31]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[32]:


df_tracks= pd.read_csv('C:/Users/anurag/Documents/Dataset/SpotifyFeatures.csv')
df_tracks.head()


# In[33]:


#null values

pd.isnull(df_tracks).sum()


# In[34]:


df_tracks.info()


# In[35]:


#10 least popular songs

sorted_df= df_tracks.sort_values('popularity',ascending= True).head(10)
sorted_df


# In[36]:


df_tracks.describe().transpose()


# In[67]:


#10 most poular songs on Spotify

df_tracks.drop_duplicates()
most_popular= df_tracks.query('popularity>90',inplace=False).sort_values('popularity',ascending=False)
most_popular[:10]


# In[37]:


#Checking artist in a specific column

df_tracks[["artist_name"]].iloc[18]


# In[38]:


#Converting tracks from ms to sec

df_tracks["duration"]= df_tracks["duration_ms"].apply(lambda x: round(x/1000))
df_tracks.drop("duration_ms", inplace= True , axis=1)
                                                  


# In[39]:


df_tracks.duration.head()


# In[44]:


corr_df=df_tracks.drop(["key","mode"],axis=1).corr(method="pearson")
plt.figure(figsize=(14,6))
heatmap=sns.heatmap(corr_df,annot=True,fmt=".1g",vmin=-1,vmax=1,center=0,cmap="inferno",linewidths=1,linecolor="Black")
heatmap.set_title("Correlation HeatMap Between Variable")
heatmap.set_xticklabels(heatmap.get_xticklabels(),rotation=90)


# In[46]:


sample_df=df_tracks.sample(int(0.004*len(df_tracks)))


# In[47]:


print(len(sample_df))


# In[50]:


plt.figure(figsize=(10,6))
sns.regplot(data=sample_df,y="loudness",x="energy",color="c").set(title="Loudness vs Energy Correlation")


# In[51]:


plt.figure(figsize=(10,6))
sns.regplot(data=sample_df,y="popularity",x="acousticness",color="b").set(title="Popularity vs Acousticness Correlation")


# In[ ]:




