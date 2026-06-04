import re
import pandas as pd
import os
import emoji
import html
import ftfy
import numpy as np



def preprocess_twitter(str):
    str = str.replace(r"\s+", " ", count=-1) #remove extra spaces
    p = r'https?://\S+|www\.\S+'
    str = re.sub(p, '[URL]', str)  # Replace URLs with [URL]
    p = r'@(\w+)'
    str = re.sub(p, '[USERNAME]', str)  # Replace usernames with [USERNAME] to protect privacy
    str = emoji.demojize(str)  
    str = html.unescape(str)  # change HTML format to normal characters
    str = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', str)  # Remove Markdown links, keeping the link text
    str = ftfy.fix_text(str)  # fix text encoding problems

    return(str)


def convert_to_timestamp(tid):
    tid = tid >> 22
    tid = tid + 1288834974657
    return tid

filepath = "climate_change_tweets.csv"
print(os.path.exists(filepath))  # Check if the file exists
df = pd.read_csv(filepath)  # Load the CSV file, skipping bad lines

mask = df['tweet'].notna()
df.loc[mask, 'tweet'] = df.loc[mask, 'tweet'].apply(lambda x: preprocess_twitter(x))
mask = df['tweetsid'].notna()
df.loc[mask, 'tweetsid'] = df.loc[mask, 'tweetsid'].apply(lambda x: convert_to_timestamp(x))
df.rename(columns={'tweetsid': 'timestamp'})
df.rename(columns={'tweet': 'text'})


df.to_csv("preprocessed_climate_change_tweets.csv", index=False)
