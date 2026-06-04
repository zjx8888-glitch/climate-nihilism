import re
import pandas as pd
import os
import emoji
import html
import ftfy
import numpy as np


filepath = "data.csv"

print(os.path.exists(filepath))  # Check if the file exists
df = pd.read_csv(filepath)  # Load the CSV file, skipping bad lines
df = df[~df['body'].str.contains('bot', na=False)]
df = df[~df['body'].str.contains('automod', na=False)]
df['body'] = df['body'].str.replace(r"\s+", " ", regex=True) #remove extra spaces
df['body'] = df['body'].str.replace(r"\*\*", "", regex=True) #remove bold markdown
df['body'] = df['body'].str.replace(r"\*", "", regex=True) #remove italics markdown
df['body'] = df['body'].str.replace(r"\~\~", "", regex=True) #remove strikethrough markdown
p = r'https?://\S+|www\.\S+'
df['body'] = df['body'].str.replace(p, '[URL]', regex=True)  # Replace URLs with [URL]
df['body'] = df['body'].apply(lambda x: emoji.demojize(x))  # Remove emojis
df['body'] = df['body'].apply(lambda x: html.unescape(x))  # convert HTML to human-readable text
df['body'] = df['body'].apply(lambda x: ftfy.fix_text(x))  # Fix text encoding issues
df['body'] = df['body'].apply(lambda x: re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', x))  # Remove links, keeping the link text
df['body'] = df['body'].apply(ftfy.fix_text)
df['body'] = df['body'].str.replace(r'(?<!\w)(?:/u/|u/)S+', '[USERNAME]', regex=True) 
df['body'] = df['body'].str.replace(r'(?<!\w)(?:/r/|r/)(\w+)', '[SUBREDDIT:\\1]', regex=True)

df.drop_duplicates(subset=['body'], keep='first', inplace=True)  # Remove duplicate comments based on the 'body' column


df.to_csv("preprocessed_data.csv", index=False)  # Save the preprocessed data to a new CSV file