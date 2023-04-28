from tkinter import *
import tkinter as tk
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from xgboost import XGBClassifier

# Data Input
path = "C:/Users/User/Desktop/Instagram/instagramcomment.csv"
df = pd.read_csv(path)

# Converting dataset to lowercase letters
df['Comments'] = df['Comments'].apply(lambda x: " ".join(x.lower() for x in x.split()))
# Removing Puncuation from dataset
df['Comments'] = df['Comments'].str.replace('[^\w\s]', '')


def removeSpecialCharacter(v):
    c = "".join([r for r in v if ('A' <= r <= 'Z') or ('a' <= r <= 'z') or (r == " ")])
    return c


df['Comments'] = df['Comments'].apply(lambda x: removeSpecialCharacter(str(x)))
df['Comments'].replace('', np.nan, inplace=True)

df.dropna(subset=['Comments'], inplace=True)


# removing numbers from text
def remove_digits(txt):
    txt_nodigit = "".join([char for char in txt if not char.isdigit()])
    return txt_nodigit


df['Comments'] = df['Comments'].apply(lambda x: remove_digits(x))
df

clf = CountVectorizer()
fb = clf.fit_transform(df['Comments'])

tf_transformer = TfidfTransformer(use_idf=True).fit(fb)
X_tf = tf_transformer.transform(fb)

# XGBoost
xg_clf = XGBClassifier()
xg_clf.fit(X_tf, df['Label'])


# Creating GUI using tkinter
def classify_comment():
    comment = comment_entry.get()
    comment = [comment]
    comment_tf = tf_transformer.transform(clf.transform(comment))
    prediction = xg_clf.predict(comment_tf)
    if prediction == 0:
        result_label.config(text="Not bullying")
    else:
        result_label.config(text="Bullying")


root = Tk()
root.title("Instagram")
root.config(bg='white')
root.iconphoto(True, tk.PhotoImage(file='logo.png'))
root.geometry('850x700')
root.resizable(False, False)

# Create a label for the second image
image2 = tk.PhotoImage(file='picture2.png')
image2_label = tk.Label(image=image2, width=805, bg='white')
image2_label.pack(side='top', pady=10)

# Create a frame for the post image and text
post_frame = Frame(root, bg='white', highlightthickness=1, highlightbackground='gray', highlightcolor='gray', width=400, height=400)
post_frame.pack(side='top', pady=20)

# Add a label for the post image
post_image = tk.PhotoImage(file='userphoto.png')
post_label = tk.Label(post_frame, image=post_image, bg='white')
post_label.pack(side='left', padx=20, pady=20)

# Add an image above the comment frame
image = tk.PhotoImage(file='Picture1.png')
image_label = tk.Label(root, image=image, width=725)
image_label.pack(side='top')

# Add a frame for the comment input and classification result
comment_frame = Frame(root, bg='white', highlightthickness=1, highlightbackground='gray', highlightcolor='gray', width=400, height=100)
comment_frame.pack(side='top', pady=20)

# Add a label and entry for the comment input
comment_label = Label(comment_frame, text="Enter Comment:", bg='white')
comment_label.pack(side='left', padx=20)

comment_entry = Entry(comment_frame, width=30)
comment_entry.pack(side='left', padx=10)

# Add a button to classify the comment
classify_button = Button(comment_frame, text="Classify Comment", command=classify_comment, bg='gray', fg='white')
classify_button.pack(side='left', padx=20)

# Add a label to display the classification result
result_label = Label(comment_frame, text="", bg='white')
result_label.pack(side='left', padx=20)

root.mainloop()
