import tkinter as tk
from tkinter import simpledialog
import pandas as pd

# Load your reviews
df = pd.read_csv(r'C:\Users\rwynn\Desktop\absadataset')  # Update with your actual file path

# Prepare a column for the annotated data
df['annotated'] = None

# Define the annotation function
def annotate_sentence(sentence):
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    aspect_terms = []

    while True:
        # Input the aspect and polarity
        aspect = simpledialog.askstring("Input", "Enter aspect for '{}' (or type 'done'):".format(sentence),
                                        parent=root)
        if aspect == 'done' or aspect is None:
            break
        polarity = simpledialog.askstring("Input", "Enter polarity for '{}' ('positive', 'negative', 'neutral'):".format(aspect),
                                          parent=root)
        aspect_terms.append({'term': aspect, 'polarity': polarity})

    root.destroy()
    return aspect_terms

# Iterate over each sentence
for index, row in df.iterrows():
    sentences = row['review'].split('.')  # Split review into sentences
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:  # If sentence is not empty
            aspects = annotate_sentence(sentence)
            df.at[index, 'annotated'] = str(aspects)  # Convert list to string to save in CSV

# Save the annotated data
df.to_csv('annotated_reviews.csv', index=False)
