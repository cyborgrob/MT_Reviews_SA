import tkinter as tk
from tkinter import simpledialog, messagebox
import pandas as pd

# # Load your reviews (can functionize - only do once)
# df = pd.read_csv(r'C:\Users\rwynn\Desktop\absadataset.csv', header=None, names=['review', 'annotated'])
# df['annotated'] = df['annotated'].fillna('')  # Fill NaNs with empty strings
#
# # Create new df splitting each review into sentences
# new_df = pd.DataFrame(columns=['raw_text'])
# for index, row in df.iterrows():
#     sentences = row['review'].split('.')  # Split review into sentences
#     for sentence in sentences:
#         sentence = sentence.strip()
#         if sentence:
#             new_df.loc[len(new_df)] = [sentence]
#
# new_df.to_csv("sentences.csv")


class ABSAAnnotationApp:
    def __init__(self, master, df_path):
        self.master = master
        self.df_path = df_path
        self.load_data()
        self.current_index = self.find_first_unannotated_index()
        self.aspects = []

        self.master.title("ABSA Annotation Tool")

        # Sentence display widget
        self.text_widget = tk.Text(self.master, wrap="word", height=10, width=60)
        self.text_widget.pack(padx=10, pady=10)

        # Widget to display annotated aspects
        self.aspect_display = tk.Text(self.master, wrap="word", height=8, width=60, bg="light gray")
        self.aspect_display.pack(padx=10, pady=5)
        self.aspect_display.insert("end", "Annotated Aspects:\n")
        self.aspect_display.configure(state="disabled")  # Make the widget read-only

        # Sentiment buttons
        self.pos_button = tk.Button(self.master, text="Positive", command=lambda: self.annotate_aspect("positive"))
        self.pos_button.pack(side="left", padx=5)

        self.neg_button = tk.Button(self.master, text="Negative", command=lambda: self.annotate_aspect("negative"))
        self.neg_button.pack(side="left", padx=5)

        self.neu_button = tk.Button(self.master, text="Neutral", command=lambda: self.annotate_aspect("neutral"))
        self.neu_button.pack(side="left", padx=5)

        # Next sentence button
        self.next_button = tk.Button(self.master, text="Next", command=self.save_and_next)
        self.next_button.pack(side="right", padx=10)

        self.load_sentence()

    def load_data(self):
        """Load or initialize the DataFrame."""
        try:
            self.df = pd.read_csv(self.df_path)
            # Ensure the 'aspectTerms' column exists and fill NaNs with empty strings
            if 'aspectTerms' not in self.df.columns:
                self.df['aspectTerms'] = ''
            else:
                self.df['aspectTerms'].fillna('', inplace=True)
        except FileNotFoundError:
            self.df = pd.DataFrame(columns=['raw_text', 'aspectTerms'])

    def find_first_unannotated_index(self):
        """Find the index of the first sentence without annotations."""
        # Check if 'aspectTerms' is empty
        unannotated = self.df[self.df['aspectTerms'] == ''].index
        if not unannotated.empty:
            return unannotated[0]
        return 0  # Default to 0 if all sentences are annotated or the DataFrame is empty

    def load_sentence(self):
        if self.current_index < len(self.df):
            sentence = self.df.iloc[self.current_index]["raw_text"]
            self.text_widget.delete(1.0, "end")
            self.text_widget.insert("end", sentence)
            self.aspects = []  # Reset for new sentence
            self.update_aspect_display()  # Clear previous aspects display
        else:
            messagebox.showinfo("Completed", "All sentences have been annotated.")
            self.master.quit()

    def annotate_aspect(self, polarity):
        try:
            aspect = self.text_widget.selection_get()
            self.aspects.append({'term': aspect, 'polarity': polarity})
            self.update_aspect_display()
        except tk.TclError:
            messagebox.showwarning("Warning", "No text selected.")

    def update_aspect_display(self):
        self.aspect_display.configure(state="normal")  # Enable widget for editing
        self.aspect_display.delete("1.1", "end")  # Clear existing content
        self.aspect_display.insert("end", "Annotated Aspects:\n")
        for aspect in self.aspects:
            self.aspect_display.insert("end", f"{aspect}\n")
        self.aspect_display.configure(state="disabled")  # Disable widget to prevent user editing

    def save_and_next(self):
        if self.aspects:
            self.df.at[self.current_index, "aspectTerms"] = str(self.aspects)
        self.current_index += 1
        self.save_progress()  # Save progress to CSV
        self.load_sentence()  # Load next sentence

    def save_progress(self):
        self.df.to_csv("annotated_sentences.csv", index=False)



# Creating the main window and passing the DataFrame to the App
root = tk.Tk()
app = ABSAAnnotationApp(root, "annotated_sentences.csv")
root.mainloop()
