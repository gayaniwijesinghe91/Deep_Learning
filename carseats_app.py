"""Simple Tkinter UI for the CarSeats sales regression model.

Run the last cell of the notebook first (model.save("carseats_model.keras")),
then keep this file in the same folder as carseats_model.keras and run:

    python carseats_app.py
"""
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import pandas as pd
import keras

MODEL_FILE = "carseats_model.keras"

# Column order MUST match X.columns in the notebook
COLUMNS = ['CompPrice', 'Income', 'Advertising', 'Population', 'Price',
           'ShelveLoc', 'Age', 'Education', 'Urban', 'US']

# Numeric fields: (label, default value)
NUMERIC = {
    'CompPrice':   ("Competitor price", 120),
    'Income':      ("Community income level", 70),
    'Advertising': ("Advertising budget", 10),
    'Population':  ("Population size in region", 300),
    'Price':       ("Price charged", 100),
    'Age':         ("Average age of local population", 50),
    'Education':   ("Education level", 15),
}

# Categorical fields: (label, {text shown: encoded value}, default text)
CATEGORICAL = {
    'ShelveLoc': ("Shelf location", {'Bad': 0, 'Medium': 1, 'Good': 2}, 'Good'),
    'Urban':     ("Urban location", {'No': 0, 'Yes': 1}, 'Yes'),
    'US':        ("Store in the US", {'No': 0, 'Yes': 1}, 'Yes'),
}


class App(tk.Tk):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.title("CarSeats Sales Predictor")
        self.resizable(False, False)

        frame = ttk.Frame(self, padding=15)
        frame.grid()

        self.widgets = {}
        row = 0
        for col in COLUMNS:
            if col in NUMERIC:
                label, default = NUMERIC[col]
                var = tk.StringVar(value=str(default))
                w = ttk.Entry(frame, textvariable=var, width=18)
            else:
                label, mapping, default = CATEGORICAL[col]
                var = tk.StringVar(value=default)
                w = ttk.Combobox(frame, textvariable=var, values=list(mapping),
                                 state="readonly", width=15)
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=3, padx=(0, 12))
            w.grid(row=row, column=1, pady=3)
            self.widgets[col] = var
            row += 1

        ttk.Button(frame, text="Predict", command=self.predict).grid(
            row=row, column=0, columnspan=2, pady=(12, 6), sticky="ew")

        self.result = tk.StringVar(value="Predicted sales: -")
        ttk.Label(frame, textvariable=self.result,
                  font=("Segoe UI", 13, "bold")).grid(row=row + 1, column=0, columnspan=2)

    def predict(self):
        values = {}
        try:
            for col in COLUMNS:
                text = self.widgets[col].get().strip()
                if col in NUMERIC:
                    values[col] = float(text)
                else:
                    values[col] = CATEGORICAL[col][1][text]
        except ValueError:
            messagebox.showerror("Invalid input",
                                 f"Please enter a valid number for '{NUMERIC[col][0]}'.")
            return

        row = pd.DataFrame([values], columns=COLUMNS)
        pred = float(self.model.predict(row, verbose=0)[0][0])
        self.result.set(f"Predicted sales: {pred:.2f}")


if __name__ == "__main__":
    try:
        model = keras.models.load_model(MODEL_FILE)
    except Exception as e:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Model not found",
                             f"Could not load '{MODEL_FILE}'.\n"
                             "Run model.save(\"carseats_model.keras\") in the notebook first.\n\n"
                             f"{e}")
        raise SystemExit(1)
    App(model).mainloop()
