# PART 1: IMPORTS AND LOAD THE MODEL FROM FILE
import os
import tkinter
import pandas as pd
from joblib import load

HERE = os.path.dirname(os.path.abspath(__file__))
model = load(os.path.join(HERE, "linearreg1.joblib"))

# If you scaled the training data, save the scaler in the notebook with
# dump(scaler, "scaler.joblib") and uncomment the next line.
# scaler = load(os.path.join(HERE, "scaler.joblib"))
scaler = None

# Must be in the same order as the training data
COLUMNS = ['CompPrice', 'Income', 'Advertising', 'Population', 'Price',
           'ShelveLoc', 'Age', 'Education', 'Urban', 'US']

SHELF_CODES = {"Bad": 0, "Medium": 1, "Good": 2}
YES_NO = {"No": 0, "Yes": 1}


def predict_sales(row):
    """Takes a dict with all 10 variables and returns predicted sales."""
    df = pd.DataFrame([row])[COLUMNS]
    if scaler is not None:
        df = scaler.transform(df)
    return float(model.predict(df, verbose=0)[0][0])


# PART 2: TEST THE MODEL FIRST WITHOUT THE GUI
test_row = {'CompPrice': 120, 'Income': 70, 'Advertising': 10,
            'Population': 300, 'Price': 100, 'ShelveLoc': 2,
            'Age': 50, 'Education': 15, 'Urban': 1, 'US': 1}
print()
print("Predicted sales for the test store:")
print(f"$ {round(predict_sales(test_row), 2)}")
print("----------------")

# PART 3: CREATE THE GUI
window = tkinter.Tk()
window.title("Car seat sales prediction GUI application")
window.geometry("760x640")
window.option_add("*font", "lucida 14 bold")

# (label shown to user, column name, default value)
NUMERIC_FIELDS = [
    ("Competitor price", "CompPrice", 120),
    ("Community income (x1000 $)", "Income", 70),
    ("Advertising budget", "Advertising", 10),
    ("Population (x1000)", "Population", 300),
    ("Price", "Price", 100),
    ("Customer age", "Age", 50),
    ("Education level", "Education", 15),
]

form = tkinter.Frame(window)
form.pack(pady=15)

entries = {}
row_index = 0
for text, key, default in NUMERIC_FIELDS:
    tkinter.Label(form, text=text, anchor="e", width=26).grid(
        row=row_index, column=0, padx=8, pady=4, sticky="e")
    entry = tkinter.Entry(form, width=12)
    entry.insert(0, str(default))
    entry.grid(row=row_index, column=1, padx=8, pady=4)
    entries[key] = entry
    row_index += 1

# Dropdown fields (shelf location, urban, US)
dropdowns = {}
DROPDOWN_FIELDS = [
    ("Shelf location", "ShelveLoc", SHELF_CODES, "Good"),
    ("Urban store", "Urban", YES_NO, "Yes"),
    ("Store in the US", "US", YES_NO, "Yes"),
]
for text, key, mapping, default in DROPDOWN_FIELDS:
    tkinter.Label(form, text=text, anchor="e", width=26).grid(
        row=row_index, column=0, padx=8, pady=4, sticky="e")
    var = tkinter.StringVar(value=default)
    tkinter.OptionMenu(form, var, *mapping.keys()).grid(
        row=row_index, column=1, padx=8, pady=4, sticky="w")
    dropdowns[key] = (var, mapping)
    row_index += 1

result_var = tkinter.StringVar()
result_var.set("Waiting for user input...")
tkinter.Label(window, textvariable=result_var).pack(pady=10)


def set_text_by_button():
    row = {}
    try:
        for key, entry in entries.items():
            row[key] = float(entry.get())
    except ValueError:
        result_var.set("Incorrect value, use numbers.")
        return  # stop here, do not run the model

    for key, (var, mapping) in dropdowns.items():
        row[key] = mapping[var.get()]

    result = predict_sales(row)
    result_var.set(f"Predicted sales: $ {round(result, 2)}")


tkinter.Button(window, height=1, width=16, text="Predict sales!",
               command=set_text_by_button).pack(pady=10)

window.bind('<Return>', lambda event: set_text_by_button())

window.mainloop()