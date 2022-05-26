import ijson
import pandas as pd
filename = "D:\Study\Python Codes\ALSAlgorithm\Data\\train.json"
data = []
with open(filename, 'r') as f:
    objects = ijson.items(f, 'meta.view.columns.item')
    columns = list(objects)
column_names = [col["fieldName"] for col in columns]

good_columns = ['reviewerID','itemID','rating']

with open(filename, 'r') as f:
    objects = ijson.items(f, 'data.item')
    for row in objects:
        selected_row = []
        for item in good_columns:
            selected_row.append(row[column_names.index(item)])
        data.append(selected_row)
df = pd.DataFrame(data, columns=good_columns)