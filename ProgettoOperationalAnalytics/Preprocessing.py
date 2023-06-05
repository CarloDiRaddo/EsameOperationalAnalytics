import csv
from datetime import datetime

import pandas as pd

if __name__ == '__main__':

    #eliminazione colonne dal csv
    def elimina_colonne(file_input, file_output, colonne_da_elim):
        # Leggi il file CSV utilizzando pandas
        dataframe = pd.read_csv(file_input)

        # Elimina le colonne specificate
        dataframe = dataframe.drop(columns=colonne_da_elim)

        # Salva il dataframe in un nuovo file CSV
        dataframe.to_csv(file_output, index=False)


    # Esempio di utilizzo
    file_input = 'final_Liquor_Sales.csv.csv'  # Specifica il percorso del file CSV di input
    file_output = 'output.csv'  # Specifica il percorso del file CSV di output
    colonne_da_elim = ['Date' ]  # Specifica le colonne da eliminare

    elimina_colonne(file_input, file_output, colonne_da_elim)

    #fitraggio per riga POLK
    def filter_csv(input_file, output_file, column_name, filter_value):
        with open(input_file, 'r') as csv_file:
            reader = csv.DictReader(csv_file)
            fieldnames = reader.fieldnames

            filtered_rows = []
            for row in reader:
                if row[column_name] == filter_value:
                    filtered_rows.append(row)

        with open(output_file, 'w', newline='') as csv_output:
            writer = csv.DictWriter(csv_output, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(filtered_rows)


    input_file = 'final_Liquor_Sales.csv.csv'
    output_file = 'output.csv'
    column_name = 'Item Description'
    filter_value = 'POLK'

    filter_csv(input_file, output_file, column_name, filter_value)

    #ordinamento dei valori del CSV per data
    def sort_csv_by_date(input_file, output_file):
        with open(input_file, 'r') as csv_file:
            reader = csv.DictReader(csv_file)
            sorted_rows = sorted(reader, key=lambda row: datetime.strptime(row['Date'], '%m/%d/%Y'))

        with open(output_file, 'w', newline='') as csv_output:
            fieldnames = reader.fieldnames
            writer = csv.DictWriter(csv_output, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sorted_rows)


    input_file = 'final_Liquor_Sales.csv.csv'
    output_file = 'output.csv'

    sort_csv_by_date(input_file, output_file)

print("fine")

