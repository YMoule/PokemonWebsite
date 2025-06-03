from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    result = ''
    poke1_img = poke2_img = ''
    poke1 = poke2 = ''
    if request.method == 'POST':
        poke1 = request.form['poke1']
        poke2 = request.form['poke2']
        result = make_prediction(poke1, poke2)
        poke1_img = poke1.strip().lower()
        poke2_img = poke2.strip().lower()

    return render_template('pokemon.html', result=result, poke1 = poke1, poke2 = poke2, poke1_img=poke1_img, poke2_img=poke2_img)

def make_prediction(poke1, poke2):
    try:
        import joblib
        import pandas as pd
        import numpy as np
        model_file_name = "mlp2_best_model.joblib"

        loaded_model = joblib.load(open(model_file_name, 'rb'))

        pokemon_data = pd.read_csv('pokemon.csv', encoding='latin1')	# Ensures special characters are read correctly
        # Normalize numeric values
        names = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
        # Equivalent to using MinMaxScaler()
        for col in names:
            pokemon_data[col] = (pokemon_data[col] - pokemon_data[col].min()) / (pokemon_data[col].max() - pokemon_data[col].min()) 

        # Performs one hot encoding on the 'Type 1' column
        pokemon_data = pd.get_dummies(pokemon_data, columns=['Type 1', ], dtype=int)
        pokemon_data =pd.get_dummies(pokemon_data, columns=['Type 2', ], dtype=int)


        #Dropping Features and encoding True/False
        pokemon_data = pokemon_data.drop('Generation', axis=1)
        pokemon_data = pokemon_data.drop('#', axis=1)

        # To convert 'Legendary' column from boolean to integer type
        pokemon_data['Legendary'] = pokemon_data['Legendary'].astype(int)

        if poke1 in pokemon_data['Name'].values and poke2 in pokemon_data['Name'].values and poke1 != poke2:
            pokemon1 = pokemon_data[pokemon_data['Name'] == poke1].iloc[0]
            pokemon2 = pokemon_data[pokemon_data['Name'] == poke2].iloc[0]

            # Drop the 'Name' column before prediction, then combine the two
            inputs = pd.concat([pokemon1.drop('Name'), pokemon2.drop('Name')]).values.reshape(1, -1)

            prediction = loaded_model.predict(inputs)

            info = ''

            if (prediction == 0):
                info = poke1
            else:
                info = poke2
        
            final_info = "The winner is {}".format(info)
            return final_info

        
        else:
            final_info = 'These Pokemon are not in the dataset. Please check your spelling and check that the first letter is capitalized.'
            return final_info
    except Exception as e:
        return f"An error occurred: {str(e)}"
    
if __name__ == '__main__':
    app.run(debug=True)