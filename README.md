# Titanic-Survival-Prediction
This repository contains files to create a web application for predicting survival on the Titanic ship. The titanic.csv dataset was used to create the model, which contains information about the passengers on the ship. The model used in this case is called LinearRegression. The web application is made with the Streamlit framework.
## Files
* titanic.csv: dataset used to train the model.
* main.py: application file.
* requirements.txt: requirements for your virtual environment.
* titanic_model.ipynb: Jupyter research file on Titanic data and how the model works.
* titanic_model.joblib: model from research file.
## How to use
1. Clone the repository to your computer.
2. Open a terminal and navigate to the repository folder.
3. Install the required dependencies by running pip install -r requirements.txt.
4. Run the main.py script in the terminal and launch the web application using the command: python -m streamlit run main.py.
5. Enter values for each of the parameters and press Enter.
6. The script will output a prediction on whether or not the passenger survived on the ship.
## Example
You can try this app by following this link: https://nicksttar-titanic-survival-prediction.streamlit.app/
## License
This project is licensed under the MIT License. See the LICENSE file for details.