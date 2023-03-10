# Gender prediction based on Name LSTM
This ipython notebook is working to build a model which will predict the gender based on the names. 
The dataset used has been taken from here:  <a href="https://www.kaggle.com/datasets/shrikrishnaparab/gender-based-names">Kaggle: Gender Based Names</a>  
You can follow the analysis on <a href="https://www.kaggle.com/code/shrikrishnaparab/gender-prediction-based-on-name-using-lstm">kaggle</a>
 
## Packeges Used:
 ![Python][python] ![TensorFlow][tensor-image] ![scikit-learn][sklearn-image] ![Pandas][Pandas-image] ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white) ![Jupyter Notebook][ipython-image] ![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
 
[python]: https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54
[tensor-image]:https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white
[sklearn-image]:https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white
[Pandas-image]: https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white
[ipython-image]: https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white


## Process:
![Process](ml-lifecycle.png)

### The ML Process can be used to build app architecture for our problem statement.  
### Process flow is as follows:  
    - Data Collection:
      In this project we have use the labeled dataset i.e name labeled with respective gender from Kaggle.
    - Data Preprocessing:  
      1. Lowercase: convert the name to lowercase
      2. Spliting: split each character
      3. Padding: pad with empty set to make the length of the names same.
      4. Encode: encode each char with respective number i.e. a=1, b=2 and so on till z=26. Blank char " "=0. 
      5. Embeddings: represent each name as a embedding using above encoding.
    - Model Training:
      1. Train a Bidirectional LSTM
    - Model Testing
      1. Test the trained model using some test set
    - Deployment
      2. Deploy the model and use to make a Webapplication. We have used Stremlit and HuggingFace to Deploy.

## Deployment:
Streamlit is used to build a front-end for the Gender prediction app and is deployed on huggingface.co and streamlit.
![app](app.png)
### Check the app:
[![Open in Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Shrikrishna/Gender_predictions_based_on_name)
### [![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://shreeparab1890-gender-prediction-based-on-name-lstm-app-o6k25d.streamlit.app/)
