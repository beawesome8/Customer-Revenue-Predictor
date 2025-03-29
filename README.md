# Customer Revenue Predictor

## Project Description

The Customer Revenue Predictor is a deep learning-based web application built using Django and TensorFlow. This project aims to predict customer revenue based on input features using a neural network model. The web interface is built using Django, and the model prediction is served dynamically via an API endpoint. The application is deployed on Heroku for easy accessibility.

## Motivation

Accurately predicting customer revenue helps businesses make data-driven decisions regarding marketing strategies and customer retention plans. This project aims to demonstrate the effectiveness of neural networks in revenue prediction while providing a user-friendly web interface.

## Technologies and Libraries Used

- Python
- Django
- TensorFlow
- NumPy
- Pandas
- Gunicorn
- Heroku
- HTML, CSS, JavaScript

## Installation and Setup

### Clone the repository:

- git clone https://github.com/username/Customer_Revenue_Predictor.git
- cd Customer_Revenue_Predictor
- Create a virtual environment and activate it:
- python -m venv venv
- source venv/bin/activate  # On Windows: venv\Scripts\activate

## Install dependencies:

- pip install -r requirements.txt

### Apply migrations:

- python manage.py migrate

### Run the development server:

- python manage.py runserver

## Future Improvements

- Implement additional prediction algorithms.
- Integrate interactive visualizations.
- Improve the user interface and UX.
- Add model retraining functionality.
