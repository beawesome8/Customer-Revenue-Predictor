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
  
### Create a virtual environment and activate it:
- python -m venv venv
- source venv/bin/activate  # On Windows: venv\Scripts\activate

### Install dependencies:

- pip install -r requirements.txt

### Apply migrations:

- python manage.py migrate

### Run the development server:

- python manage.py runserver

## Usage

1. Access the application at:

- The URL generated

2. Enter the feature values as CSV in the input box.
3. Click the **Predict** button to get the predicted revenue.

## Deployment on Heroku
1. Log in to Heroku:

- ```bash

1. heroku login
2. Create a Heroku app:
  - heroku create customer-revenue-predictor
3. Add a Procfile:
  - echo 'web: gunicorn revenue_predictor.wsgi' > Procfile
4. Push to Heroku:
  - git add .
  - git commit -m "Deploying to Heroku"
  - git push heroku main
5.Open the app:

## Results 

- The model outputs the predicted revenue based on user inputs. The model has been evaluated and tuned for accuracy. The web interface provides a simple way to get predictions in real-time.

## Future Improvements

- Implement additional prediction algorithms.
- Integrate interactive visualizations.
- Improve the user interface and UX.
- Add model retraining functionality.
