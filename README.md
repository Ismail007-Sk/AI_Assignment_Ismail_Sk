# Customer Support Ticket Classification System

An NLP-based Customer Support Ticket Classification System that automatically categorizes customer support tickets using Machine Learning.

The system compares three classification algorithms — Multinomial Naive Bayes, Logistic Regression, and Linear SVM — using a TF-IDF-based text classification pipeline. Based on the comparison, Multinomial Naive Bayes is selected as the final model.

The project also includes an optional AI-assisted customer response feature using the Groq API and `openai/gpt-oss-120b`. After predicting the ticket category, the system uses the ticket description and predicted category to generate a short, professional response with relevant guidance.

---

## 1. Project Overview

Customer support teams receive a large number of tickets describing different types of issues. Manually categorizing each ticket can be time-consuming and inefficient.

This project provides an automated solution that:

- Accepts a customer support ticket description.
- Cleans and processes the text using NLP techniques.
- Converts the text into numerical features using TF-IDF.
- Predicts the appropriate ticket category using Machine Learning.
- Provides a prediction confidence score.
- Generates an AI-assisted customer response using an LLM.
- Provides an interactive Streamlit interface for testing the system.

---

## 2. Problem Statement

Customer support operations involve handling a wide variety of issues, and each ticket needs to be assigned to the appropriate category before it can be handled efficiently. When this categorization is performed manually, the process depends on human review for every incoming ticket.

The challenge is to determine whether ticket descriptions alone contain enough useful information to automatically identify the type of issue being reported. This project addresses that challenge by treating ticket categorization as a text-classification problem and evaluating different Machine Learning approaches on the same dataset.

The project also explores how an AI-assisted response can provide immediate basic guidance to the customer after the ticket has been classified.

---

## 3. Technologies Used

### Programming Language

- Python 3.10

### Data Processing

- Pandas
- NumPy

### Machine Learning

- Scikit-learn
- TF-IDF Vectorization
- Multinomial Naive Bayes
- Logistic Regression
- Linear SVM
- GridSearchCV
- 5-Fold Cross-Validation

### Data Visualization

- Matplotlib
- Seaborn

### Development Environment

- Jupyter Notebook
- VS Code

### Web Application

- Streamlit

### LLM Integration

- Groq API
- `openai/gpt-oss-120b`

### Model Saving

- Joblib

### Environment Variables

- python-dotenv

---

## 4. Dataset Information

The project uses a synthetic dataset created specifically according to the requirements of the assignment.

The dataset contains:

- **Total records:** 250
- **Number of categories:** 5

### Categories

1. Login Issue
2. Application Error
3. Report
4. Account Update
5. Performance

### Dataset Fields

| Field | Description |
|---|---|
| `ticket_id` | Unique identifier of the ticket |
| `ticket_description` | Description of the customer's issue |
| `category` | Category assigned to the ticket |
| `priority` | Priority level of the ticket |
| `status` | Current ticket status |

The final dataset is stored in:

```text
data/tickets.csv
```

## 5. Project Structure
```text
AI_Assignment_Ismail_Sk/
│
├── data/
│   └── tickets.csv
│
├── model/
│   ├── linear_svm_model.pkl
│   ├── logistic_regression_model.pkl
│   └── multinomial_nb_model.pkl
│
├── notebooks/
│   ├── create_dataset.ipynb
│   ├── preprocessing_eda.ipynb
│   ├── train.ipynb
│   └── predict.ipynb
│
├── screenshots/
│   ├── Linear SVM/
│   ├── Logistic Regression/
│   └── Multinomial Naive Bayes/
│
├── src/
│   └── app.py
│ 
├── .gitignore
├── .env
├── README.md
└── requirements.txt
```


## 6. Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/Ismail007-Sk/AI_Assignment_Ismail_Sk.git
cd AI_Assignment_Ismail_Sk
```
### Step 2: Create a Virtual Environment
Create and activate the environment on Windows:
```bash
uv venv --python 3.10.0
.venv\Scripts\activate
```
### Step 3: Install Dependencies
```bash
uv pip install -r requirements.txt
```

## 7. Required Dependencies
The required Python packages are listed in: requirements.txt
The project requires libraries for:
- Data processing
- Machine Learning
- NLP
- Visualization
- Model serialization
- Streamlit
- Groq API integration
- Environment variable management

## 8. API Key Configuration
The Groq API key is stored in an environment variable.
Create a .env file in the project root:
```env
GROQ_API_KEY=your_api_key_here
```


## 9. Dataset Creation
The dataset can be created using: notebooks/create_dataset.ipynb
This notebook creates the synthetic customer support ticket dataset according to the project requirements.
The generated dataset is saved as: data/tickets.csv

## 10. Data Preprocessing and EDA
Data preprocessing and exploratory data analysis are performed in: notebooks/preprocessing_eda.ipynb

The notebook performs tasks such as:
- Loading the dataset
- Checking dataset structure
- Checking missing values
- Checking duplicate records
- Examining category distribution
- Examining priority distribution
- Examining ticket status
- Analyzing category and priority relationships
- Performing text preprocessing
- Text Preprocessing

Ticket descriptions are processed by:
- Converting text to lowercase
- Removing special characters
- Removing unnecessary whitespace
- Removing English stop words during TF-IDF processing

This helps provide cleaner text input to the Machine Learning pipeline.


## 11. Model Training
Model training is performed in: notebooks/train.ipynb

Three classification algorithms are trained and compared:
- Multinomial Naive Bayes
- Logistic Regression
- Linear SVM

All three models use the same basic text-classification approach:
Ticket Description->Text Preprocessing->TF-IDF Vectorization->Classification Model->Predicted Category

Training and Validation
- 80% Training Data
- 20% Testing Data

For model tuning, GridSearchCV with 5-fold cross-validation is used.
For the Multinomial Naive Bayes model, parameters such as:
- alpha
- max_df
- min_df
- max_features
- ngram_range
- sublinear_tf
are tested to find a suitable configuration.

The models are evaluated using metrics such as:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
The trained models are saved in: model/

## 12. Final Model
After comparing the three classification algorithms, Multinomial Naive Bayes is selected as the final model.
The final model is stored as: model/multinomial_nb_model.pkl

The model achieved approximately:
- Training Accuracy: 99.5%
- Test Accuracy:     98.0%
- Macro Precision:   98.0%
- Macro Recall:      98.0%
- Macro F1 Score:    98.0%
The model correctly classified 49 out of 50 test tickets.

## 13. Prediction
New ticket predictions can be tested using: notebooks/predict.ipynb
The prediction pipeline automatically applies the required text preprocessing and TF-IDF transformation before generating the prediction.

```text
Example
Input: I forgot my password and cannot login.
Expected Category: Login Issue
Predicted Category: Login Issue

Another example:
Input: The application is running very slowly.
Predicted Category: Performance
```

## 14. How to run the Streamlit Application
The interactive application is located at: src/app.py
Run the application from the project root using: 
```bash
cd src
streamlit run app.py
```
The application provides:
- Customer ticket input
- Predicted category
- Prediction confidence
- AI-assisted customer response

## 15. AI-Assisted Customer Response
The application includes an additional LLM-based feature using the Groq API.
Provider: Groq
Model: openai/gpt-oss-120b

After the Machine Learning model predicts the ticket category, the application provides the following information to the LLM:
- Original ticket description
- Predicted category
The LLM generates a short, professional response focused on providing practical guidance or possible next steps for the customer's issue.

## 16. Sample Input / Output
```text
Sample Input: The application is very slow today.

Machine Learning Prediction
Predicted Category: Performance
Confidence: 89.5%

AI-Assisted Response
Hi,
I’m sorry you’re experiencing slow performance today. Please try the following steps:
1. Close the app completely and reopen it.
2. Clear the app’s cache (you can usually find this in the app settings).
3. Ensure you have a stable internet connection and, if possible, switch to a different network or Wi‑Fi.
4. Check for any available app updates and install them.
If the issue continues after these steps, let us know and we’ll investigate further. Thank you for your patience.
```

## 17. Limitations
The current implementation has several limitations:
- The dataset contains only 250 records.
- The dataset is synthetic and does not fully represent real customer language.
- Only five ticket categories are supported.
- Synthetic ticket descriptions may follow similar writing patterns.
- Real customers may describe the same problem using very different wording.
- The model may therefore perform differently on a larger real-world dataset.
- The current dataset is suitable for demonstrating and evaluating a basic NLP - classification approach, but is not sufficient by itself for a production-level support system.


## 18. Future Improvements
If this system were developed for a real company, it could be improved by:
- Collecting a much larger dataset from real customer support tickets.
- Adding more categories and subcategories.
- Improving NLP preprocessing.
- Testing more advanced Machine Learning and deep learning models.
- Exploring LLM-based classification.
- Continuously retraining the model using newly resolved tickets.
- Incorporating feedback from human support agents.
- Adding confidence-based human review for uncertain predictions.
- Exposing the classifier through a production API.
- Integrating a database for storing and managing support tickets.
- Adding authentication and role-based access for support staff.
- Monitoring model performance after deployment.

## 19. Project Links
GitHub Repository: <YOUR_GITHUB_LINK>
Live Application: <YOUR_LIVE_APPLICATION_LINK>

## Author
Ismail Sheikh

B.Tech Graduate
MCKV Institute of Engineering