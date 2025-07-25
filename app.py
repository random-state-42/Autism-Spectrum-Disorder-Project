from flask import Flask, render_template, request, redirect, url_for,jsonify
import numpy as np
import joblib

# Load model and preprocessing tools
model = joblib.load("autism_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")
app = Flask(__name__)

details = {}

questions = [
    {"text": "Does your child look at you when you call his/her name?", "explanation": "Observe how your child responds when you call their name."},
    {"text": "How easy is it for you to get eye contact with your child?", "explanation": "Consider how readily your child makes eye contact during interactions."},
    {"text": "Does your child point to indicate that they want something?", "explanation": "For example, pointing at a toy that is out of reach."},
    {"text": "Does your child point to share interest with you?", "explanation": "For example, pointing at an interesting sight or object to direct your attention to it."},
    {"text": "Does your child pretend play?", "explanation": "For example, caring for dolls or talking on a toy phone."},
    {"text": "Does your child follow where you're looking?", "explanation": "Notice if your child follows your gaze or looks where you are looking."},
    {"text": "If someone in the family is visibly upset, does your child show signs of wanting to comfort them?", "explanation": "For example, stroking hair, hugging them, or showing concern."},
    {"text": "Would you describe your child's first words as delayed?", "explanation": "Consider the timing and quality of your child's early language development."},
    {"text": "Does your child use simple gestures?", "explanation": "For example, waving goodbye or nodding yes."},
    {"text": "Does your child stare at nothing with no apparent purpose?", "explanation": "Notice if your child appears to fixate on empty space or stares without focus."}
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/features')
def features():
    return render_template('features.html')

@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/survey', methods=['GET', 'POST'])
def survey():
    if request.method == 'POST':
        details.update({
            "age": request.form.get("age"),
            "sex": request.form.get("sex"),
            "ethnicity": request.form.get("ethnicity"),
            "jaundice": request.form.get("jaundice"),
            "asd_history": request.form.get("asdHistory"),
            "respondent": request.form.get("respondent")
        })
        return redirect(url_for('questionnaire', q_num=0))  
    return render_template('basic_form.html')



@app.route('/questionnaire/<int:q_num>', methods=['GET', 'POST'])
def questionnaire(q_num):
    if request.method == 'POST':
        # print(request.form)
        answer = request.form.get('answer')
        details[f"Q{q_num + 1}"] = {"question": questions[q_num]["text"], "answer": answer}  # Store answers
        
        # Redirect to next question or results page
        if q_num + 1 < len(questions):  
            return redirect(url_for('questionnaire', q_num=q_num + 1))  # Move forward normally
        else:
            return redirect(url_for('predict'))  # Redirect to final page

    return render_template('questions.html', question=questions[q_num], q_num=q_num)





# Define expected input features
expected_features = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10',
                     'Age_Mons', 'Sex', 'Ethnicity', 'Jaundice', 'Family_mem_with_ASD', 'Who completed the test']

def transform_input(user_input):
    """Transforms raw user input into the format expected by the model."""
    transformed = {
        "Age_Mons": int(user_input["age"]),
        "Sex": user_input["sex"],
        "Ethnicity": user_input["ethnicity"],
        "Jaundice": user_input["jaundice"],
        "Family_mem_with_ASD": user_input["asd_history"],
        "Who completed the test": user_input["respondent"]
    }

    # Questions where "No" is concerning (value 1), "Yes" is not concerning (value 0)
    for i in [1, 2, 3, 4, 5, 6, 7, 9]:
        q_key = f"Q{i}"
        a_key = f"A{i}"
        transformed[a_key] = 0 if user_input[q_key]["answer"] == "Yes" else 1  # Invert Yes/No responses

    # Questions where "Yes" is concerning (value 1), "No" is not concerning (value 0)
    for i in [8, 10]:
        q_key = f"Q{i}"
        a_key = f"A{i}"
        transformed[a_key] = 1 if user_input[q_key]["answer"] == "Yes" else 0

    return transformed

def analyze_risk(user_input):
    """Identifies potential risk factors and overall risk level."""
    risk_questions = []
    risk_threshold = 3  # Based on Q-Chat-10 scoring guidelines

    # Calculate score by summing concerning responses
    score = sum(user_input.get(f'A{i}', 0) for i in range(1, 11))

    # Identify which questions had concerning answers
    risk_questions = [f'A{i}' for i in range(1, 11) if user_input.get(f'A{i}', 0) == 1]

    # Determine risk level based on scoring thresholds
    if score <= risk_threshold:
        risk_level = "Low"
    elif score <= 7:
        risk_level = "Medium"
    else:
        risk_level = "High"

    return risk_questions, score, risk_level

def preprocess_input(user_input):
    """Preprocesses input by handling missing values and encoding categorical features."""
    user_input = transform_input(user_input)
    
    # Encode categorical features
    for col in ['Sex', 'Ethnicity', 'Jaundice', 'Family_mem_with_ASD', 'Who completed the test']:
        if col in user_input and user_input[col] in label_encoders[col].classes_:
            user_input[col] = label_encoders[col].transform([user_input[col]])[0]
        else:
            user_input[col] = 0  # Default category if missing or unknown

    return user_input

def predict_autism(user_input):
    """Predicts autism risk based on user input dictionary."""
    user_input = preprocess_input(user_input)
    
    input_data = np.array([user_input[col] for col in expected_features]).reshape(1, -1)
    input_data = scaler.transform(input_data)
    prediction = model.predict(input_data)
    
    risk_questions, score, risk_level = analyze_risk(user_input)
    
    return {
        "prediction": label_encoders['Class/ASD Traits '].inverse_transform(prediction)[0],
        "risk_questions": risk_questions,
        "score": score,
        "risk_level": risk_level
    }


@app.route('/predict', methods=['POST','GET'])
def predict():
    try:
        data = details
        if not data:
            return jsonify({"error": "No input data provided"}), 400
        
        result = predict_autism(data)
        details.clear()
        return render_template('results.html', result=result,questions=questions)
    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__=="__main__":
    app.run(debug=True)