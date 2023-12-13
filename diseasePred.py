from tkinter import *
import numpy as np
import pandas as pd
from sklearn import tree, ensemble, naive_bayes

# List of symptoms
l1 = ['back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine',
      'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach',
      'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation',
      'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs',
      'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool',
      'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs',
      'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails',
      'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips',
      'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints',
      'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness',
      'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine',
      'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)',
      'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain',
      'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes', 'increased_appetite', 'polyuria',
      'family_history', 'mucoid_sputum',
      'rusty_sputum', 'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion',
      'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen',
      'history_of_alcohol_consumption', 'fluid_overload', 'blood_in_sputum', 'prominent_veins_on_calf',
      'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling',
      'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister', 'red_sore_around_nose',
      'yellow_crust_ooze']

# List of diseases
disease = ['Fungal infection', 'Allergy', 'GERD', 'Chronic cholestasis', 'Drug Reaction',
           'Peptic ulcer diseae', 'AIDS', 'Diabetes', 'Gastroenteritis', 'Bronchial Asthma', 'Hypertension',
           ' Migraine', 'Cervical spondylosis', 'Paralysis (brain hemorrhage)', 'Jaundice', 'Malaria', 'Chicken pox',
           'Dengue', 'Typhoid', 'hepatitis A',
           'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E', 'Alcoholic hepatitis', 'Tuberculosis',
           'Common Cold', 'Pneumonia', 'Dimorphic hemmorhoids(piles)', 'Heartattack', 'Varicoseveins', 'Hypothyroidism',
           'Hyperthyroidism', 'Hypoglycemia', 'Osteoarthristis',
           'Arthritis', '(vertigo) Paroymsal  Positional Vertigo', 'Acne', 'Urinary tract infection', 'Psoriasis',
           'Impetigo']

# Initialize binary list for symptoms
l2 = [0] * len(l1)

# Load data from a CSV file
def load_data(file_path):
    df = pd.read_csv(file_path)
    df.replace({'prognosis': disease_labels}, inplace=True)
    X_data = df[l1]
    y_data = df[['prognosis']]
    np.ravel(y_data)
    return X_data, y_data


# Train models
def train_decision_tree(X_train, y_train):
    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    return clf

def train_random_forest(X_train, y_train):
    clf = ensemble.RandomForestClassifier()
    clf.fit(X_train, np.ravel(y_train))
    return clf

def train_naive_bayes(X_train, y_train):
    clf = naive_bayes.GaussianNB()
    clf.fit(X_train, np.ravel(y_train))
    return clf

# Predict disease based on symptoms and display the result in a Tkinter text box
def predict_disease(symptoms, model, label_text):
    for k in range(len(l1)):
        for z in symptoms:
            if z == l1[k]:
                l2[k] = 1

    input_test = [l2]
    prediction = model.predict(input_test)[0]

    if prediction in disease_labels.values():
        disease_name = [key for key, value in disease_labels.items() if value == prediction][0]
        label_text.delete("1.0", END)
        label_text.insert(END, disease_name)
    else:
        label_text.delete("1.0", END)
        label_text.insert(END, "Not Found")

# Validate user input for name and symptoms
def validate_input(name, symptoms):
    if not name:
        return "Please enter the name."
    elif all(symptom == "None" for symptom in symptoms):
        return "Please select at least one symptom."
    return None

# Display an error message in a Tkinter messagebox
def show_error_message(message):
    messagebox.showerror("Error", message)


# Button click event for Prediction
def on_decision_tree_click():
    name = Name.get()
    symptoms = [Symptom1.get(), Symptom2.get(), Symptom3.get(), Symptom4.get(), Symptom5.get()]

    validation_result = validate_input(name, symptoms)
    if validation_result is None:
        predict_disease(symptoms, decision_tree_model, t1)
    else:
        show_error_message(validation_result)

def on_random_forest_click():
    name = Name.get()
    symptoms = [Symptom1.get(), Symptom2.get(), Symptom3.get(), Symptom4.get(), Symptom5.get()]

    validation_result = validate_input(name, symptoms)
    if validation_result is None:
        predict_disease(symptoms, random_forest_model, t2)
    else:
        show_error_message(validation_result)

def on_naive_bayes_click():
    name = Name.get()
    symptoms = [Symptom1.get(), Symptom2.get(), Symptom3.get(), Symptom4.get(), Symptom5.get()]

    validation_result = validate_input(name, symptoms)
    if validation_result is None:
        predict_disease(symptoms, naive_bayes_model, t3)
    else:
        show_error_message(validation_result)

# Define disease labels using a dictionary
disease_labels = {
    'Fungal infection': 0, 'Allergy': 1, 'GERD': 2, 'Chronic cholestasis': 3, 'Drug Reaction': 4,
    'Peptic ulcer diseae': 5, 'AIDS': 6, 'Diabetes ': 7, 'Gastroenteritis': 8,
    'Bronchial Asthma': 9, 'Hypertension ': 10, 'Migraine': 11, 'Cervical spondylosis': 12,
    'Paralysis (brain hemorrhage)': 13, 'Jaundice': 14, 'Malaria': 15, 'Chicken pox': 16,
    'Dengue': 17, 'Typhoid': 18, 'hepatitis A': 19, 'Hepatitis B': 20, 'Hepatitis C': 21,
    'Hepatitis D': 22, 'Hepatitis E': 23, 'Alcoholic hepatitis': 24, 'Tuberculosis': 25,
    'Common Cold': 26, 'Pneumonia': 27, 'Dimorphic hemmorhoids(piles)': 28, 'Heart attack': 29,
    'Varicose veins': 30, 'Hypothyroidism': 31, 'Hyperthyroidism': 32, 'Hypoglycemia': 33,
    'Osteoarthristis': 34, 'Arthritis': 35, '(vertigo) Paroymsal  Positional Vertigo': 36,
    'Acne': 37, 'Urinary tract infection': 38, 'Psoriasis': 39, 'Impetigo': 40
}

# Load training data
X_train, y_train = load_data("Training.csv")

# Load testing data
X_test, y_test = load_data("Testing.csv")

decision_tree_model = train_decision_tree(X_train, y_train)
random_forest_model = train_random_forest(X_train, y_train)
naive_bayes_model = train_naive_bayes(X_train, y_train)

# Tk class to create a root window
root = Tk()
root.configure(background='#F0F0F0')
root.title('Smart Disease Predictor System')
root.resizable(0, 0)

# Input symptom
Symptom1 = StringVar()
Symptom1.set("Select Here")

Symptom2 = StringVar()
Symptom2.set("Select Here")

Symptom3 = StringVar()
Symptom3.set("Select Here")

Symptom4 = StringVar()
Symptom4.set("Select Here")

Symptom5 = StringVar()
Symptom5.set("Select Here")
Name = StringVar()

# Reset the given inputs
prev_win = None

pred1 = " "
pred2 = " "
pred3 = " "
try:
    prev_win.destroy()
    prev_win = None
except AttributeError:
    pass


def Reset():
    global prev_win

    Symptom1.set("Select Here")
    Symptom2.set("Select Here")
    Symptom3.set("Select Here")
    Symptom4.set("Select Here")
    Symptom5.set("Select Here")

    NameEn.delete(first=0, last=100)

    pred1.set(" ")
    pred2.set(" ")
    pred3.set(" ")
    try:
        prev_win.destroy()
        prev_win = None
    except AttributeError:
        pass

def reset_output_fields():
    t1.delete("1.0", END)
    t2.delete("1.0", END)
    t3.delete("1.0", END)

# Exit buttonm
from tkinter import messagebox
def Exit():
    qExit = messagebox.askyesno("System", "Do you want to exit the system")
    if qExit:
        root.destroy()
        exit()

# Label for title
w2 = Label(root, justify=LEFT, text="Disease Predictor using Machine Learning", fg="#333", bg='#F0F0F0')
w2.config(font=("Times", 30, "bold"))
w2.grid(row=1, column=0, columnspan=2, padx=100)

# Label for the name
NameLb = Label(root, text="Name of the Patient", fg="#333", bg='#F0F0F0')
NameLb.config(font=("Times", 15, "bold"))
NameLb.grid(row=6, column=0, pady=15, sticky=W)

# Labels for the symtoms
S1Lb = Label(root, text="Symptom 1", fg="Black", bg="Ivory")
S1Lb.config(font=("Times", 15, "bold"))
S1Lb.grid(row=7, column=0, pady=10, sticky=W)

S2Lb = Label(root, text="Symptom 2", fg="Black", bg="Ivory")
S2Lb.config(font=("Times", 15, "bold"))
S2Lb.grid(row=8, column=0, pady=10, sticky=W)

S3Lb = Label(root, text="Symptom 3", fg="Black", bg="Ivory")
S3Lb.config(font=("Times", 15, "bold"))
S3Lb.grid(row=9, column=0, pady=10, sticky=W)

S4Lb = Label(root, text="Symptom 4", fg="Black", bg="Ivory")
S4Lb.config(font=("Times", 15, "bold"))
S4Lb.grid(row=10, column=0, pady=10, sticky=W)

S5Lb = Label(root, text="Symptom 5", fg="Black", bg="Ivory")
S5Lb.config(font=("Times", 15, "bold"))
S5Lb.grid(row=11, column=0, pady=10, sticky=W)

# Labels for the different algorithms
lrLb = Label(root, text="DecisionTree", fg="Black", bg="Ivory", width=20)
lrLb.config(font=("Times", 15, "bold"))
lrLb.grid(row=15, column=0, pady=10, sticky=W)

destreeLb = Label(root, text="RandomForest", fg="Black", bg="Ivory", width=20)
destreeLb.config(font=("Times", 15, "bold"))
destreeLb.grid(row=17, column=0, pady=10, sticky=W)

ranfLb = Label(root, text="NaiveBayes", fg="Black", bg="Ivory", width=20)
ranfLb.config(font=("Times", 15, "bold"))
ranfLb.grid(row=19, column=0, pady=10, sticky=W)

OPTIONS = sorted(l1)

# Name as input
NameEn = Entry(root, textvariable=Name)
NameEn.grid(row=6, column=1)

# Symptoms as input from the dropdown
S1 = OptionMenu(root, Symptom1, *OPTIONS)
S1.grid(row=7, column=1)

S2 = OptionMenu(root, Symptom2, *OPTIONS)
S2.grid(row=8, column=1)

S3 = OptionMenu(root, Symptom3, *OPTIONS)
S3.grid(row=9, column=1)

S4 = OptionMenu(root, Symptom4, *OPTIONS)
S4.grid(row=10, column=1)

S5 = OptionMenu(root, Symptom5, *OPTIONS)
S5.grid(row=11, column=1)

# Buttons
dst = Button(root, text="Prediction 1", command=lambda: on_decision_tree_click(), bg="#333", fg="white")
dst.config(font=("Times", 15, "bold"))
dst.grid(row=6, column=3, padx=10)

rnf = Button(root, text="Prediction 2", command=lambda: on_random_forest_click(), bg="#555", fg="#F0F0F0")
rnf.config(font=("Times", 15, "bold"))
rnf.grid(row=7, column=3, padx=10)

lr = Button(root, text="Prediction 3", command=lambda: on_naive_bayes_click(), bg="#222", fg="white")
lr.config(font=("Times", 15, "bold"))
lr.grid(row=8, column=3, padx=10)

rs = Button(root, text="Reset Inputs", command=Reset, bg="#F0F0F0", fg="#333", width=15)
rs.config(font=("Times", 15, "bold"))
rs.grid(row=10, column=3, padx=10)

rs_output = Button(root, text="Reset Output", command=reset_output_fields, bg="#F0F0F0", fg="#333", width=15)
rs_output.config(font=("Times", 15, "bold"))
rs_output.grid(row=11, column=3, padx=10)

ex = Button(root, text="Exit System", command=Exit, bg="#F0F0F0", fg="#333", width=15)
ex.config(font=("Times", 15, "bold"))
ex.grid(row=12, column=3, padx=10)

# Output of different algorithms
t1 = Text(root, font=("Times", 15, "bold"), height=1, bg="#DDD", width=40, fg="#333", relief="sunken")
t1.grid(row=15, column=1, padx=10)

t2 = Text(root, font=("Times", 15, "bold"), height=1, bg="#BBB", width=40, fg="#333", relief="sunken")
t2.grid(row=17, column=1, padx=10)

t3 = Text(root, font=("Times", 15, "bold"), height=1, bg="#999", width=40, fg="#F0F0F0", relief="sunken")
t3.grid(row=19, column=1, padx=10)

# Calling to run the app
root.mainloop()