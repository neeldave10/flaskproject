from flask import Flask, render_template, request
import nltk
#from newspaper import Article
import random
#import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
#import warnings
#import numpy as np
import pandas as pd
from urllib import request
#from django.shortcuts import render
#import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import webbrowser
import pickle
import warnings

warnings.filterwarnings('ignore')
import ssl
try:
    _create_unverified_https_context=ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context=_create_unverified_https_context

nltk.download('punkt', quiet=True)

#article=Article('https://www.mayoclinic.org/diseases-conditions/heart-disease/symptoms-causes/syc-20353118')
#article.download()
#article.parse()
article="Diabetes is a disease that occurs when your blood glucose, also called blood sugar, is too high, Blood glucose is your main source of energy and comes from the food you eat, Insulin, a hormone made by the pancreas, helps glucose from food get into your cells to be used for energy, Sometimes your body doesn’t make enough or any insulin or doesn’t use insulin well, Glucose then stays in your blood and doesn’t reach your cells. Over time, having too much glucose in your blood can cause health problems, Although diabetes has no cure, you can take steps to prevent diabetes and stay healthy, Sometimes people call diabetes “a touch of sugar” or “borderline diabetes”. The different types of diabetes are type 1, type 2, and gestational diabetes. TYPE-1: If you have type 1 diabetes, your body does not make insulin, Your immune system attacks and destroys the cells in your pancreas that make insulin., Type 1 diabetes is usually diagnosed in children and young adults, although it can appear at any age, People with type 1 diabetes need to take insulin every day to stay alive, TYPE-2: If you have type 2 diabetes, your body does not make or use insulin well, You can develop type 2 diabetes at any age, even during childhood, However, this type of diabetes occurs most often in middle-aged and older people, Type 2 is the most common type of diabetes, Gestational diabetes: It develops in some women when they are pregnant, Most of the time, this type of diabetes goes away after the baby is born, However, if you’ve had gestational diabetes, you have a greater chance of developing type 2 diabetes later in life, Sometimes diabetes diagnosed during pregnancy is actually type 2 diabetes, Other types: Less common types include monogenic diabetes, which is an inherited form of diabetes, and cystic fibrosis-related diabetes, People diagnosed with diabetes are: As of 2015, 30.3 million people in the United States, or 9.4 percent of the population, had diabetes, More than 1 in 4 of them didn’t know they had the disease, Diabetes affects 1 in 4 people over the age of 65, About 90-95 percent of cases in adults are type 2 diabetes. Over time, diabetes leads to problems such as, heart disease, stroke, kidney disease, eye problems, dental disease, nerve damage, foot problems. Symptoms of diabetes are: Urinate (pee) a lot, often at night, Are very thirsty, Lose weight without trying, Are very hungry, Have blurry vision, Have numb or tingling hands or feet, Feel very tired, Have very dry skin. Ways to prevent diabetes are, 1 Reduce carb intake, 2 Exercise regularly, 3 Drink a lot of water, 4 Try to lose excess weight, 5 Reduce potion sizes, 6 Follow high fibre diet, etc.Treatment for diabetes are: Treatment for type 1 diabetes involves insulin injections or the use of an insulin pump, frequent blood sugar checks, and carbohydrate counting, Treatment of type 2 diabetes primarily involves lifestyle changes, monitoring of your blood sugar, along with diabetes medications, insulin or both.The term “heart disease” refers to several types of heart conditions, The most common type of heart disease is coronary artery disease (CAD), which affects the blood flow to the heart, Decreased blood flow can cause a heart attack. Symptoms of heart disease include, Heart attack: Chest pain or discomfort, upper back or neck pain, indigestion, heartburn, nausea or vomiting, extreme fatigue, upper body discomfort, dizziness, and shortness of breath, Arrhythmia: Fluttering feelings in the chest (palpitations), Heart failure: Shortness of breath, fatigue, or swelling of the feet, ankles, legs, abdomen, or neck veins. In general, treatment for heart disease usually includes: Lifestyle changes: You can lower your risk of heart disease by eating a low-fat and low-sodium diet, getting at least 30 minutes of moderate exercise on most days of the week, quitting smoking, and limiting alcohol intake, Medications: If lifestyle changes alone aren't enough, your doctor may prescribe medications to control your heart disease, The type of medication you receive will depend on the type of heart disease, Medical procedures or surgery: If medications aren't enough, it's possible your doctor will recommend specific procedures or surgery, The type of procedure or surgery will depend on the type of heart disease and the extent of the damage to your heart. Prevention of heart disease: 1 Don’t smoke or use tobacco, 2 Get moving: Aim for minimum 30 mins of daily activity, 3 Eat heart healthy diet, avoid oily food, 4 Maintain healthy weight, 5 Manage stress, 6 Get regular health screenings."
#article.nlp()
#corpus=article.text
#print(corpus)

sentence_list=nltk.sent_tokenize(text=article)
#print(sentence_list)

#Greeting
def greeting(text):
    text=text.lower()

    bot_greeting=['hello','hey, how you doing','heya','hola','hi']

    user_greeting=['hello','hey, how you doing','heya','hola','hi','hey','oii']

    for word in text.split():
        if word in user_greeting:
            return random.choice(bot_greeting)

def index_sort(list_var):
    length=len(list_var)
    list_index=list(range(0,length))

    x=list_var
    for i in range (length):
        for j in range (length):
            if x[list_index[i]]>x[list_index[j]]:
                temp=list_index[i]
                list_index[i]=list_index[j]
                list_index[j]=temp
    return list_index

def bot_response(user_input):
    user_input=user_input.lower()
    sentence_list.append(user_input)
    bot_response=''
    cm=CountVectorizer().fit_transform(sentence_list)
    similarity_scores=cosine_similarity(cm[-1],cm)
    similarity_scores_list=similarity_scores.flatten()
    index=index_sort(similarity_scores_list)
    index=index[1:]
    response_flag=0

    j=0
    for i in range(len(index)):
        if similarity_scores_list[index[i]]>0.0:
            bot_response=bot_response+''+sentence_list[index[i]]
            response_flag=1
            j=j+1
        if j>2:
            break

    if response_flag==0:
        bot_response=bot_response+''+"Sorry,I don't understand"

    sentence_list.remove(user_input)
    return bot_response

flaskapp = Flask(__name__)

@flaskapp.route('/index')
def index():
    return render_template("index.html")

@flaskapp.route('/diabetes')
def diabetes():
    return render_template('diabetes.html')


@flaskapp.route('/diabetespredectic')
def result(request):
    diabetes_data = pd.read_csv(r"C:\Users\Neel\PycharmProjects\Bank\diabetes (1).csv")

    x = diabetes_data.drop("Outcome", axis=1)
    y = diabetes_data["Outcome"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    model = LogisticRegression()
    model.fit(x_train, y_train)

    val1 = float(request.GET['n1'])
    val2 = float(request.GET['n2'])
    val3 = float(request.GET['n3'])
    val4 = float(request.GET['n4'])
    val5 = float(request.GET['n5'])
    val6 = float(request.GET['n6'])
    val7 = float(request.GET['n7'])
    val8 = float(request.GET['n8'])

    pred = model.predict([[val1, val2, val3, val4, val5, val6, val7, val8]])

    result1 = ""
    if pred == [1]:
        result1 = "Positive"
    else:
        result1 = "Negative"

    return render_template('diabetes.html', {"result2": result1})


@flaskapp.route('/depression')
def Depression():
    return render_template('depression.html')

@flaskapp.route('/depressionpredict')
def depressionpredict():
    return render_template('depressionpredict.html')


@flaskapp.route('/heart')
def heart():
    return render_template("heart.html")

@flaskapp.route('/heartpredict')
def outcome(request):
    heart_data = pd.read_csv(r"C:\Users\Neel\PycharmProjects\Bank\heart (1).csv")

    X = heart_data.drop(columns='target', axis=1)
    Y = heart_data['target']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

    model = LogisticRegression()
    model.fit(X_train, Y_train)

    value1 = float(request.GET['x1'])
    value2 = float(request.GET['x2'])
    value3 = float(request.GET['x3'])
    value4 = float(request.GET['x4'])
    value5 = float(request.GET['x5'])
    value6 = float(request.GET['x6'])
    value7 = float(request.GET['x7'])
    value8 = float(request.GET['x8'])
    value9 = float(request.GET['x9'])
    value10 = float(request.GET['x10'])
    value11 = float(request.GET['x11'])
    value12 = float(request.GET['x12'])
    value13 = float(request.GET['x13'])

    prediction = model.predict(
        [[value1, value2, value3, value4, value5, value6, value7, value8, value9, value10, value11, value12, value13]])

    outcome1 = ""
    if prediction == [1]:
        outcome1 = "You Have Heart Diseases"
    else:
        outcome1 = "You Don't Have Heart Diseases"


    return render_template('heartpredict.html', {"outcome2": outcome1})

@flaskapp.route('/')
def home():
    return render_template('home.html')


@flaskapp.route('/get', methods=['GET','POST'])
def chat():
    userinput = request.args.get('msg')
    exit_list = ['bye', 'thanks', 'thank you', 'good bye', 'see you later', 'quit']
    while (True):
        if userinput.lower() in exit_list:
            return "I will miss talking to you!! Bye Bye :("
        else:
            ans=''
            if greeting(userinput) != None:
                ans+=greeting(userinput)
            else:
                ans+=bot_response(userinput)
            return ans


if __name__ == "__main__":
    flaskapp.run(debug=True)

