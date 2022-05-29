#Importing Required Libraries
import numpy as np
import pandas as pd
import streamlit as st
import docx2txt
from pdfminer.high_level import extract_text
import re
import spacy
from spacy.matcher import Matcher
from pyresparser import ResumeParser
import pickle
from pickle import dump
from pickle import load
from sklearn.tree import  DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer

#Visualization
import matplotlib.pyplot as plt

#NLP Feature Extraction
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
import pickle

#Title
st.title('RESUME PARSER')
st.write('Parse resumes by yourself & check if they are of good format or not??')

#Initializing the Stemming Process
stemmer = PorterStemmer()
lemmetizer = WordNetLemmatizer()

#Setting Up the Stopwords
#nltk.download('omw-1.4')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('maxent_ne_chunker')
#nltk.download('words')
stop_words = stopwords.words('english')

#Checking the Punctuation
punc = string.punctuation

#Adding Our Own Stopwords
more_stop_words = ['\x0c','"','-','_','.']
stop_words.extend(more_stop_words)

# load pre-trained model
nlp = spacy.load('en_core_web_sm')
# initialize matcher with a vocab
matcher = Matcher(nlp.vocab)

#Function to Extract 'Name'

def extract_name(text):
    nlp_text = nlp(text)
    # First name and Last name are always Proper Nouns
    pattern = [{'POS': 'PROPN'}, {'POS': 'PROPN'}]
    matcher.add('NAME', [pattern])
    matches = matcher(nlp_text)
    for match_id, start, end in matches:
        span = nlp_text[start:end]
        return span.text
        
#Function To Extract Mobile Number
def extract_mobile_number(text):
    phone = re.findall(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]', text)
    if len(phone) == 1:
        return phone[0]
    else:
        return "Not Available"

#Function To Extract Email ID
def extract_email(email):
    email = re.findall("[A-Za-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+", email)
    if len(email) == 1:
        return email[0].split()[0].strip(';')
    else:
        return "Not Available"

#To Extract LinkedIn        
def extract_linkedin(text):
    LINKED_REG = re.compile('linkedin\.com/in/\w+[a-z0-9-]+\w+')
    linked_in = re.findall(LINKED_REG , text)
    if len(linked_in) == 1:
        return linked_in[0]
    else:
        return "Not Available"
    
#To Extract Github
def extract_github(text):
    GITHUB_LINK = re.compile('github\.com/\w+')
    github = re.findall(GITHUB_LINK , text)
    if len(github) == 1:
        return github[0]
    else:
        return "Not Available"

skill_terms = []
with open('skills.txt', 'r') as file:
    skill_terms = file.readlines()
    
skill_terms = [term.strip('\n') for term in skill_terms]

def extract_skills(text):
    stop_words = set(nltk.corpus.stopwords.words('english'))
    word_tokens = nltk.tokenize.word_tokenize(text)

    # remove the stop words
    filtered_tokens = [w for w in word_tokens if w not in stop_words]

    # remove the punctuation
    filtered_tokens = [w for w in word_tokens if w.isalpha()]

    # generate bigrams and trigrams (such as artificial intelligence)
    bigrams_trigrams = list(map(' '.join, nltk.everygrams(filtered_tokens, 2, 3)))

    # we create a set to keep the results in.
    found_skills = set()

    # we search for each token in our skills database
    for token in filtered_tokens:
        if token.lower() in skill_terms:
            found_skills.add(token)

    # we search for each bigram and trigram in our skills database
    for ngram in bigrams_trigrams:
        if ngram.lower() in skill_terms:
            found_skills.add(ngram)

    return found_skills


# Education Degrees
EDUCATION = [
            'BE','B.E.', 'B.E', 'BS','Bachelor of Technology','Senior Secondary' 'B.S', 'B.SC', 'B E', 'B. E.','B. E','B S','B. S','B. SC'
            'ME', 'M.E', 'M.E.', 'MS', 'M.S', 'B-TECH','M-TECH','M E', 'M. E','B.COM','B.ED','L.L.B.','LLB','LLM','L.L.M.', 'M. E.', 'M S', 'M. S',
            'BTECH', 'B.TECH', 'M.TECH', 'MTECH','B TECH', 'B. TECH', 'M. TECH', 'M TECH',
            'B. TECH','M. TECH','B TECH','M TECH','M.D.','NDA','N.D.A.','PHD','PGDM','P.G.D.M.' 'MBA','M.B.A.','MCA','M.C.A.','MS','M.S.','MD',
            'SSC', 'HSC', 'CBSE', 'ICSE', 'X', 'XII',
            'BBA','B.B.A.','BCA','B.C.A.','BA','B.A.','BSCIT'
        ]


#education = []
#Function to Extract Education
def extract_education(resume_text):
    nlp_text = nlp(resume_text)
    # Sentence Tokenizer
    nlp_text = [sent.text.strip() for sent in nlp_text.sents]

    edu = {}
    # Extract education degree
    for index, text in enumerate(nlp_text):
        for tex in text.split():
            # Replace all special symbols
            tex = re.sub(r'[?|$|.|!|,]', r'', tex)
            if tex.upper() in EDUCATION and tex not in stop_words:
                edu[tex] = text + nlp_text[index]   

    education = []
    for key in edu.keys():
        education.append(key)

    return education

#Function to Extract Years of Experience
def extract_years_of_experience(text):
    lines = sent_tokenize(text)           
    experience = []
    for sentence in lines:
        if re.search('experience',sentence.lower()):        
            sen_tokenized = nltk.word_tokenize(sentence)
            tagged = nltk.pos_tag(sen_tokenized)            
            entities = nltk.chunk.ne_chunk(tagged)          
            for subtree in entities.subtrees():
                for leaf in subtree.leaves():
                    if leaf[1] == 'CD':                     
                        experience.append(leaf[0])
                        
    exp = []
    for ele in experience:
        if len(ele) <= 3 or (len(ele) <= 4 and ele[-1] == '0' 
                                and ele not in ('2020','2010','2000')):       
            exp.append(ele)
    if exp:
        return exp[0]
    else:
        return "0"

#clean_data() takes sentence separated string input and cleans the data i.e. removing stopwords and punctuation and returning lemmetized string output
def clean_data(text):
    re.sub(r'[\d]','',text)
    re.sub(r'[^a-zA-Z]','',text)
    re.sub('\s+',' ',text)
    text_clean = []
    text_tokens = word_tokenize(text)
    #text_tokens = tokenizer.tokenize(text)    
    for word in text_tokens:
        if (word not in stop_words and # remove stopwords
            word not in string.punctuation): # remove punctuation
            stem_word = lemmetizer.lemmatize(word) # stemming word
            text_clean.append(stem_word)
    
    list_to_str = ' '.join([str(ele) for ele in text_clean])
    return list_to_str.lower()

#call_to_clean() takes full text as input, breaks them into sentences and calls the clean_data() with the separated sentences as arguments
clean_text = []
def call_to_clean(text):
    sentences = re.split(r'\n+',text)
    sentence_df = pd.DataFrame(sentences, columns = ['Text'])
    sentence_df['Text'] = sentence_df['Text'].apply(clean_data)
    clean_text.append(' '.join(sentence_df['Text'])) 

uploaded_file = st.file_uploader("Choose your file :", type=['pdf' , 'docx'], accept_multiple_files=True)

text = []
def extract_text_from_docx(file):
    txt = docx2txt.process(file)
    text = [line.replace('\t', ' ') for line in txt.split('\n') if line]
    return ' '.join(text)

if uploaded_file is not None:
    
    for files in uploaded_file:
        if files.type == 'application/pdf':
            text.append(extract_text(files))
        elif files.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            text.append(extract_text_from_docx(files))
            
    df = pd.DataFrame(text, columns=['Text'])
    #Calling the Function to Get Clean Text
    for i in range(len(df)):
        call_to_clean(df['Text'].loc[i])

    df['Clean Text'] = clean_text

    Name = []
    for i in range(len(df)):
        Name.append(extract_name(df['Clean Text'][i]))

    mobile = []
    for i in range(len(df)):
      mobile.append(extract_mobile_number(df['Clean Text'][i]))

    email = []
    for i in range(len(df)):
      email.append(extract_email(df['Text'][i]))

    linkedin = []
    for i in range(len(df)):
      linkedin.append(extract_linkedin(df['Text'][i]))

    Github = []
    for i in range(len(df)):
      Github.append(extract_github(df['Text'][i]))

    def listToString(s):
        str1 = ", "
        return (str1.join(s))

    Skills = []
    for i in range(len(text)):
        Skills.append(extract_skills(text[i]))

    Education = [] 
    for i in range(len(text)):
        Education.append(extract_education(text[i]))

    Experience = []
    for i in range(len(df)):
      Experience.append(extract_years_of_experience(df['Text'][i]))


    #Creating Features DataFrame For Model
    Features_new = pd.DataFrame(Name,columns = ['Names'])
    Features_new['Skills'] = Skills
    cleaned_skills = []
    #Calling Function to clean Skills
    for skills in Features_new['Skills']:
        cleaned_skills.append(listToString(skills))
    Features_new['Cleaned Skills'] = cleaned_skills
    #Dropping Skills
    Features_new = Features_new.drop(['Skills'],axis = 1)
    #Loading the Features CSV.
    Features = pd.read_csv("Final_features.csv")
    #Dropping Non Sigificant Column
    Features = Features.drop(['Skills','Label'],axis = 1)
    #Appending with New Features
    Final = pd.concat([Features,Features_new],axis = 0,ignore_index=True)
    
    #Taking only Skills 
    x = Final['Cleaned Skills'].values
    #Vectoriation 
    CV = CountVectorizer(max_features=20,stop_words = 'english')
    #Vectorizing x 
    x_cv = CV.fit_transform(x).toarray()
    
    #Loading the Model from disk
    model = load(open('resume.pickle', 'rb'))

    Prediction = model.predict(x_cv)
    #st.write(Prediction[-1])

    #Creating Category list to store Predicted Class
    category = []

    def details(Name,Mobile,Email,LinkedIn,Github,Education,Skills,Experience):
        NAME = st.write('Name:', Name)
        MOBILE = st.write('Mobile Number:', Mobile)
        EMAIL = st.write('Email ID:', Email)
        LINKEDIN = st.write('LinkedIn:', LinkedIn)
        GITHUB = st.write('Github:', Github)
        EDUCATIONS = st.write('Education:', listToString(Education))
        SKILLS = st.write('Skills:', listToString(Skills))
        EXPERIENCE = st.write('Experience:', Experience)
        
        return NAME,MOBILE,EMAIL,LINKEDIN,GITHUB,EDUCATIONS,SKILLS,EXPERIENCE

    try:
        #For One or More Files
        if len(df) >= 1:
            st.subheader("Resume 1: ")
            st.write('Name:', Name[0])
            st.write('Mobile Number:', mobile[0])
            st.write('Email ID:', email[0])
            st.write('LinkedIn:', linkedin[0])
            st.write('Github:', Github[0])
            st.write('Education:', listToString(Education[0]))
            st.write('Skills:', listToString(Skills[0]))
            st.write('Experience:', Experience[0])

            if len(df) >= 2:
                st.subheader("Parse Other Resumes Below...")
                if st.button("Parse Resume 2"):
                    st.subheader("Resume 2:")
                    details(Name[1],mobile[1],email[1],linkedin[1],Github[1],Education[1],Skills[1],Experience[1])

            if len(df) >= 3:
                if st.button("Parse Resume 3"):
                    st.subheader("Resume 3:")
                    details(Name[2],mobile[2],email[2],linkedin[2],Github[2],Education[2],Skills[2],Experience[2])

            if len(df) >= 4:
                if st.button("Parse Resume 4"):
                    st.subheader("Resume 4:")
                    details(Name[3],mobile[3],email[3],linkedin[3],Github[3],Education[3],Skills[3],Experience[3])

            if len(df) >= 5:
                if st.button("Parse Resume 5"):
                    st.subheader("Resume 5:")
                    details(Name[4],mobile[4],email[4],linkedin[4],Github[4],Education[4],Skills[4],Experience[4])

            if len(df) >= 6:
                if st.button("Parse Resume 6"):
                    st.subheader("Resume 6:")
                    details(Name[5],mobile[5],email[5],linkedin[5],Github[5],Education[5],Skills[5],Experience[5])

            if len(df) >= 7:
                if st.button("Parse Resume 7"):
                    st.subheader("Resume 7:")
                    details(Name[6],mobile[6],email[6],linkedin[6],Github[6],Education[6],Skills[6],Experience[6])

            if len(df) >= 8:
                if st.button("Parse Resume 8"):
                    st.subheader("Resume 8:")
                    details(Name[7],mobile[7],email[7],linkedin[7],Github[7],Education[7],Skills[7],Experience[7])

            if len(df) >= 9:
                if st.button("Parse Resume 9"):
                    st.subheader("Resume 9:")
                    details(Name[8],mobile[8],email[8],linkedin[8],Github[8],Education[8],Skills[8],Experience[8])

            if len(df) >= 10:
                if st.button("Parse Resume 10"):
                    st.subheader("Resume 10:")
                    details(Name[9],mobile[9],email[9],linkedin[9],Github[9],Education[9],Skills[9],Experience[9])


            if len(df) >10:
                if st.button("Parse Latest Upload Resume"):
                    st.subheader(" Lastest Resume:")
                    details(Name[-1],mobile[-1],email[-1],linkedin[-1],Github[-1],Education[-1],Skills[-1],Experience[-1])

            
            if len(Final) >= 80:
                st.sidebar.title("Resume Classifier")
                st.sidebar.write("Classify your resume...")
                if Prediction[79] == 0:
                    category.append("Different Resume (DS / DA)")
                    if st.sidebar.button("Classify Resume 1"):
                        st.sidebar.write("Different Resume (DS / DA)")
                elif Prediction[79] == 1:
                    category.append("PeopleSoft")
                    if st.sidebar.button("Classify Resume 1"):
                        st.sidebar.write("PeopleSoft")
                elif Prediction[79] == 2:
                    category.append("React Developer")
                    if st.sidebar.button("Classify Resume 1"):
                         st.sidebar.write("React Developer")
                elif Prediction[79] == 3:
                    category.append("SQL Developer")
                    if st.sidebar.button("Classify Resume 1"):
                        st.sidebar.write("SQL Developer")
                else :
                    category.append("Workday Resume")
                    if st.sidebar.button("Classify Resume 1"):
                        st.sidebar.write("Workday Resume")

            if len(Final) >= 81:
                if Prediction[80] == 0:
                    category.append("Different Resume (DS / DA)")
                    if st.sidebar.button("Classify Resume 2"):
                        st.sidebar.write("Different Resume (DS / DA)")
                elif Prediction[80] == 1:
                    category.append("PeopleSoft")
                    if st.sidebar.button("Classify Resume 2"):
                        st.sidebar.write("PeopleSoft")
                elif Prediction[80] == 2:
                    category.append("React Developer")
                    if st.sidebar.button("Classify Resume 2"):
                        st.sidebar.write("React Developer")
                elif Prediction[80] == 3:
                    category.append("SQL Developer")
                    if st.sidebar.button("Classify Resume 2"):
                        st.sidebar.write("SQL Developer")
                else :
                    category.append("Workday Resume")
                    if st.sidebar.button("Classify Resume 2"):
                        st.sidebar.write("Workday Resume")

            if len(Final) >= 82:
                if Prediction[81] == 0:
                    category.append("Different Resume (DS / DA)")
                    if st.sidebar.button("Classify Resume 3"):
                        st.sidebar.write("Different Resume (DS / DA)")
                elif Prediction[81] == 1:
                    category.append("PeopleSoft")
                    if st.sidebar.button("Classify Resume 3"):
                        st.sidebar.write("PeopleSoft")
                elif Prediction[81] == 2:
                    category.append("React Developer")
                    if st.sidebar.button("Classify Resume 3"):
                        st.sidebar.write("React Developer")
                elif Prediction[81] == 3:
                    category.append("SQL Developer")
                    if st.sidebar.button("Classify Resume 3"):
                        st.sidebar.write("SQL Developer")
                else :
                    category.append("Workday Resume")
                    if st.sidebar.button("Classify Resume 3"):
                        st.sidebar.write("Workday Resume")

            if len(Final) >= 83:
                if Prediction[82] == 0:
                    category.append("Different Resume (DS / DA)")
                    if st.sidebar.button("Classify Resume 4"):
                        st.sidebar.write("Different Resume (DS / DA)")
                elif Prediction[82] == 1:
                    category.append("PeopleSoft")
                    if st.sidebar.button("Classify Resume 4"):
                        st.sidebar.write("PeopleSoft")
                elif Prediction[82] == 2:
                    category.append("React Developer")
                    if st.sidebar.button("Classify Resume 4"):
                        st.sidebar.write("React Developer")
                elif Prediction[82] == 3:
                    category.append("SQL Developer")
                    if st.sidebar.button("Classify Resume 4"):
                        st.sidebar.write("SQL Developer")
                else :
                    category.append("Workday Resume")
                    if st.sidebar.button("Classify Resume 4"):
                        st.sidebar.write("Workday Resume")

            if len(Final) >= 84:
                if Prediction[83] == 0:
                    category.append("Different Resume (DS / DA)")
                    if st.sidebar.button("Classify Resume 5"):
                        st.sidebar.write("Different Resume (DS / DA)")
                elif Prediction[83] == 1:
                    category.append("PeopleSoft")
                    if st.sidebar.button("Classify Resume 5"):
                        st.sidebar.write("PeopleSoft")
                elif Prediction[83] == 2:
                    category.append("React Developer")
                    if st.sidebar.button("Classify Resume 5"):
                        st.sidebar.write("React Developer")
                elif Prediction[83] == 3:
                    category.append("SQL Developer")
                    if st.sidebar.button("Classify Resume 5"):
                        st.sidebar.write("SQL Developer")
                else :
                    category.append("Workday Resume")
                    if st.sidebar.button("Classify Resume 5"):
                        st.sidebar.write("Workday Resume")

            if len(Final) >= 85:
                if Prediction[84] == 0:
                    category.append("Different Resume (DS / DA)")
                    if st.sidebar.button("Classify Resume 6"):
                        st.sidebar.write("Different Resume (DS / DA)")
                elif Prediction[84] == 1:
                    category.append("PeopleSoft")
                    if st.sidebar.button("Classify Resume 6"):
                        st.sidebar.write("PeopleSoft")
                elif Prediction[84] == 2:
                    category.append("React Developer")
                    if st.sidebar.button("Classify Resume 6"):
                        st.sidebar.write("React Developer")
                elif Prediction[84] == 3:
                    category.append("SQL Developer")
                    if st.sidebar.button("Classify Resume 6"):
                        st.sidebar.write("SQL Developer")
                else :
                    category.append("Workday Resume")
                    if st.sidebar.button("Classify Resume 6"):
                        st.sidebar.write("Workday Resume")

            if len(Final) >= 86:
                if Prediction[85] == 0:
                    category.append("Different Resume (DS / DA)")
                    if st.sidebar.button("Classify Resume 7"):
                        st.sidebar.write("Different Resume (DS / DA)")
                elif Prediction[85] == 1:
                    category.append("PeopleSoft")
                    if st.sidebar.button("Classify Resume 7"):
                        st.sidebar.write("PeopleSoft")
                elif Prediction[85] == 2:
                    category.append("React Developer")
                    if st.sidebar.button("Classify Resume 7"):
                        st.sidebar.write("React Developer")
                elif Prediction[85] == 3:
                    category.append("SQL Developer")
                    if st.sidebar.button("Classify Resume 7"):
                        st.sidebar.write("SQL Developer")
                else :
                    category.append("Workday Resume")
                    if st.sidebar.button("Classify Resume 7"):
                        st.sidebar.write("Workday Resume")

            if len(Final) >= 87:
                if Prediction[86] == 0:
                    category.append("Different Resume (DS / DA)")
                    if st.sidebar.button("Classify Resume 8"):
                        st.sidebar.write("Different Resume (DS / DA)")
                elif Prediction[86] == 1:
                    category.append("PeopleSoft")
                    if st.sidebar.button("Classify Resume 8"):
                        st.sidebar.write("PeopleSoft")
                elif Prediction[86] == 2:
                     category.append("React Developer")
                     if st.sidebar.button("Classify Resume 8"):
                        st.sidebar.write("React Developer")
                elif Prediction[86] == 3:
                    category.append("SQL Developer")
                    if st.sidebar.button("Classify Resume 8"):
                        st.sidebar.write("SQL Developer")
                else :
                    category.append("Workday Resume")
                    if st.sidebar.button("Classify Resume 8"):
                        st.sidebar.write("Workday Resume")

            if len(Final) >= 88:
                if Prediction[87] == 0:
                    category.append("Different Resume (DS / DA)")
                    if st.sidebar.button("Classify Resume 9"):
                        st.sidebar.write("Different Resume (DS / DA)")
                elif Prediction[87] == 1:
                    category.append("PeopleSoft")
                    if st.sidebar.button("Classify Resume 9"):
                        st.sidebar.write("PeopleSoft")
                elif Prediction[87] == 2:
                    category.append("React Developer")
                    if st.sidebar.button("Classify Resume 9"):
                        st.sidebar.write("React Developer")
                elif Prediction[87] == 3:
                    category.append("SQL Developer")
                    if st.sidebar.button("Classify Resume 9"):
                        st.sidebar.write("SQL Developer")
                else :
                    category.append("Workday Resume")
                    if st.sidebar.button("Classify Resume 9"):
                        st.sidebar.write("Workday Resume")

            if len(Final) >= 89:
                if Prediction[88] == 0:
                    category.append("Different Resume (DS / DA)")
                    if st.sidebar.button("Classify Resume 10"):
                        st.sidebar.write("Different Resume (DS / DA)")
                elif Prediction[88] == 1:
                    category.append("PeopleSoft")
                    if st.sidebar.button("Classify Resume 10"):
                        st.sidebar.write("PeopleSoft")
                elif Prediction[88] == 2:
                    category.append("React Developer")
                    if st.sidebar.button("Classify Resume 10"):
                        st.sidebar.write("React Developer")
                elif Prediction[88] == 3:
                    category.append("SQL Developer")
                    if st.sidebar.button("Classify Resume 10"):
                        st.sidebar.write("SQL Developer")
                else :
                    category.append("Workday Resume")
                    if st.sidebar.button("Classify Resume 10"):
                        st.sidebar.write("Workday Resume")

            if len(Final) >= 90:
                if Prediction[-1] == 0:
                    category.append("Different Resume (DS / DA)")
                    if st.sidebar.button("Classify Latest Upload Resume"):
                        st.sidebar.write("Different Resume (DS / DA)")
                elif Prediction[-1] == 1:
                    category.append("PeopleSoft")
                    if st.sidebar.button("Classify Latest Upload Resume"):
                        st.sidebar.write("PeopleSoft")
                elif Prediction[-1] == 2:
                    category.append("React Developer")
                    if st.sidebar.button("Classify Latest Upload Resume"):
                        st.sidebar.write("React Developer")
                elif Prediction[-1] == 3:
                    category.append("SQL Developer")
                    if st.sidebar.button("Classify Latest Upload Resume"):
                        st.sidebar.write("SQL Developer")
                else :
                    category.append("Workday Resume")
                    if st.sidebar.button("Classify Latest Upload Resume"):
                        st.sidebar.write("Workday Resume")


        #Creating Features DataFrame Visualization
        viz = pd.DataFrame(Name,columns = ['Names'])
        #Function to Convert Experience to Number.
        num = []
        def tofloat(exp):
            for x in exp:
                num.append(float(x))
            return num
        viz['Experience'] = tofloat(Experience)
        viz['Category'] = category

        #Creating Pie Chart for Category of Resumes
        value = viz['Category'].value_counts()
        label  = viz['Category'].unique()
        if len(viz) >= 2:
            st.sidebar.title("View Analysis")
            if st.sidebar.button("View Pie Chart"):
                st.sidebar.write("Resume Distribution Chart: ")
                fig1, ax1 = plt.subplots()
                ax1.pie(value, labels = label,startangle = 90)
                ax1.axis('equal') 
                st.sidebar.write(fig1)
            if st.sidebar.button("View Bar Chart"):
                st.sidebar.write("Experience Distribution Bar Chart: ")
                #viz1 = viz.drop(['Category'],axis = 1)
                fig1, ax1 = plt.subplots()
                viz_sorted = viz.sort_values('Experience',ascending = False)
                ax1.bar('Names', 'Experience', data = viz_sorted)
                ax1.axis('equal') 
                st.sidebar.write(fig1)

    except IndexError:
        pass

else:
    st.write('Drop or Select your file : ')
    
