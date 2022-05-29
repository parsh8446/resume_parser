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
#import nltkmodules
#NLP Feature Extraction
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize

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


education = []
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

    # Extract year
    if edu:
        for key in edu.keys():
            year = re.search(re.compile(r'(((21|19)(\d{2})))'), edu[key])  
            if year:
                education.append((key, ''.join(year[0])))
            else:
                education.append(key)
        return education
    else:
        EDU_PATTERN = re.findall(r'Bachelors? \D+|Masters? \D+',resume_text)   
        return EDU_PATTERN

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
        return "NA"

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

def pdf_file():
    uploaded_file = st.file_uploader("Upload a file", type=["zip","csv", "json",'pdf' , 'docx'], accept_multiple_files=False)

    ## For PDF
    st.write(uploaded_file)
    text = extract_text(uploaded_file)
    # st.write(text)  
    if text == None:
        st.write('Drop or Select your file : ')
    else :
          return text  

uploaded_file = st.file_uploader("Choose your file :", type=['pdf' , 'docx'], accept_multiple_files=False)

# For Doc file
doc_text = []
pdf_text = []
def extract_text_from_docx(file):
    txt = docx2txt.process(file)
    text = [line.replace('\t', ' ') for line in txt.split('\n') if line]
    return ' '.join(text)

if uploaded_file is not None:
    
    st.text('')
    st.text('')
    # for x in uploaded_file:
    file_type = uploaded_file.name.split('.')

    if file_type[1] == 'pdf':
        text = extract_text(uploaded_file)
        pdf_text.append(extract_text(uploaded_file))
        df = pd.DataFrame(pdf_text, columns=['Text'])
    elif file_type[1] == 'docx':
        text = (extract_text_from_docx(uploaded_file))
        doc_text.append(extract_text_from_docx(uploaded_file))
        df = pd.DataFrame(doc_text, columns=['Text'])

    #df = pd.DataFrame(doc_text, columns=['Text'])
    #df = pd.DataFrame(pdf_text, columns=['Text'])
    #text = text.replace('\n',' ') # remove \n in text
    #doc = nlp_model(text)
    #st.write(type(doc_text))



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

    Skills = extract_skills(text)

    Education = extract_education(text)

    Experience = []
    for i in range(len(df)):
      Experience.append(extract_years_of_experience(df['Text'][i]))


    st.subheader('Name:')
    st.write(Name[0])
    st.subheader('Mobile Number:')
    st.write(mobile[0])
    st.subheader('Email ID:')
    st.write(email[0])
    st.subheader('LinkedIn:')
    st.write(linkedin[0])
    st.subheader('Github:')
    st.write(Github[0])
    st.subheader('Skills:')
    st.write(listToString(Skills))
    st.subheader('Educations:')
    st.write(listToString(Education))
    st.subheader('Experience:')
    st.write(Experience[0])

    #Creating a 'Features' DataFrame
    Features = pd.DataFrame(Name,columns = ['Names'])
    Features['Mobile_Number'] = mobile
    Features['Email ID'] = email
    Features['Linkedin'] = linkedin
    Features['Github'] = Github
    Features['Skills'] = listToString(Skills)
    Features['Education'] = listToString(Education)
    Features['Experience'] = Experience
    
    def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv(index= False).encode('utf-8')

    csv = convert_df(Features)
    st.download_button(label="Download data as CSV", data = csv ,file_name='Parsed_Info.csv', mime='text/csv',)


else:
    st.write('Drop or Select your file : ')
