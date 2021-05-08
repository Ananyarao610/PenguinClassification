#!pip install gradio 
import gradio as gr
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import random
from sklearn.linear_model import LogisticRegression




penguins=pd.read_csv("penguins.csv")
print(penguins)
x=penguins.iloc[:,1:].values
print(x)
y=penguins.iloc[:,0].values
print(y)


c=0
for i in x[:,-1]:
    if i=='MALE':
        x[:,-1][c]=1
    elif i=='FEMALE':
        x[:,-1][c]=0
    else:
        x[:,-1][c]=random.randint(0, 1)
    c+=1

#Converting all strings into numeric data
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x= np.array(ct.fit_transform(x))
le = LabelEncoder()
y = le.fit_transform(y)
print(x)
print(y)


#Conveting to test and train and applying the model
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)


imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
x_train[:,3:-1] = imputer.fit_transform(x_train[:,3:-1])
x_test[:,3:-1]= imputer.transform(x_test[:,3:-1])
print(x_test)
print(y_test)


log_model=LogisticRegression(penalty='l2')
log_model.fit(x_train,y_train)
print(log_model.score(x_test,y_test))
result=log_model.predict(x_test)
print(result)
print(y_test)


def fn(island,sex,cul_len,cul_dep,fli_len,mass):
    if island=="Torgersen Islands":
        island_1=[0.0,0.0,1.0]
    elif island=="Biscoe Islans":
        island_1=[1.0,0.0,0.0]
    else:
        island_1=[0.0,1.0,0.0]
    
    if sex=="Female":
        sex_1=1
    else:
        sex_1=0
        
    island_1.append(cul_len)
    island_1.append(cul_dep)
    island_1.append(fli_len)
    island_1.append(mass)
    island_1.append(sex_1)
    
    
    input_array=np.array([island_1])
    
    
    #log_model=LogisticRegression(penalty='l2')
    #print(x_train)
    #print(y_train)
    #log_model.fit(x_train,y_train)
    pred=log_model.predict(input_array)
    print(input_array)
    output=species(pred[0])
    print(pred[0])
    return output
    #print(island,sex,mass,fli_len,cul_len,cul_dep)
    
def species(ans):
    print(ans)
    if ans==0:
        print("A")
        k=r'C:\Users\Ananya Rao\ML\adelie.jpg'
        return "Adelie Penguin",k
    elif ans==1:
        print("c")
        k=r'C:\Users\Ananya Rao\ML\emperor.jpg'
        return "Emperor Penguin",k
    else:
        print("g")
        k=r'C:\Users\Ananya Rao\ML\gentoo.jpg'
        return "Gentoo Penguin",k


island=gr.inputs.Dropdown(["Torgersen Islands","Biscoe Islans","Dream Islands"],type="value",label="Island of habitat")
cul_len=gr.inputs.Slider(minimum=30,maximum=60,label="Culmen length")
cul_dep=gr.inputs.Slider(minimum=10,maximum=25,label="Culmen depth")
fli_len=gr.inputs.Slider(minimum=150,maximum=250,label="Flipper length")
mass=gr.inputs.Slider(minimum=2500,maximum=6500,label="Body mass")
sex=gr.inputs.Radio(["Female","Male"],type="value",label="Sex")
img=gr.outputs.Image(type="auto",label=" ")
t=gr.outputs.Textbox(type="auto",label="SPECIES")   

gr.Interface(fn, inputs=[island,sex,cul_len,cul_dep,fli_len,mass], outputs=[t,img],title="PENGUIN SPECIES IN ANTARCTIA",live=False).launch(share=True)






#print(log_model.predict([[0.0,1.0,0.0,0,50.7,19.7,203,4050]]))


