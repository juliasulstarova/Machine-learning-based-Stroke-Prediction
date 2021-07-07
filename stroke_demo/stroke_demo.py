#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import tkinter as tk
from PIL import ImageTk,Image 
from pickle import load
from sklearn.preprocessing import StandardScaler

def run_demo():
    # tkinter GUI
    root= tk.Tk()


    canvas1 = tk.Canvas(root, width = 700, height = 700, bg='floral white')
    canvas1.pack()

    frame1 = tk.Frame(master=root, width=700, height=350)
    frame1.pack



    title = tk.Label(root, text='Welcome to our Stroke Prevention Booth, please fill in the information below:',bg='bisque')
    canvas1.create_window(340, 20, window=title, )

    # age
    label1 = tk.Label(root, text='Age :')
    canvas1.create_window(140, 100, window=label1)

    entry1 = tk.Entry (root)
    canvas1.create_window(350, 100, window=entry1)

     #gender
    OPTIONS = [
    "Female",
    "male",
    ]

    variable = tk.StringVar(root)
    variable.set(OPTIONS[0]) # default value

    label2 = tk.Label(root, text='Gender :     ')
    canvas1.create_window(140, 140, window=label2)



    entry2 = tk.OptionMenu(root, variable, *OPTIONS)#tk.Entry (root)
    canvas1.create_window(350, 140, window=entry2)




    # Hypertension
    OPTIONS_t = [
    "Yes",
    "No",
    ]

    variable_t = tk.StringVar(root)
    variable_t.set(OPTIONS_t[1]) # default value

    label3 = tk.Label(root, text='History of Hypertension ?:')
    canvas1.create_window(140, 180, window=label3)

    entry3 = tk.OptionMenu(root, variable_t, *OPTIONS_t)
    canvas1.create_window(350, 180, window=entry3)


    # heart input

    OPTIONS_he = [
    "Yes",
    "No",
    ]

    variable_he = tk.StringVar(root)
    variable_he.set(OPTIONS_he[1])

    label4 = tk.Label(root, text='History of heart Disease?:')
    canvas1.create_window(140, 220, window=label4)

    entry4 = tk.OptionMenu(root, variable_he, *OPTIONS_he)
    canvas1.create_window(350, 220, window=entry4)

    # marriage input

    OPTIONS_m = [
    "Yes",
    "No",
    ]

    variable_m = tk.StringVar(root)
    variable_m.set(OPTIONS_m[1])

    label5 = tk.Label(root, text='Have you been married before?:')
    canvas1.create_window(140, 260, window=label5)

    entry5 = tk.OptionMenu(root, variable_m, *OPTIONS_m)
    canvas1.create_window(350, 260, window=entry5)



    # Age input

    OPTIONS_d = [
    "Yes",
    "No",
    ]

    variable_d = tk.StringVar(root)
    variable_d.set(OPTIONS_d[1])

    label6 = tk.Label(root, text='Do you have diabetes?')
    canvas1.create_window(140, 300, window=label6)

    entry6 = tk.OptionMenu(root, variable_d, *OPTIONS_d)
    canvas1.create_window(350, 300, window=entry6)

    # Age input
    label7 = tk.Label(root, text='what is your height? in cm')
    canvas1.create_window(140, 340, window=label7)

    entry7 = tk.Entry (root)
    canvas1.create_window(350, 340, window=entry7)

    # Age input
    label8 = tk.Label(root, text='What is your mass? in kg')
    canvas1.create_window(140, 380, window=label8)

    entry8 = tk.Entry (root)
    canvas1.create_window(350, 380, window=entry8)


    # Age input
    OPTIONS0 = [
    "Never Smoked",
    "Formerly Smoked",
    "Currently Smoke",
    ] #etc


    variable0 = tk.StringVar(root)
    variable0.set(OPTIONS0[0]) # default value


    label9 = tk.Label(root, text='Smoking Status:')
    canvas1.create_window(140, 420, window=label9)

    entry9 = tk.OptionMenu(root, variable0, *OPTIONS0)#tk.Entry (root)
    canvas1.create_window(350, 420, window=entry9)


    # Age input

     #etc
    OPTIONS1 = [
    "Self-employed",
    "Governement job",
    "Private",
    "Never worked before",
    "not an adult yet"
    ]

    variable1 = tk.StringVar(root)
    variable1.set(OPTIONS1[0]) # default value


    label10 = tk.Label(root, text='work type:')
    canvas1.create_window(140, 460, window=label10)

    entry10 = tk.OptionMenu(root, variable1, *OPTIONS1)#tk.Entry (root)
    canvas1.create_window(350, 460, window=entry10)

    # Age input

     #etc
    OPTIONS2 = [
    "Urban",
    "Rural"
    ]

    variable2 = tk.StringVar(root)
    variable2.set(OPTIONS2[0]) # default value


    label11 = tk.Label(root, text='Residence type:')
    canvas1.create_window(140, 500, window=label11)

    entry11 = tk.OptionMenu(root, variable2, *OPTIONS2)#tk.Entry (root)
    canvas1.create_window(350, 500, window=entry11)




    def values(): 
        global age
        age = float(entry1.get()) 

        global gender
        gender = str(variable.get()) 

        global tension
        tension = str(variable_t.get()) 

        global heart
        heart = str(variable_he.get()) 

        global marriage
        marriage = str(variable_m.get()) 

        global diabetes
        diabetes = str(variable_d.get()) 

        global height
        height = float(entry7.get()) 

        global mass
        mass = float(entry8.get()) 

        global smoke
        smoke = str(variable0.get()) 

        global work
        work = str(variable1.get()) 

        global residence
        residence = str(variable2.get()) 

        global BMI
        BMI = mass / (height/100)**2


        work_f=[0,0,0,0,0]
        if work=='Self-employed':
            work_f=[0,0,0,1,0]
        elif work=='Governement job':
            work_f=[1,0,0,0,0]
        elif work=='Never worked before':
            work_f=[0,1,0,0,0]
        elif work=='Private':
            work_f=[0,0,1,0,0]
        else:
            work_f=[0,0,0,1,0]

        smoke_f=[0,0,0]
        if smoke=="Never Smoked":
            smoke_f=[0,1,0]
        elif smoke=='Formerly Smoked':
            smoke_f=[1,0,0]
        elif smoke=='Currently Smoke':
            smoke_f=[0,0,1]


        res_f=0
        if residence=="Urban":
            res_f=1


        gen_f=0
        if gender=="Male":
            gen_f=1

        gluc=130
        if diabetes=='Yes':
            gluc=200

        tension_f=0
        if tension=="Yes":
            tension_f=1

        heart_f=0
        if heart=="Yes":
            heart_f=1


        marriage_f=0
        if marriage=="Yes":
            marriage_f=1

        di= {'gender':[gen_f], 'age':[age], 'hypertension':[tension_f], 'heart_disease':[heart_f], 'ever_married':[marriage_f],
           'Residence_type':[res_f], 'avg_glucose_level':[gluc], 'bmi':[BMI], 'smoking_status_0.0':[smoke_f[0]],
           'smoking_status_1.0':[smoke_f[1]], 'smoking_status_2.0':[smoke_f[2]], 'work_type_Govt_job':[work_f[0]],
           'work_type_Never_worked':[work_f[1]], 'work_type_Private':[work_f[2]],
           'work_type_Self-employed':[work_f[3]], 'work_type_children':[work_f[4]]}
        global do
        do = pd.DataFrame(di)
        st=StandardScaler()
        #st.fit(df_test)
        st=load(open('scaler.pkl', 'rb'))
        test_demo=st.transform(do)
        model=load(open('rf_best_stroke.pkl', 'rb'))
        prob=model.predict_proba(test_demo)[0][1]



        prevention=''
        measure=''
        text_doc=''
        color='blue'
        if prob<0.5:

            prevention='You are not at risk of having a stroke!'
            color='light green'

        elif prob>=0.5 and prob<0.65:
            prevention= 'You are at moderate risk of having a stroke. To prevent it: '
            color='yellow'

            measure+='- Consider exercising more.'
            if BMI > 24:
                measure+='- loosing weight.'
            if smoke=='Currently Smoke':
                measure+='- Quit Smoking.'

        elif prob>=0.65:
            prevention= 'You are at high risk of having a stroke. To prevent it: '
            color='red'
            text_doc='Meet with your doctor every 3 months'
            measure+='- Consider exercising more.'
            if BMI > 24:
                measure+='- loosing weight.'
            if smoke=='Currently Smoke':
                measure+='- Quit Smoking.'

        l1 = tk.Label(root, text= text_doc, bg=color)
        canvas1.create_window(350, 580, window=l1)
            
        l2 = tk.Label(root, text= prevention, bg=color)
        canvas1.create_window(350, 540, window=l2)

        l3 = tk.Label(root, text= measure, bg=color)
        canvas1.create_window(350, 560, window=l3)

        Prediction_result  = ('  Predicted Probability: ', prob)
        label_Prediction = tk.Label(root, text= Prediction_result, bg=color)
        canvas1.create_window(350, 600, window=label_Prediction)

    button1 = tk.Button (root, text='      Predict      ',command=values, bg='green', fg='black', font=11)
    canvas1.create_window(350, 650, window=button1)
    def deleteall():
        entry1.delete(0, 'end')
        variable.set(OPTIONS[0])
        variable_t.set(OPTIONS_t[1])
        variable_he.set(OPTIONS_he[1])
        variable_d.set(OPTIONS_d[1])
        variable_m.set(OPTIONS_m[1])
        entry7.delete(0, 'end')
        entry8.delete(0, 'end')
        variable0.set(OPTIONS0[0])
        variable1.set(OPTIONS1[0])
        variable2.set(OPTIONS2[0])
        l1 = tk.Label(root, text= '                                                                                                          ', bg='floral white')
        canvas1.create_window(350, 580, window=l1)
            
        l2 = tk.Label(root, text= '                                                                                                   ', bg='floral white')
        canvas1.create_window(350, 540, window=l2)

        l3 = tk.Label(root, text= '                                                                                                  ', bg='floral white')
        canvas1.create_window(350, 560, window=l3)
        
        label_Prediction = tk.Label(root, text= '                                                                                                    ', bg='floral white')
        canvas1.create_window(350, 600, window=label_Prediction)
        
        
        values()

    Reset=tk.Button(text="Reset",command=deleteall)
    Reset.pack(pady=25,padx=28)


    root['background']='bisque'


    root.mainloop()



if __name__ == "__main__":
    run_demo()


# In[ ]:




