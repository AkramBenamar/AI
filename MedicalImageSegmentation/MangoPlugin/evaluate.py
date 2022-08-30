from tkinter import *
import json

import argparse

parser = argparse.ArgumentParser(description='Script so useful.')
parser.add_argument("--folderstat")
parser.add_argument("--disease")


args = parser.parse_args()

stat_path = args.folderstat
disease = args.disease

 
 
class Table:
     
    def __init__(self,root,lst_title1,lst_title2,data_table):
        total_rows = len(data_table)
        total_columns = len(data_table[0]) 
        # code for creating table
        for i in range(total_rows+1):
            for j in range(total_columns+1):
                if(i==0)and(j!=0):

                    self.e = Entry(root, width=10, fg='blue',
                                font=('Arial',11,'bold'))
                    element=lst_title1[j-1]
                elif(j==0)and(i!=0):
                    self.e = Entry(root, width=15, fg='blue',
                                font=('Arial',11,'bold'))
                    element=lst_title2[i-1]
                elif(i==0)and(j==0):
                    self.e = Entry(root, width=15, fg='blue',
                                font=('Arial',11))
                    element=""
                else:
                    self.e = Entry(root, width=10,
                                font=('Arial',11))
                    element=data_table[i-1][j-1]
                 
                self.e.grid(row=i, column=j)
                self.e.insert(END,element )
 
# take the data

with open(stat_path) as f:
   data = json.load(f)

# print("{:,.4f}".format(data['dice'][0]['WT']))



# find total number of rows and
# columns in list
if(disease=="covid-19"):
    lst_title_ct=[['COVID-19'],['Dice','Sensitivity']]
    lst_ct = [["{:,.4f}".format(data["dice"][0])],
       ["{:,.4f}".format(data["sen"][0])],]
    Ltitle=lst_title_ct
    Ldata=lst_ct

elif(disease=="brain-tumor"):
    lst_title_brats=[['WT','TC','ET'],['Dice','Sensitivity','Specificity','Hausdorff95']]
    lst_brats = [["{:,.4f}".format(data['dice'][0]['WT']),"{:,.4f}".format(data['dice'][0]['TC']),"{:,.4f}".format(data['dice'][0]['ET'])],
       ["{:,.4f}".format(data['sen'][0]['WT']),"{:,.4f}".format(data['sen'][0]['TC']),"{:,.4f}".format(data['sen'][0]['ET'])],
       ["{:,.4f}".format(data['spec'][0]['WT']),"{:,.4f}".format(data['spec'][0]['TC']),"{:,.4f}".format(data['spec'][0]['ET'])],
       ["{:,.4f}".format(data['hau95'][0]['WT']),"{:,.4f}".format(data['hau95'][0]['TC']),"{:,.4f}".format(data['hau95'][0]['ET'])]]
    Ltitle=lst_title_brats
    Ldata=lst_brats
  
# create root window
root = Tk()
t = Table(root,Ltitle[0],Ltitle[1],Ldata)
root.mainloop()