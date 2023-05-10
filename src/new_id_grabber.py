import csv
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# TO DO: Add arguments for each command, as well as name.
def printAvailableCommands():
    print('--r   Create a list of subjects from a csv.')
    print('--e   Create a csv file containing all of the subjects from a \n        subject list. This list can be created with --e')
    print('--h   Generate histograms from available data.')
    print()


def main():

    action = ''
    
    while action != 'quit':# or action != 'exit':
    
        action = input('What would you like to do? ')
        print()
    
        if 'quit' in action:
            break
    
        # Checks to see if the user want to use a command
        if '--' in action:
    
            available_actions = ['r', 'e','h']
    
            # --e [edit]:
            if 'e' in action:
                matchToList()
            # --r [read]:
            elif 'r' in action:
                createSubjectList()
            # --h [visualize]:
            elif 'h' in action:
                generateHistogram()
    
            if action[2] not in available_actions:
                print("Unknown command. Try: ")
                printAvailableCommands()
        else:
            print('No actions selected')
        

def createSubjectList():

    input_dir = 'csv_input/'
    
    for file_name in os.listdir(input_dir):
        #print(file_name)

        if '.csv' in file_name:
            
            subjects = []
            
            print(file_name)
            
            df = pd.read_csv(input_dir + file_name, encoding='utf-8')
            
            for i in range(0, len(df['CourseID'])):
                course = df['CourseID'][i].replace(u'\xa0', ' ')
                if ' ' in course:
                    space = course.index(' ')
                    cat = course[0:space]
                    
                    if cat not in subjects:
                        subjects.append(cat)
            print(subjects)
            
            
            # Create a txt file with the list of subjects
            base_file_name = file_name[0:file_name.index('.')]
            #print(base_file_name)
    
            try:
                output_file = 'csl_data/' + base_file_name + '_course_subjects.txt'
                #print(output_file)
        
                txtfile = open(output_file, "w", encoding="utf8")
        
                for s in subjects:
                    txtfile.write(s)
                    txtfile.write(' \n')
        
                txtfile.close
            except:
                print('Could not create subject list for', file_name)
            
def matchToList():
    return
            
def generateHistogram():
    unique_subjects = []
    num_schools = []
     
    all_subjects = []
    
    input_dir = 'csv_input/'
    
    for file_name in os.listdir(input_dir):
        #print(file_name)

        if '.csv' in file_name:
            
            subjects = []
            
            #print(file_name)
            
            df = pd.read_csv(input_dir + file_name, encoding='utf-8')
            
            for i in range(0, len(df['CourseID'])):
                course = df['CourseID'][i].replace(u'\xa0', ' ')
                if ' ' in course:
                    space = course.index(' ')
                    cat = course[0:space]       
                    if cat not in subjects:
                        subjects.append(cat)
            #print(subjects)
            
            for subject in subjects:
                if subject not in unique_subjects:
                    unique_subjects.append(subject)
                    num_schools.append(1)
                else:
                    index = unique_subjects.index(subject)
                    num_schools[index] += 1
                  
                all_subjects.append(subject)
                
    #print()
    unique_subjects, num_schools = bubbleSort(unique_subjects, num_schools)
    
    """
    for i in range(0, len(unique_subjects)):
        output = unique_subjects[i] + ': ' + str(num_schools[i])
        print(output)
    """
    # Generate a numpy array from a sorted all_subjects
    all_subjects = sorted(all_subjects)
    subs = np.array(all_subjects)
    
    # Plot histogram of all course subject ids
    #plot = plt.hist(subs)#, figsize=(40,20))
    #fig1 = plt.figsize
    #plt.figure(figsize=(40,200))
    
     #Show histogram
    #plt.show()
    print(len(all_subjects))
    
    plt.figure(figsize=(600,5))
    plt.hist(subs)
    
    plt.show()
    
    #print(unique_subjects)
    #print
    return subs


def bubbleSort(arr, linked_arr):
    n = len(arr)
    # Traverse through all array elements
    for i in range(n):
        # Last i elements are already in place
        for j in range(0, n-i-1):
            # Traverse the array from 0 to n-i-1
            # Swap if the element found is greater than the next element
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
                linked_arr[j], linked_arr[j+1] = linked_arr[j+1], linked_arr[j]
    return arr, linked_arr

            
main()

