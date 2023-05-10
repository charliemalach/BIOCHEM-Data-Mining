import csv
import os
import pandas as pd
import numpy as np

# TO DO: Add arguments for each command, as well as name.
def printAvailableCommands():
    print('--r   Create a list of subjects from a csv.')
    print('--e   Create a csv file containing all of the subjects from a \n        subject list. This list can be created with --e')
    print('--h   Generate histograms from available data.')
    print()


def main():
    action = input('What would you like to do? ')
    print()

    # Checks to see if the user want to use a command
    if '--' in action:
        
        available_actions = ['r', 'e','h']

        if action[2] not in available_actions:
            print("Unknown command. Try: ")
            printAvailableCommands()
            return
        # --e [edit]:
        if 'e' in action:
            matchToList()
        # --r [read]:
        elif 'r' in action:
            createSubjectList()
        # --h [visualize]:
        elif 'h' in action:
            generateHistogram()
    else:
        print('No actions selected')


def createSubjectList():
    
    for file_name in os.listdir('csv_input/'):
        #print(file_name)
        
        course_titles = []
    
        base_file_name = file_name[0:file_name.index('.')]

        try:
            output_file = 'csl_data/' + base_file_name + '_course_subjects.txt'
            print(output_file)
    
            file_name = 'csv_input/' + file_name
            with open(file_name) as csvfile:
                #print(file_name)
                readCSV = csv.reader(csvfile, delimiter=',')
                index = 0
                for row in readCSV:
                    if(index > 0):
                        space = row[2].index(' ')
                        title = row[2][0:space]
                        if not title in course_titles:
                            course_titles.append(title)
                    index += 1
            print(course_titles)
    
            txtfile = open(output_file, "w", encoding="utf8")
    
            for ct in course_titles:
                txtfile.write(ct)
                txtfile.write(' \n')
    
            txtfile.close
        except:
            print('Could not create subject list')


def matchToList():
    
    for file_name in os.listdir('csv_input/'):
        print(file_name)
    
        course_titles = []
    
        base_file_name = file_name[0:file_name.index('.')]
    
        raw_file = 'csv_input/' + file_name
    
        input_file = 'csl_data/' + base_file_name + '_course_subjects.txt'
    
        txtfile = open(input_file, "r", encoding="utf8")
        input_list = txtfile.readlines()
        for i in input_list:
            i = i[0:i.index(' ')]
            course_titles.append(i)
        print(course_titles)
        txtfile.close
    
        output_file = 'csv_output/trimmed_' + file_name
    
        with open(raw_file) as csvfile:
            with open(output_file, 'w', newline='') as outfile:
                readCSV = csv.reader(csvfile, delimiter=',')
                writer = csv.writer(outfile, delimiter=',' ,quotechar='"')
                index = 0
                for row in readCSV:
                    if(index > 0):
                        space = row[2].index(' ')
                        title = row[2][0:space]
                        if title in course_titles:
                            writer.writerow([row[0], row[1], row[2], row[3]])
                    else:
                        writer.writerow([row[0], row[1], row[2], row[3]])
                    index += 1
    
        print('Finished writing out ' + output_file)


def generateHistogram():
    
    # List of unique course title generated from course titles
    unique_course_titles = []
    # Int list of number of schools with matching course titles
    num_schools = []
    
    for file_name in os.listdir('csv_output/'):
        print(file_name)
    
        try:
            course_titles = []
            
            file_name = 'csv_output/' + file_name
            with open(file_name) as csvfile:
                #print(file_name)
                readCSV = csv.reader(csvfile, delimiter=',')
                index = 0
                
                for row in readCSV:
                    if(index > 0):
                        space = row[2].index(' ')
                        title = row[2][0:space]
                        course_titles.append(title)
                    index += 1
            #print(course_titles)
            
            subjects = []
            num_courses = []
            count = 0
            prev = ''
            
            for course in course_titles:
                if course == prev:
                    count += 1
                else:
                    if(prev != ''):
                        subjects.append(prev)
                        num_courses.append(count)
                    prev = course
                    count = 1
            # Account for the last group of courses
            subjects.append(prev)
            num_courses.append(count)
            
            print(subjects)
            print(num_courses)
    
            #for i in range(0, len(subjects)):
            #    if subjects[i] not in unique_course_titles:
            for subject in subjects:
                if subject not in unique_course_titles:
                    unique_course_titles.append(subject)
                    num_schools.append(1)
                else:
                    index = unique_course_titles.index(subject)
                    num_schools[index] += 1
                    
        except:
            print('Could not read file', file_name)
        """
        course_titles = []
        for course in course_titles:
            if course not in unique_course_titles:
                unique_course_titles.append(course)
                num_schools.append(1)
        """
        
        print
        
        # TO-DO: Iterate over course titles and add in unique_course_titles

main()