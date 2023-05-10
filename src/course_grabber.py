import pandas as pd

def parseComparisonIndex():
    df = pd.read_csv('comparison_index.csv', encoding='utf-8')
    
    #print(df.info())
    #print()
    
    schools = []
    courses = []
    accredited = []
    pdf = []
    xml = []
    found = []
    
    for i in range(0, len(df['School'])):
        # If new school, start collecting entries
        if df['School'][i] not in schools:
            schools.append(df['School'][i])
            courses.append([])
            accredited.append(df['Accredited'][i])
            pdf.append(df['PDF'][i])
            xml.append(df['XML'][i])
            found.append(df['Found'][i])
        # Else append to previous entry
        temp_courses = df['Courses'][i].split(', ')
        for course in temp_courses:
            course.replace('-',' ')
        courses[len(courses)-1].extend(temp_courses)
        
    count = 0
    available_data_count = 0
    available_school_count = 0
    courses_to_str = []
    for i in range(0, len(schools)):
        temp = ''
        for j in range(0, len(courses[i])):
            if j != len(courses[i])-1:
                temp += (courses[i][j] + ', ')
            else:
                temp += courses[i][j]
        courses_to_str.append(temp)
        count += len(courses[i])
        print(schools[i], courses[i], accredited[i], pdf[i], xml[i], found[i])
        if(pdf[i]=='1' or xml[i]=='1' or found[i].lower()=='y'):
            available_data_count += len(courses[i])
            available_school_count += 1
    
    data = {
        'School': schools,
        'Courses': courses_to_str,
        'Accredited': accredited,
        'PDF': pdf,
        'XML': xml,
        'Found': found
    }
    
    new_df = pd.DataFrame(data)
    
    #print(new_df.info())
    #print()
    
    new_df.to_csv('parsed_comparison_index.csv', index=False, header=True, encoding='utf-8')
    
    print('Indexed', str(count), 'courses across', str(len(schools)), 'schools.')
    print('We have', str(available_data_count), 'courses to work with across', available_school_count, 'schools.')


def createMiscCSV():
    df = pd.read_csv('parsed_comparison_index.csv', encoding='utf-8')
    
    schools = []
    course_ids = []
    descriptions = []
    
    for i in range(0, len(df['Accredited'])):
        if df['PDF'][i]==0 and df['XML'][i]==0 and df['Found'][i].lower()=='x':
            temp_courses = df['Courses'][i].split(', ')
            for course in temp_courses:
                course.replace('-',' ')
                schools.append(df['School'][i])
                course_ids.append(course)
                descriptions.append('')
        
    data = {
        'School': schools,
        'CourseID': course_ids,
        'Descriptions': descriptions
    }
    
    new_df = pd.DataFrame(data)

    #print(new_df.info())
    #print()
    
    new_df.to_csv('misc.csv', index=True, header=True, encoding='utf-8')
    
    school_count = 0
    prev_school = ''
    for school in schools:
        if school != prev_school:
            school_count += 1
        prev_school = school
        
    print('Indexed', str(len(schools)), 'courses across', str(school_count), 'schools.')

    
def main():
    parseComparisonIndex()
    createMiscCSV()
  
main()
    


