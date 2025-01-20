import os
from itertools import chain
import xml.etree.ElementTree as ET
import pandas as pd

import xlsxwriter
from openpyxl import Workbook

from bs4 import BeautifulSoup
import re

def remove_html_tags(text):
    soup = BeautifulSoup(text, 'html.parser')
    return soup.get_text()

def remove_html_tags_regx(text):
    # LaTeX commands with braces r'\\[a-zA-Z]+\{.*?\}|' \
    # LaTeX commands without braces r'\\[a-zA-Z]+|' \
    # Anything in curly braces r'\{.*?\}|' \
    # LaTeX parentheses/brackets r'\\[\(\)\[\]]|' \
    # HTML entities for < and > r'&[lg]t;|' \
    # Equation tags r'\\tag\{.*?\}'
    pattern = r'<[^>]+>|' \
              r'\$\$.*?\$\$|' \
              r'\$.*?\$|' \
              r'\\left\|.*?\\right&gt;|' \
              r'\\[a-zA-Z]+\{.*?\}|' \
              r'\\[a-zA-Z]+|' \
              r'\{.*?\}|' \
              r'\\[\(\)\[\]]|' \
              r'&[lg]t;|' \
              r'\\tag\{.*?\}'

    return re.sub(pattern, '', text, flags=re.DOTALL)

def get_qa_pairs_best_answers(xml_path):
    current_questions = {}
    early_answers = {}  # Store answers that appear before their questions

    for event, elem in ET.iterparse(xml_path):
        if elem.tag == 'row':
            post_type = elem.get('PostTypeId')

            if post_type == '1':  # Question
                post_id = elem.get('Id')
                current_questions[post_id] = {
                    'ViewCount': elem.get('ViewCount'),
                    'Title': elem.get('Title'),
                    'Body': elem.get('Body'),
                    'BestAnswer': None,
                    'BestScore': int(-100)
                }

                # Check if we have early answers for this question
                if post_id in early_answers:
                    answer, score = early_answers[post_id]
                    current_questions[post_id]['BestAnswer'] = answer
                    current_questions[post_id]['BestScore'] = score
                    del early_answers[post_id]  # Clean up

            elif post_type == '2':  # Answer
                parent_id = elem.get('ParentId')
                score = int(elem.get('Score', 0))

                if parent_id in current_questions:
                    if score > current_questions[parent_id]['BestScore']:
                        current_questions[parent_id]['BestAnswer'] = elem.get('Body')
                        current_questions[parent_id]['BestScore'] = score
                else:
                    # Store answer if question hasn't appeared yet
                    if parent_id not in early_answers or score > early_answers[parent_id][1]:
                        early_answers[parent_id] = (elem.get('Body'), score)

            elem.clear()

    # Yield all complete Q&A pairs
    for q_id, q_data in current_questions.items():
        # only if there is an answer
        vcnt = 0 if q_data['ViewCount'] is None else int(q_data['ViewCount'])
        if q_data['BestScore'] != -100 and vcnt >= 1000:
            yield {
                #'QuestionID': q_id,
                #'ViewCount': q_data['ViewCount'],
                'Title': q_data['Title'],
                #'Question': remove_html_tags_regx(str(q_data['Body'])),
                'Answer': remove_html_tags_regx(str(q_data['BestAnswer']))#,
                #'AnswerScore': q_data['BestScore']
            }

def qa_to_dataframe(generator_func, filename):

    ## option 1
    # data = []
    # for item in questions_dict.items():
    #     data.append(item)
    # df = pd.DataFrame(data)

    ## option 2
    # df = pd.DataFrame(generator_func)

    ## option 3
    df = pd.DataFrame(chain.from_iterable([generator_func]))

    df.to_excel(filename, index=False)

    return df


def useOpenpyxl(generator_func, filename):
    wb = Workbook()
    ws = wb.active

    # Write headers (assuming first dict has all columns)
    first_row = next(generator_func)
    headers = list(first_row.keys())
    ws.append(headers)

    # Write first row
    ws.append(list(first_row.values()))

    # Write remaining rows
    for row_dict in generator_func:
        ws.append(list(row_dict.values()))

    wb.save(filename)

def useXlsxwriter(generator_func, filename):
    workbook = xlsxwriter.Workbook(filename)
    worksheet = workbook.add_worksheet()

    # Write headers
    first_row = next(generator_func)
    headers = list(first_row.keys())
    for col, header in enumerate(headers):
        worksheet.write(0, col, header)

    # Write first row
    for col, value in enumerate(first_row.values()):
        worksheet.write(1, col, value)

    # Write remaining rows
    for row_num, row_dict in enumerate(generator_func, start=2):
        for col, value in enumerate(row_dict.values()):
            worksheet.write(row_num, col, value)

    workbook.close()

def writeExcel(xmlfile, xlsxfile):

    generator_func = get_qa_pairs_best_answers(xmlfile)

    ## option 1
    # df = qa_to_dataframe(generator_func, xlsxfile)

    ## Option 2
    # useOpenpyxl(generator_func, xlsxfile)

    ## option 3
    useXlsxwriter(generator_func, xlsxfile)


if __name__ == "__main__":

    # Get the raw file
    path = r'/home/nachiketa/Documents/Workspaces/data/stackexchange/quantumcomputing'
    xmlfile = os.path.join(path,'Posts.xml')

    # Write the excel file
    xpath = r'/home/nachiketa/Documents/Workspaces/pytorch/smol/data'
    xlsxfile = os.path.join(xpath, 'qc.xlsx')
    writeExcel(xmlfile, xlsxfile)




