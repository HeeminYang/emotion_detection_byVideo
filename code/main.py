import pandas as pd
import numpy as np

import argparse

from frame2video import frame2video_eval
from data import processing, crop_data

emotion_dict = {'Angry':'Negative', 'Disgust':'Negative', 'Fear':'Negative', 'Happy':'Positive', 
    'Sad':'Negative', 'Suprise':'Negative', 'Neutral':'Neutral'}

label = processing()

crop_data(label)

n_total = 0
n_correct = 0
e_total = 0
e_correct = 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type = int)
    args = parser.parse_args()

    total_result = []
    for i in range(len(label)):
        file = label['video'][i]
        df = pd.read_csv('./data/crop/'+file+'.csv', index_col=0)
        lbl = label.loc[label['video'] == file]['mode'][i]

        if lbl['Neutral'][i] == 0:
            result = frame2video_eval(file, label, df, args.theshold, emotion_dict) # [file, emotion, result['correct', 'worng']]
            total_result.append(result)
            e_total+=1
            if result[2] == 'correct':
                e_correct+=1

        else:
            result = frame2video_eval(file, label, df, args.theshold, emotion_dict) # [file, emotion, result['correct', 'worng']]
            total_result.append(result)
            n_total+=1
            if result[2] == 'correct':
                n_correct+=1
    
    n_acc = n_correct / n_total
    e_acc = e_correct / e_total
    acc = (n_correct+e_correct) / (n_total+e_total)
    print(f'Accuracy for Neutral data: {n_acc*100:0.2f}%\nAccuracy for Non-Neutral data: {e_acc*100:0.2f}%\nAccuracy for Total data: {acc*100:0.2f}%\n')

    tt = pd.DataFrame(total_result, columns=['file, emotion, result'])
    tt.to_excel('./result/result.xlsx')
