import pandas as pd

import os
import natsort

def processing():
    df = pd.read_excel('./data/DuckEEsStim/VideoStimSet_SuppTable1.xlsx', engine='openpyxl') # download first
    df = df[df['% correct']>=0.6] # cross validation accuracy (upper 0.6)

    fl = os.listdir('./data/DuckEEs')
    file_list = natsort.natsorted(fl)

    # file drop
    drop_file =[]
    for file in file_list:
        df1 = pd.read_csv('./data/DuckEEs/'+file)[['total_count', 'emotion']]
        if len(df1) == 0:
            drop_file.append(file)
    
    true_label = df[['video', 'mode']]
    true_label['video'] = df['video'].str[6:]
    drop_filename = []
    for file in drop_file:
        drop_filename.append(file.split('.')[0])
    drop_df = pd.DataFrame(drop_filename, columns=['video'])
    drop_df['bad'] = 1

    label1 = pd.merge(true_label, drop_df, how='left', on='video')
    label2 = label1[label1['bad'].isnull()].iloc[:,0:-1]

    label2['Neutral'] = 0
    label2.loc[label2['mode'] == 'neutral', 'Neutral'] = 1
    label2 = label2.reset_index()
    return label2

def crop_data(label):
    for idx in range(len(label)):
        file = label['video'][idx]
        df1 = pd.read_csv('./data/DuckEEs/'+file+'.csv')[['total_count', 'emotion']]
        end_frame = int(df1['total_count'][-1:])
        df2 = pd.merge(pd.DataFrame([i for i in range(1, end_frame+1)], columns=['total_count']), df1, how='outer').fillna('Neutral')
        if label['Neutral'][idx] == 0:
            crop_frame_start = 0

            for i in range(end_frame):
                if df2['emotion'][i] == 'Neutral' or df2['emotion'][i] == 'none':
                    crop_frame_start = i+1
                else:
                    break
            crop_frame_end = end_frame

            reverse_df2 = pd.merge(pd.DataFrame([i for i in range(end_frame, -1, -1)], columns=['total_count']), df1, how='outer').fillna('Neutral')
            for i in range(end_frame):
                if reverse_df2['emotion'][i] == 'Neutral' or reverse_df2['emotion'][i] == 'none':
                    crop_frame_end = reverse_df2['total_count'][i] - 1
                else:
                    break
            if crop_frame_end == 0:
                crop_frame_start = 0
                crop_frame_end = end_frame

            df3 = df2[crop_frame_start:crop_frame_end]
        else:
            df3 = df2
        df3.to_csv('/data/crop/'+file+'.csv')
    print('crop finished')