import pandas as pd

from collections import Counter

duckEE_dict = {'disgust':'Negative', 'fear':'Negative', 'happiness':'Positive',  'embarrassment':'Negative',
    'sadness':'Negative', 'pride':'Positive', 'neutral':'Neutral'}

def frame2video_eval(file, label, df, theshold, emotion_dict):
    l1 = df['total_count'].tolist()
    l2 = df['emotion'].tolist()
    df1 = pd.DataFrame([l1,l2]).T
    df1['index'] = df1[0]
    a = list(range(l1[-1]+1))
    index_df = pd.DataFrame(a, columns=['index'])
    df2 = pd.merge(index_df, df1, how="outer")
    df2 = df2.fillna('none')
    emolist = df2[1].to_list()
    emocount = Counter(emolist)
    top3 = emocount.most_common(n=3)
    l3 = []

    total_len = len(l1)
    p10_len = int(total_len * theshold)

    if top3[0][0] == 'none':
        if len(top3) == 1:
            l3.append('Neutral')
        elif top3[1][0] == 'Neutral':
            if len(top3) == 3:
                if top3[2][1] >= p10_len:
                    l3.append(top3[2][0])
                else:
                    l3.append(top3[1][0])
            else:
                l3.append(top3[1][0])
    elif top3[0][0] == 'Neutral':
        if len(top3) == 1:
            l3.append('Neutral')
        elif top3[1][0] == 'none':
            if len(top3) == 2: 
                l3.append('Neutral')
            else:
                if top3[2][1] >= p10_len:
                    l3.append(top3[2][0])
                else:
                    l3.append(top3[0][0])
        else:
            if top3[1][1] >= p10_len:
                l3.append(top3[1][0])
            else:
                l3.append(top3[0][0])
    else:
        l3.append(top3[0][0])
    for emotion in l3:
        if emotion_dict[emotion] == duckEE_dict[label]:
            total_result = [file, l3[0], 'correct']
        else:
            total_result = [file, l3[0], 'wrong']
        print(file, emotion, label, emotion_dict[emotion], duckEE_dict[label])
    
    return total_result


def frame2video(df, theshold):
    l1 = df['total_count'].tolist()
    l2 = df['emotion'].tolist()
    df1 = pd.DataFrame([l1,l2]).T
    df1['index'] = df1[0]
    a = list(range(l1[-1]+1))
    index_df = pd.DataFrame(a, columns=['index'])
    df2 = pd.merge(index_df, df1, how="outer")
    df2 = df2.fillna('none')
    emolist = df2[1].to_list()
    emocount = Counter(emolist)
    top3 = emocount.most_common(n=3)
    l3 = []

    total_len = len(l1)
    p10_len = int(total_len * theshold)

    if top3[0][0] == 'none':
        if len(top3) == 1:
            l3.append('Neutral')
        elif top3[1][0] == 'Neutral':
            if len(top3) == 3:
                if top3[2][1] >= p10_len:
                    l3.append(top3[2][0])
                else:
                    l3.append(top3[1][0])
            else:
                l3.append(top3[1][0])
    elif top3[0][0] == 'Neutral':
        if len(top3) == 1:
            l3.append('Neutral')
        elif top3[1][0] == 'none':
            if len(top3) == 2: 
                l3.append('Neutral')
            else:
                if top3[2][1] >= p10_len:
                    l3.append(top3[2][0])
                else:
                    l3.append(top3[0][0])
        else:
            if top3[1][1] >= p10_len:
                l3.append(top3[1][0])
            else:
                l3.append(top3[0][0])
    else:
        l3.append(top3[0][0])
    
    return l3[0]