# emotion_detection_byVideo
가)	인식 모델 구성  
①	인식 모델은 이미지 데이터에 대한 인지 모델로 설계되어 있음 (비디오 데이터 x)  
②	FER2013 Dataset으로 학습된 fer2013_mini_XCEPTION.110-0.65 사용  
나)	비디오 데이터 구성  
①	Presentation and validation of the DuckEES child and adolescent dynamic facial expressions stimulus set
    Background  
    1) 대상: 8-18세 남, 여 36명이 7가지 표정을 지음(평균 13.24세, 남성 14명), 142개의 비디오   
    2) 종류: 영상(1.1초) / 6가지 감정(+ 중립)3) 검증: 36명이 Labeling(평균 19.5세, 남성 13명)   
    Methods  
    1) 감정표현 세션을 진행  
    2) 세션이 끝난 후, 세션 비디오 전체를 검토하고 감정표현을 가장 잘 묘사한 부분을 1.1초짜리 클립으로 잘라냄   
다)	평가 방법  
①	Frame단위로 평가하는 인지모델의 output을 통해 비디오가 가지고 있는 true label과 비교해 정확도를 측정  
②	Video를 통한 모델의 output은 frame 개수만큼 나오지만, video는 하나의 label만 가지고 있음 → 일대일의 비교가 어렵기 때문에 모델의 output을 특정 기준을 통해 평가하는 알고리즘이 필요  
