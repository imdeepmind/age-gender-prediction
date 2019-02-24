import cv2
import keras

camera = cv2.VideoCapture(0)
haar = cv2.CascadeClassifier('cascades/haarcascade_frontalface_alt2.xml')

model = keras.models.load_model('gender/custom/gender.h5')
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

age_model = keras.models.load_model('/age/csutom/age.h5')
age_model.compile(loss='mean_squared_error', optimizer='adam')

while True:
    try:
        ret, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        face = haar.detectMultiScale(gray, 1.2, 5)
        
        for x,y,w,h in face:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            roi = gray[x:x+w, y:y+h]
            
            target = cv2.resize(roi, (64,64))
            target = target.reshape(-1, 64,64,1)
            
            dt = model.predict_classes(target)
            
            if dt[0] == 0:
                gname = 'Female'
            else:
                gname = 'Male'

            age = age_model.predict_classes(target)

            text = "Gender: " + str(gname) + " Age: " + str(int(age[0]))
            
            cv2.putText(frame, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)
            
        cv2.imshow('ME', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as ex:
        print(ex)
        

camera.release()
cv2.destroyAllWindows()
