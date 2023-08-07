#importing the necessary libraries
import mediapipe as mp
import cv2
import numpy as np
import autopy as ap

mp_drawing = mp.solutions.drawing_utils #used for rendering the landmarks
mp_hands= mp.solutions.hands #already present hand utility in mediapipe

#start video 
cap=cv2.VideoCapture(0) #argument specified is the camera number

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    
    while cap.isOpened():
         wScr,hScr = ap.screen.size()

        #read frame-by frame
         ret, frame = cap.read()
        
        #change default frame color from BGR to RGB
         image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        #invert the image as it is laterally inverted
         image=cv2.flip(image,1) #1 means flipping about y axis; 0 for x axis; -1 for both axis

         #frame = cv2.flip(frame,1)

         #stop loading further image frames
         image.flags.writeable = False
         #frame.flags.writeable = False

         #process the current image frame
         results = hands.process(image)

         #revert back to the previous parameters
         #image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

         image.flags.writeable = True
         #frame.flags.writeable = True
         #print(results.multi_hand_landmarks)

         #landmark detection

         def handTracking(image):
         #checks condition if hands are detected
              landmarkList = [] #create list for storing data about landmarks
              if (results.multi_hand_landmarks):
                  
                  for hand in (results.multi_hand_landmarks):
                      for index,position in enumerate(hand.landmark):
                       
                              mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS,
                                                mp_drawing.DrawingSpec(color=(185,22,76), thickness=4, circle_radius=4), #landmark
                                                                      mp_drawing.DrawingSpec(color=(222,44,250), thickness=4,circle_radius=4))   #connection
                              h,w,c = image.shape
                              centerX,centerY  = int (position.x*w), int (position.y*h)#converts to decimal coordinates relative to the dimensions of the image
                              landmarkList.append([index,centerX,centerY])
              #print(landmarkList)
              return landmarkList       
                         
         def fingerTips(lmList):
              finger_positions = []
              tipID = [4,8,12,16,20]

              #thumb
              if lmList[tipID[0]][1]>lmList[tipID[0]-1][1]: #check if x-coordinate of top joint is greater than that of the joint preceding it
                  finger_positions.append(1)
              else:
                  finger_positions.append(0)
                  
              # remaining fingers
              for i in range(1,5): #loop runs for remaining for fingers
                  
                  if lmList[tipID[i]][2]<lmList[tipID[i]-1][2]: #check if y-coordinate of top joint is less than that of the joint preceding it
                      finger_positions.append(1)
                  else:
                      finger_positions.append(0)
              return finger_positions
                 
      
         lmlist = handTracking(image)
         if (results.multi_hand_landmarks):
               x1,y1 = lmlist[8][1:]
               xf = np.interp(x1,(75,640),(0,wScr))
               yf = np.interp(y1,(75,480),(0,hScr))
               
               
               if fingerTips(lmlist) == [0,1,0,0,0]:
                   ap.mouse.move(xf,yf)
               elif fingerTips(lmlist) == [0,1,1,0,0]:
                   ap.mouse.click()

        #display the image window        
         cv2.imshow('Hand Tracking', image)

         
        #terminate the video response
         if cv2.waitKey(1) &0xFF==ord('q'):
             break

cap.release()
cv2.destroyAllWindows()
#mp_drawing.DrawingSpec()
            
         
        

        
                
    
