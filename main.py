import cv2
import mediapipe as mp
import time
import math
import numpy as np

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

shield = cv2.imread("magic_shield_1.png", cv2.IMREAD_UNCHANGED)
shield_radius=300
shield = cv2.resize(shield, dsize=(shield_radius, shield_radius))

for i in range(10):
    success, img = cap.read()
    h, w, c = img.shape



ones = np.ones((img.shape[0], img.shape[1]),dtype="uint8")*255


ang=0
speed=5

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    p8=[]
    p12=[]
    p5=[]
    p9=[]
    p0=[]
    p4=[]
    p6=[]
    p17=[]

    phands=[0,0]
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            #print(handLms)
            for id, lm in enumerate(handLms.landmark):
                # print(id, lm)
               
                cx, cy = int(lm.x * w), int(lm.y * h)
                if(id==8):p8=[cx,cy]
                if(id==12):p12=[cx,cy]
                if(id==5):p5=[cx,cy]
                if(id==9):p9=[cx,cy]
                if(id==0):p0=[cx,cy]
                if(id==4):p4=[cx,cy]
                if(id==6):p6=[cx,cy]
                if(id==17):p17=[cx,cy]
                #print(id, cx, cy)
                # if id == 4:
                #cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
            
            dtop=math.dist(p8,p12)+1
            dbtm=math.dist(p5,p9)+1
            dpalm=math.dist(p0,p9)+1
            dthumb=math.dist(p4,p6)+1

            speed=(dthumb/(dpalm/3))*10

            if(dtop<=(dbtm*1.5)):
                sr=4
            else:
                sr=int(dpalm*3*(dtop/dbtm/2.5))
                
            cnx=int((p9[0]+p0[0])//2)
            cny=int((p9[1]+p0[1])//2)


            basics=[sr,(cnx,cny)]

            if(p17[0]<p5[0]):phands[0]=basics
            else:phands[1]=basics
            
            
            #mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            
        print(phands)
        img = np.dstack([img, ones])
        
        for singlehand in phands:
            if(singlehand!=0):
                replace_shiled=shield.copy()
                shield_radius=singlehand[0]
   
                shield = cv2.resize(shield, dsize=(shield_radius, shield_radius))

                sh,sw,sc=shield.shape
                M = cv2.getRotationMatrix2D((sh//2, sw//2),ang, 1.0)
                shield = cv2.warpAffine(shield, M, (sw, sh))  
                
                shiled_alph1 = shield[:, :, 3] / 410.0
                shiled_alph2  = 1 - shiled_alph1 

        
                centerx=singlehand[1][0]
                centery=singlehand[1][1]

                startx=centerx-(shield_radius//2)
                starty=centery-(shield_radius//2)
                # print("center",centerx,centery)
          
                for i in range(0, 3):
                    try:
                        img[starty:starty+shield_radius, startx:startx+shield_radius, i] = ((shiled_alph2*img[starty:starty+shield_radius, startx:startx+shield_radius, i]) + (shiled_alph1*shield[:, :, i]))
                    except:
                        continue
                                
                shield=replace_shiled.copy()

    img = cv2.resize(img, (int(1.5*w),int(1.5*h)))
    cv2.imshow("Image", img)

    ang=(ang+speed)%360


    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()

# sh,sw,sc=shield.shape

# if(centerx<(shield_radius//2)):
#     shield=shield[((shield_radius//2)-centerx):sw , 0:sh ];print(1)
# if(centery<(shield_radius//2)):
#     shield=shield[((shield_radius//2)-centery):sh, 0:sw  ];print(2)


# if(centerx>(w-(shield_radius//2))):
#     shield=shield[0:((shield_radius//2)+w-centerx) , 0:sh ];print(3)
# if(centery>(h-(shield_radius//2))):
#     shield=shield[ 0:sw , 0:((shield_radius//2)+h-centery) ];print(4)
