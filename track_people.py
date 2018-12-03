import numpy as np
import cv2
import Person

cnt_up   = 0
cnt_down = 0

cap = cv2.VideoCapture('test6.mp4')#mở video file

#lấy độ phân giải video
w = cap.get(3)
h = cap.get(4)
frameArea = h*w
areaTH = frameArea/400

#xây dựng các đường thẳng và vùng trên video
line_down   = int(3*(h/5))
line_up = int(2*(h/5))

up_limit =   int(1*(h/5))
down_limit = int(4*(h/5))


pt1 =  [0, line_down];
pt2 =  [w, line_down];
pts_L1 = np.array([pt1,pt2], np.int32)
pts_L1 = pts_L1.reshape((-1,1,2))
pt3 =  [0, line_up];
pt4 =  [w, line_up];
pts_L2 = np.array([pt3,pt4], np.int32)
pts_L2 = pts_L2.reshape((-1,1,2))


#khởi tạo loại bỏ background
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows = True)

#tạo các kernel cho morphographic filters
kernelOp = np.ones((3,3),np.uint8)

kernelCl = np.ones((11,11),np.uint8)

#khởi tạo các biến ban đầu
font = cv2.FONT_HERSHEY_SIMPLEX
persons = []
max_p_age = 5
pid = 1

while(cap.isOpened()):
    #đọc từng frame
    ret, frame = cap.read()

    for i in persons:
        i.age_one() 

    
    #loại bỏ background
    fgmask = fgbg.apply(frame)

    try:
        ret,imBin= cv2.threshold(fgmask,190,255,cv2.THRESH_BINARY)
        mask1 = cv2.morphologyEx(imBin, cv2.MORPH_OPEN, kernelOp)
        mask =  cv2.morphologyEx(mask1 , cv2.MORPH_CLOSE, kernelCl)
        cv2.imshow('Background Substraction',fgmask)
    except:
        print('End')
        break
    
    _, contours0, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours0:
        cv2.drawContours(frame, cnt, -1, (0,255,0), 3, 8)
        area = cv2.contourArea(cnt)
        if area > areaTH:
            
            #thêm điều kiện khi xuất hiện nhiều người cùng lúc
            
            M = cv2.moments(cnt)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            x,y,w,h = cv2.boundingRect(cnt)

            new = True
            #if cy in range(up_limit,down_limit):
            for i in persons:
                if abs(cx-i.getX()) <= w and abs(cy-i.getY()) <= h:
                    # đối tượng gần với đối tượng đã được xác định trước đó
                    new = False
                    i.updateCoords(cx,cy)   #cập nhật lại vị trí đối tượng
                    if i.going_UP(line_down,line_up) == True:
                        cnt_up += 1;
                        
                    elif i.going_DOWN(line_down,line_up) == True:
                        cnt_down += 1;
                        
                    break
                if i.timedOut():
                    index = persons.index(i)
                    persons.pop(index)
                    del i
            if new == True:
                p = Person.MyPerson(pid,cx,cy, max_p_age)
                persons.append(p)
                pid += 1     

            #vẽ các vùng xác định được
            cv2.circle(frame,(cx,cy), 5, (0,0,255), -1)
            img = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)            
            
    #kết thúc vòng lặp for i in persons
    #vẽ các hướng di chuyển
    for i in persons:
        if len(i.getTracks()) >= 2:
            pts = np.array(i.getTracks(), np.int32)
            pts = pts.reshape((-1,1,2))
            frame = cv2.polylines(frame,[pts],False,i.getRGB())

        cv2.putText(frame, str(i.getId()),(i.getX(),i.getY()),font,0.3,i.getRGB(),1,cv2.LINE_AA)
        
    #hiện các thông số, khu vực hoặc đường xác định
    str_up = 'UP: '+ str(cnt_up)
    str_down = 'DOWN: '+ str(cnt_down)
    frame = cv2.polylines(frame,[pts_L1],False,(0,255,0),thickness=2)
    frame = cv2.polylines(frame,[pts_L2],False,(0,255,0),thickness=2)

    cv2.putText(frame, str_up ,(10,40),font,0.5,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(frame, str_up ,(10,40),font,0.5,(0,0,255),1,cv2.LINE_AA)
    cv2.putText(frame, str_down ,(10,90),font,0.5,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(frame, str_down ,(10,90),font,0.5,(255,0,0),1,cv2.LINE_AA)

    cv2.imshow('Frame',frame)
    cv2.imshow('Mask',mask)
    cv2.imshow('Mask1',mask1)       
    
    #kết thúc và thoát vòng lặp bằng nút Q hoặc ESC
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
#kết thúc vòng lặp while(cap.isOpened())
    

cap.release()
cv2.destroyAllWindows()