

import numpy as np
import cv2
import imutils


class Line : 
    """
    투시변환할 부분을 찾아주는 class
    """
    def __init__(self, data1, data2):
   
        self.line1 = data1
        self.line2 = data2
        #print(self.line1)
    def slope(self):
        (x1, y1), (x2, y2) = self.line1
        (x3, y3), (x4, y4) = self.line2
        
        if (y2-y1) == 0 :
            #print('Ys are equal, m1 = 0')
            m1 = 0
        else:
            m1 = (float(y2)-y1)/(float(x2)-x1)
        
        if (y4-y3) == 0 :
            #print('Ys are equal, m2 = 0')
            m2 = 0
        else:
            m2 = (float(y4)-y3)/(float(x4)-x3)
            
        return m1, m2
                    
    def yintercept(self, m1, m2):
        (x1, y1), (x2, y2) = self.line1
        (x3, y3), (x4, y4) = self.line2
        
        if m1 != 0 :
            b1 = y1 - m1*x1
        else :
            b1 = y1
            
        if m2 != 0 :
            b2 = y4 - m2*x4
            
        else: b2 = y4
        
        return b1, b2
    
    def findIntersect(self, m1,m2, b1, b2):
        
        if m1 != 0 | m2 != 0 :
            px = (b2-b1) / (m1-m2)
            py = (b2*m1 - b1*m2)/(m1-m2)
        elif m1 == 0 :
            px = (b1-b2)/m2
            py = b1
        elif m2 == 0 : 
            px = (b2-b1)/m1
            py = b2 
        else :  print('No points')
        
        return px, py

class IMP: 
    """
    투시변환 실시하여 전경이미지 -> 정사영으로 변환해주는 class
    """
    def __init__(self, img):
        
        #import cv2
        #img = cv2.imread('c:/OpenCV/image-003.jpeg')     
        self.img = img
        
        #self.topHeight = 565
        #self.height, self.width = 1080, 1920
        
    def impTransformer(self):  
        
        import numpy as np
        import cv2 
        
        topHeight = 545
        height, width = self.img.shape[:2]
        left = [(960, 342), (0, 690)]
        right = [(960, 342), (1920, 650)]
        up =  [(0, topHeight), (width+1000, topHeight)]
        down =  [(-10000,height), (width+100000, height)]
        
        leftup = Line(left, up)
        leftdown = Line(left, down)
        rightup = Line(right, up)
        rightdown = Line(right, down)
        m1, m2 = leftup.slope()
        b1, b2 = leftup.yintercept(m1,m2)
        p1x, p1y = leftup.findIntersect(m1,m2,b1,b2)
        
        #print('point1 : ', p1x, p1y)
        
       
        m1, m2 = leftdown.slope()
        b1, b2 = leftdown.yintercept(m1,m2)
        p2x, p2y = leftdown.findIntersect(m1,m2,b1,b2)
        #print('point2 : ', p2x, p2y)
        
       
        
        m1, m2 = rightup.slope()
        b1, b2 = rightup.yintercept(m1,m2)
        p3x, p3y = rightup.findIntersect(m1,m2,b1,b2)
        #print('point3 : ', p3x, p3y)
        
        m1, m2 = rightdown.slope()
        b1, b2 = rightdown.yintercept(m1,m2)
        p4x, p4y = leftup.findIntersect(m1,m2,b1,b2)
        #print('point4 : ', p4x, p4y)
         
        dst = np.array([[0,0], [0, 1080], [1920,0], [1920,1080]], dtype=np.float32)
        src = np.array([ [p1x,p1y], [p2x,p2y], [p3x,p3y], [p4x,p4y]], dtype=np.float32)
        mtrx = cv2.getPerspectiveTransform(src, dst) 
        
        transformedFHD = cv2.warpPerspective(self.img, mtrx, (1920,1080))
        ##  C++코드에서 정사영 자체는 원본 사이즈로 생성,  ( 1080, 560 ) -> (1080,1920)
        outimg = cv2.resize(transformedFHD,dsize=(960,540))
        ## C++ 코드에서 정사영을 resize 함. 
        
        #cv2.imshow('out_image',outimg)
        #cv2.waitKey()
        #cv2.destroyAllWindows()
        return outimg




class PerspectiveTransformer:
    """
    비디오에서 프레임 뽑아서 정사영 형태로 변환한 이미지들을 리스트에 저장하는 역할. 
    -> 속도(m/s)의 변화를 고려하여 프레임 추출 
    """
    
    def __init__(self,video_path,gps_info):
        self.video_path = video_path
        self.gps_info = gps_info
        
    def CalckFrame_skip(self,speed,fps):
        magicNum=3.6 # gps가 m/s 이므로 얘도 m로 기준잡기
        if (speed>=3.6):
            skipVal = (magicNum*fps)/speed
        else:
            skipVal = (magicNum*fps)/3.6
        return skipVal
    
    def makeImagesList(self):
        video_path =self.video_path 
        gps_info = self.gps_info
        cap = cv2.VideoCapture(video_path) 

        #skipSeconds = 0 
        #startFrame = 100 + skipSeconds * 60
        #cap.set(cv2.CAP_PROP_POS_FRAMES, startFrame)   

        fps = round(cap.get(cv2.CAP_PROP_FPS))  # get frame numbers per seconds
        if (fps == 0) :
            fps = 60 

        frameNumber = cap.get(cv2.CAP_PROP_FRAME_COUNT) # total frame count

        # if currentFrameNumber >= frameNumber:
        #     print("VideoProcessor: selected video ended")

        currentFrameNumberTemp = 1

        result = []
        
        while cap.isOpened():

            # read frame 
            ret, frame = cap.read()

            # variable define
            WaitGPS = True
            currentFrameNumber=cap.get(cv2.CAP_PROP_POS_FRAMES) #  frame number , start 1 

        #     if True:
        #         # Match video frame with gps coordinate
        #         if (currentFrameNumber%6==0):
        #             waitGPS = False
        #         lastSecond = currentFrameNumber / 6 # fps for 1 hz

            if (cap.get(cv2.CAP_PROP_FRAME_WIDTH)*cap.get(cv2.CAP_PROP_FRAME_WIDTH)==0):
                print("No frame earned: skip this frame (%d)\n", currentFrameNumber)
                if (currentFrameNumber>=frameNumber):
                    print("last frame print")
                    break
                continue

            stop = False    

            # reachs end of Mp4 file 
            if (currentFrameNumber>(frameNumber-10)):
                print("End of MP4file")
                stop = True
                break

            # skipping frame 
            currentFrameNumber_idx = int(currentFrameNumber)
            if ((currentFrameNumber_idx//60) < len(gps_info)):
                pass
            else:
                break

            speed = gps_info.iloc[currentFrameNumber_idx//60]['GPS (2D speed) [m/s]']
            skipVal=self.CalckFrame_skip(speed,fps)
            if(currentFrameNumber!=1)&(currentFrameNumber-skipVal<currentFrameNumberTemp):
                continue

            currentFrameNumberTemp = currentFrameNumber
            #print("check",currentFrameNumberTemp)

            # save frameFHD
            #cv2.imwrite("./result/full_image/%s.jpg"%('frame_image_'+str(currentFrameNumber_idx)),frame)

            # transformation frame
            curr_frame = IMP(frame)   
            curr_outimg = curr_frame.impTransformer()
            curr_cropimg = curr_outimg[0:250, 0:960] 
            curr_cropimg = cv2.rotate(curr_cropimg, cv2.cv2.ROTATE_90_CLOCKWISE) # (960,250)
            result.append(curr_cropimg)
            
            if result is None:
                print('[info] homography could not computed')
                break
                
          #  cv2.imwrite('outimg'+ str(i)+'.jpg',outimg)    

        return result


