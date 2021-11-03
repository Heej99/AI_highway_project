# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 10:13:55 2021

@author: KONIDE

코드 설명 
- 파노라마 생성에 필요한 코드 정의 
"""


##----------------Stitcher------------------

import numpy as np
import cv2
import imutils # imutils 는 OpenCV가 제공하는 기능 중에 좀 복잡하고 사용성이 떨어지는 부분을 잘 보완해 주는 패키지
               # 기본적으로 모두 OpenCV의 기능을 사용하고 있기 때문에 해당 기능을 사용하는 것은 아주 권장

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
        
        topHeight = 565
        height, width = self.img.shape[:2]
        #print('height', height,'width :', width)
        left = [(960, 380), (0, 650)]
        right = [(960, 380), (1920, 650)]
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
         
        dst = np.array([[0,0], [0, 565], [1080,0], [1080,565]], dtype=np.float32)
        src = np.array([ [p1x,p1y], [p2x,p2y], [p3x,p3y], [p4x,p4y]], dtype=np.float32)
        mtrx = cv2.getPerspectiveTransform(src, dst) #4개의 꼭짓점으로 정확한 원근 변환 행렬을 반환
        
        outimg = cv2.warpPerspective(self.img, mtrx, (1080,560))
        #cv2.imshow('out_image',outimg)
        #cv2.waitKey()
        #cv2.destroyAllWindows()
        return outimg


class Stitcher: 
    
    """
    정사영 이미지들을 특징매칭하여 파노라마를 생성하는 class
    """
    
    def __init__(self):
		# determine if we are using OpenCV v3.X and initialize the
		# cached homography matrix
        self.isv3 = imutils.is_cv3()
        self.cachedH = None
        
    def stitch(self, images, ratio=0.75, reprojThresh=4.0):
		# unpack the images
        (imageB, imageA) = images
        
		# if the cached homography matrix is None, then we need to
		# apply keypoint matching to construct it
        if self.cachedH is None:
			# detect keypoints and extract
            (kpsA, featuresA) = self.detectAndDescribe(imageA)
            (kpsB, featuresB) = self.detectAndDescribe(imageB)
		# match features between the two images
            M = self.matchKeypoints(kpsA, kpsB,featuresA, featuresB, ratio, reprojThresh)
            
			# if the match is None, then there aren't enough matched
			# keypoints to create a panorama
            if M is None:
                return None
			# cache the homography matrix
            self.cachedH = M[1]
		# apply a perspective transform to stitch the images together
		# using the cached homography matrix
        result = cv2.warpPerspective(imageA, self.cachedH,(imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
        
        result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
		# return the stitched image
        return result
        
    def detectAndDescribe(self, image):
        """
        특징점과 특징 디스크립터를 찾아주는 함수
            - 특징점 :  특징점은 영어로 키 포인트(Keypoints)라고도 합니다. 보통 특징점이 되는 부분은 물체의 모서리나 코너
            - 특징 디스크립터 : 특징점 주변 픽셀을 일정한 크기의 블록으로 나누어 각 블록에 속한 픽셀의 그레디언트 히스토그램을 계산한 것. 
                                주로 특징점 주변의 밝기, 색상, 방향, 크기 등의 정보가 포함되어 있
        """
		# convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		# check to see if we are using OpenCV 3.X
        if self.isv3:
			# detect and extract features from the image
            descriptor = cv2.xfeatures2d.SIFT_create()
            (kps, features) = descriptor.detectAndCompute(image, None) # 각 이미지에 대해 키포인트와 디스크립터 추출
                                                                       # 디스크립터 : keypoint에 해당하는 정보 <- 실제 유사도를 판별하기 위한 데이터로 활용됨.            
		# otherwise, we are using OpenCV 2.4.X
        else:
			# detect keypoints in the image
            detector = cv2.SIFT_create()
            kps = detector.detect(gray)
			# extract features from the image
            # extractor = cv2.DescriptorExtractor_create("SIFT")
            (kps, features) = detector.compute(gray, kps)
            
		# convert the keypoints from KeyPoint objects to NumPy
		# arrays
        kps = np.float32([kp.pt for kp in kps]) # kp.pt : 특징점의 좌표
		# return a tuple of keypoints and features 
        return (kps, features)
    
    
    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB,ratio, reprojThresh):
        """
        특징 매칭된 keypoint 찾아주는 알고리즘
        - 특징 매칭 : 서로 다른 두 이미지에서 특징점과 특징 디스크립터들을 비교해서 비슷한 객체끼리 짝짓는 것
        """
		# compute the raw matches and initialize the list of actual
		# matches
        matcher = cv2.DescriptorMatcher_create("BruteForce") # 원하는 매칭 알고리즘이 BruteForse인 특징 매칭기
                                                             # Brute-Forse 매칭기 queryDescriptor와 trainDescriptor를 일일이 전수조사해서 매칭하는 알고리즘
            
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)  # 여러개의 특징 매칭점들(BruteForse에 의해 전수조사됨)에 대한 k개의 근접 이웃 개수 = 각 행에 거리 값이 작은 순서대로 리스트에 추가
        # queryDescriptors 한개당 최근접 이웃 개수 만큼 trainDescriptor 한개 찾아 결과 반영,최적 매칭 없을 수도 있음, DMatch 객체 리스트 
        # k의 값에 따라 결과 매칭의 각행에 거리 값이 작은 순서대로 리스트에 추가
        
        '''        
        # knnMatch(queryDescriptors, trainDescriptors, k) <- 특징 매칭 방식으로 채택된 것
        
        # k(매칭할 근접 이웃 개수)개의 가장 근접한 매칭
        # queryDescriptors : 매칭의 기준이 될 특징 디스크립터 배열 - queryInx : queryDescriptors의 인덱스
        # trainDescriptors : 매칭의 대상이 될 특징 디스크립터 배열 - trainIdx : trainDescriptors의 인덱스
        
        '''
        
        matches = []
		# loop over the raw matches
        for m in rawMatches:
			# ensure the distance is within a certain ratio of each
			# other (i.e. Lowe's ratio test)
            if len(m) == 2 and m[0].distance < m[1].distance * ratio: matches.append((m[0].trainIdx, m[0].queryIdx))
            # m.distance : 유사도 거리
    
	# computing a homography requires at least 4 matches
        if len(matches) > 4:
			# construct the two sets of points
            ptsA = np.float32([kpsA[i] for (_, i) in matches]) # _ :  의미없는 변수
            ptsB = np.float32([kpsB[i] for (i, _) in matches])
			# compute the homography between the two sets of points
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,reprojThresh) # 여러 개의 점으로 근사 계산한 원근 변환 행렬 반환
            
			# return the matches along with the homograpy matrix
			# and status of each matched point
        return (matches, H, status)
		# otherwise, no homograpy could be computed
 
    def makeImagesList(self):       
        """
        비디오에서 프레임 뽑아서 정사영 형태로 변환한 이미지들을 리스트에 저장하는 역할. 
        """
        cap = cv2.VideoCapture('C:/OpenCV/GH021047.MP4') # 동영상 파일을 읽어옴
        
        #startFrame = 100 + skipSeconds*60        
        fps = round(cap.get(cv2.CAP_PROP_FPS)) # get frame numbers = 프레임속도
                    # cap.get(propid) # 동영상 속성 반환
        
        #delay = int(5000/fps)
        
        if (fps == 0) :
            fps = 60 # 속력 60으로 고정
            
        #frameNumber = cap.get(cv2.CAP_PROP_FRAME_COUNT) # total frame numbers
        # frames = cap.get(cv2.CAP_PROP_POS_FRAMES) # current frame numbers
        
        #img = cv2.imread('c:/OpenCV/image-003.jpeg')
        
        i = 0        
        #stitcher = Stitcher()
        result = []
        
        while cap.isOpened(): # isOpened() 동영상 파일 열기 성공 여부 확인
            ret, frame = cap.read() # 비디오의 한프레임씩 읽음 
                                    # 제대로 프레임 읽으면 ret = True, 아니면 ret = False
                                    # frame은 읽은 프레임 나옴
            if not ret :
                break
            
            curr_frame = IMP(frame)   
            curr_outimg = curr_frame.impTransformer() # 정사영으로 변환하기
            curr_cropimg = curr_outimg[0:250, 0:1920] # 그중 위에 제대로 있는 부분만 사용
            curr_cropimg = cv2.rotate(curr_cropimg, cv2.cv2.ROTATE_90_CLOCKWISE)
            result.append(curr_cropimg)
                
            i += 1
            
            #cv2.imwrite('road'+str(i)+'.jpg', curr_cropimg)
        
            if result is None:
                print('[info] homography could not computed')
                break
              
          #  cv2.imwrite('outimg'+ str(i)+'.jpg',outimg)    
        
        return result
    