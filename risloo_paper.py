import cv2
import numpy as np
from statistics import mode 
import json
import os
import imutils
from imutils.perspective import four_point_transform
from imutils import contours
import argparse
from pyzbar import pyzbar
import MTM
from MTM import matchTemplates, drawBoxesOnRGB





radii = []
# Define acceptable error
radiusDelta = 3

# Define acceptable range of dimension sizes, 15-20 usually.
minCricleW = 8
#maxCricleW
minCricleH = 8
#maxCricleW
minCricleArea = ((minCricleW+minCricleH)/4)*((minCricleW+minCricleH)/4)*3.1
#maxCricleArea =
class _mcOption:
    ID = None
    questionID = None
    optionID = None
    centerX = None
    centerY = None
    circleContour = None
    radius = None
    isChecked = False
    centroidID = None

    def __init__(self, ID, questionID, optionID):
        self.ID = ID
        self.questionID = questionID
        self.optionID = optionID

    def initCenters(self, circleContour):
        self.circleContour = circleContour
        self.centerX, self.centerY, self.radius = extractFromCricleContour(circleContour)

def extractFromCricleContour(circleContour):
    (x, y, w, h) = cv2.boundingRect(circleContour)  # This boudingRect function in opencv takes in a "contour" datatype
    return [x+w/2, y+h/2, w/2]                      # and return 4 values that form a bouding rectangle of the contour.

def processImage(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurry = cv2.GaussianBlur(gray, (3, 3), 1)
   
    
    adapt_thresh = cv2.adaptiveThreshold(blurry, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    #adapt_thresh = cv2.threshold(warped, 0, 255,
    # cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU )[1]
    #input(len(adapt_thresh.shape))
    return adapt_thresh

def findCircleContours(image):
    processed_image = processImage(image)
    # cv2.fincContours return 3 values, 1st one not needed, "contours" contains all contours in the given image,
    # "hierarchy", refer https://docs.opencv.org/3.4.0/d9/d8b/tutorial_py_contours_hierarchy.html
    # for understanding, basically it describe a "level" relationship between contours and contours.
    # Opencv hierarchy structure: [Next, Previous, First_Child, Parent]
    contours, hierarchy = cv2.findContours(processed_image.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(image, contours, -1, (0,255,0), 1)
    #cv2.imshow("Contours in warped image", image)
    #cv2.waitKey(0)

    hierarchy = hierarchy[0]
    circleContours = []
    i = 0
    nCirlces = 0
    # Looping over contours one by one till the end, and for each contour, check if it satisfy our rules to be
    # considered as a valid circle, if it's valid, add this contour to the list "circleContours for storage".
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        ar = w / float(h)
        if hierarchy[i][3] == -1 and w >= minCricleW and h >= minCricleH and 0.5 <= ar and ar <= 2.1 :
            epsilon = 0.01*cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            area = cv2.contourArea(contour)
            if ( (len(approx) >= 4) & (len(approx) < 26) & (area >= minCricleArea) ):
                circleContours.append(contour)
                nCirlces = nCirlces + 1
        i = i + 1
    return [circleContours, nCirlces]   # Return a list containing another list, and the number of circles detected.


def filterBadCricles(objList):
    initN = len(objList)
    objList = sorted(objList , key=lambda k: [k.centerY, k.centerX])
    for i in range(nCirlces):
        radii.append(objList[i].radius)
    radiiMode = mode(radii)
    removed = 0
    for mcOption in objList[:]:
        if not ( mcOption.radius - radiusDelta < radiiMode and mcOption.radius + radiusDelta > radiiMode ):
            objList.remove(mcOption)
            removed = removed + 1
    return [objList, initN - removed]

def makeMCs(nCirlces):
    mcOptions_ObjList = []
    for i in range(nCirlces):
        instance = None
        instance = _mcOption(nCirlces-i, None, None)
        instance.initCenters(circleContours[i])
        mcOptions_ObjList.append(instance)
    return mcOptions_ObjList


#################################################### ُSTEP 0 : Input ####################################################

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image_folder", required=True,
	help="path to the input image folder")
args = vars(ap.parse_args())
########################################################################################################################




################################################## STEP 1 : mode of operation ###########################################



    
imgFolder = args["image_folder"]

if not os.path.exists(imgFolder):
    raise NameError("This directory was NOT found!")


onlyfiles = [f for f in os.listdir(imgFolder) if os.path.isfile(os.path.join(imgFolder, f))]





    
########################################################################################################################




################################## STEP 2 :Reading images and finding points of any page ################################
currentPage = 0
for file in onlyfiles:
    
    ERROR = False

    image = cv2.imread(os.path.join(imgFolder,file))
    currentPage = currentPage + 1
    input("Press <ENTER> to begin processing the current page [" + str(currentPage) + "]: ")


    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    # find contours in the edge map, then initialize
    # the contour that corresponds to the document
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)

    cnts = imutils.grab_contours(cnts)
    docCnt = None
    # ensure that at least one contour was found
    if len(cnts) > 0:
        # sort the contours according to their size in
        # descending order
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        # loop over the sorted contours
        for c in cnts:
            # approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)


            # if our approximated contour has four points,
            # then we can assume we have found the paper
            if len(approx) == 4:
                docCnt = approx
                break
            else :
                print("4 points of sorrounding contour can't be found in ", file, "You should take a better picture of paper !")
                ERROR = True
                break

    if ERROR :
        continue


    # apply a four point perspective transform to both the
    # original image and grayscale image to obtain a top-down
    # birds eye view of the paper
    rgb_paper = four_point_transform(image, docCnt.reshape(4, 2))
    warped = four_point_transform(gray, docCnt.reshape(4, 2))

    #cv2.imshow('warped_paper', rgb_paper)
    #cv2.waitKey()

    
    # create a CLAHE object
    CLAHE_CLIPLIMIT = 4.0
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIPLIMIT)
    # apply CLAHE for better contrast
    
    warped = clahe.apply(warped)
    

    #cv2.imshow("warped_gray_paper",cv2.resize(warped,(720,720)))
    #cv2.waitKey()

#########################################################################################################################################



################################## STEP 3 : Proccessing QR code AND paper parameters #####################################################
    ### just for test ###
    # find the barcodes in the image and decode each of the barcodes
    
    part_paper = rgb_paper.copy()
    
    ############################### Finding our QR code in all 4 parts of image
    
    height , width ,_= part_paper.shape
    
    Found = False
    for i in range(4): # 4parts of page
        part_paper = rgb_paper.copy()
        if i==0 :
            part_paper = part_paper[0 :round(height*0.25)  , round( width*0.75):,:]
        elif i==1 :
            part_paper = part_paper[0 :round(height*0.25)  , 0 :round(width*0.25),:]
        elif i==2 :
            part_paper = part_paper[round(height *0.75):  , 0 :round(width*0.25),:]
        elif i==3 :
            part_paper = part_paper[round(height *0.75):   , round( width*0.75):,:]
        
        

        part_paper = cv2.cvtColor(part_paper, cv2.COLOR_BGR2GRAY)
        
        
        
        #cv2.imshow("ss",part_paper)
        #cv2.waitKey()

        barcodes = pyzbar.decode(part_paper)
        if len(barcodes) !=0 :
            rotation_count = i
            Found = True
        else :
            
            #part_paper = clahe.apply(part_paper)
            part_paper = cv2.threshold(part_paper, 0, 255,
            cv2.THRESH_BINARY | cv2.THRESH_OTSU )[1]#
            
            
            barcodes = pyzbar.decode(part_paper)
            
            if len(barcodes) !=0 :
                rotation_count = i
                Found = True
        if Found :
            break
        
        ##################################################################
    
    new_paper = rgb_paper.copy()
    paper = cv2.cvtColor(new_paper, cv2.COLOR_BGR2GRAY)
    
      
    # if there is a QR code
    
    if len(barcodes) !=0 :
        for barcode in barcodes:
            barcodeData = barcode.data.decode("utf-8")
            
            paper_information = barcodeData.split('&utm_content=')
            
            if paper_information is not None :
                paper_information = paper_information[1].split('&')[0]
                paper_information = paper_information.split('-')
                
                num_questions =int(paper_information[0])
                nchoices = int(paper_information[1])
                num_question_each_col = int(paper_information[2])

                ncols =(num_questions // num_question_each_col) + 1
                num_questions_last_column = num_questions %  num_question_each_col
            else :
                print("QR code detected but didn't read its information in ", file, "You should take a better picture of paper !")
                ERROR = True
                break

             
            
            # length of bounding box
            
            (x, y, w, h) = barcode.rect
            cv2.rectangle(paper, (x, y), (x + w, y + h), (0, 0, 255), 2)
                
    
    else :
        print("No QR code detected in ", file, "You should take a better picture of paper !")
        ERROR = True

    if ERROR :
        continue
    
    # display the result
    #cv2.imshow("img", paper)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()



#################################################################################################################################




#############################################  STEP 4 :Correct orientation of page ##############################################
   

    height = paper.shape[0]
    width = paper.shape[1]
  

    # Rotate the image such that the page is correctly oriented
    paper = rgb_paper.copy()
    
    for c in range(rotation_count):
        paper = cv2.transpose(paper)
        paper = cv2.flip(paper, flipCode=1)
        warped = cv2.transpose(warped)
        warped = cv2.flip(warped, flipCode=1)
        

    # display the result
    
    #cv2.imshow("rotated page", cv2.resize(paper,(620,620)))
    #cv2.waitKey(0)


#########################################################################################################################################



############################################### STEP 5 :Template matching for find our page #############################################


    paper_gray = cv2.cvtColor(paper, cv2.COLOR_BGR2GRAY)
    
    from pathlib import Path

    current_directory = str(Path(__file__).resolve().parent) # like this /home/skovorodkin/stack/scripts

    print("current_directory is: "+ current_directory)

    omr_marker = cv2.imread(current_directory + r'/omr_marker.jpg',0)
    
    
    marker_width = omr_marker.shape[1]
    marker_height = omr_marker.shape[0]

    

    height = paper_gray.shape[0]
    width = paper_gray.shape[1]

   
 
    
    allMaxT =0
    
    for scale in np.linspace(0.1,0.3, 100): 
        Sum_scores = 0
        # resize the marker according to the scale, and keep track 
        # of the ratio of the resizing 
        new_copy = paper_gray.copy()
        resized = imutils.resize(omr_marker, width= int(omr_marker.shape[1] * scale)) 
        r = omr_marker.shape[1] / float(resized.shape[1]) 
        
        
        if resized.shape[0] >height or resized.shape[1] > width: 
                break
        
        
        listTemplate = [('marker', resized)]
        Hits = matchTemplates(listTemplate, new_copy, N_object=4,score_threshold=0.4,   
            maxOverlap=0, method=cv2.TM_CCOEFF_NORMED)#,searchBox=(76,781,1856,353)

        Overlay = paper.copy()
        
        for index in Hits.index:
            x1,y1,w,h = Hits.BBox[index]
            Overlay = drawBoxesOnRGB(paper, Hits)
            Sum_scores = Sum_scores + Hits.Score[index]
            
        #print("Sum_scores : ",Sum_scores)

            
        
        #cv2.imshow('template_matched_paper',cv2.resize(Overlay,(620,620))) 
        #cv2.waitKey(0)
        
        
        

        ### Checking is it the best scale ? 
        if(allMaxT < Sum_scores):
            centres = []
            allMaxT , best_scale = Sum_scores , scale
            Best_BBox = Hits.BBox.copy()
            
            h_resized = resized.shape[0]
            w_resized = resized.shape[1]
            
            for index in Hits.index:
                x1,y1,w,h = Best_BBox[index]
                centre = np.array([x1+ w_resized//2 , y1 + h_resized//2 ])
                centres.append(centre)
                
    
    centres = np.asarray(centres)
    

    if centres.shape[0] ==4 :
        paper = four_point_transform(paper,centres)   
        warped = four_point_transform(warped,centres)   
         

    else :
        print("Please Scan Again! markers not found in ", file)
        ERROR = True

    if ERROR:
        continue    


    #cv2.imshow('template_matched_paper',cv2.resize(paper,(620,620))) 
    #cv2.waitKey(0)
    cv2.imwrite(current_directory + r"/adjusted_tests/adjusted_" + file , paper )

############################################################## ############################################################## 







######################################################## STEP 6 : finding circle contours ########################################


    circleContours, nCirlces = findCircleContours(paper.copy()) #paper or thresh ?
    print(nCirlces)
    mcOptions_ObjList = makeMCs(nCirlces)
    mcOptions_ObjList, nCirlces = filterBadCricles(mcOptions_ObjList)

    

    # Show results
    del circleContours[:]
    for mcOption in mcOptions_ObjList[:]:
        circleContours.append(mcOption.circleContour)
    cv2.drawContours(paper, circleContours,  -1, (0,0,255), 1)
    print(nCirlces, "Circles Detected")
    #cv2.imshow('Circles Detected',cv2.resize(paper,(620,620)))
    #cv2.waitKey(0)



    #check if circles not found correctly
    if nCirlces!=(nchoices*num_questions):
        print("Please Scan Again! choices not found completely !", file)
        ERROR = True
    
    if ERROR:
        continue 
############################################################## ############################################################## 



    ####################################################### STEP 7 : Sorting contours ########################################


    # update the threshold image 
    #warped = clahe.apply(warped)
    thresh = cv2.threshold(warped, 0, 255,
        cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU )[1]

    

    # sort the question contours top-to-bottoms

    questionCnts = contours.sort_contours(circleContours,
        method="top-to-bottom")[0]


    # Answers json 
    temp_Answers={}
    location_Answers={}
    Answers = {}


    i=0
    q=0
    while( i < len(questionCnts)):
        # sort the contours for the current row from
        # right to left , then initialize the index of the
        # bubbled answer
        
        temp_cnts = contours.sort_contours(questionCnts[i:(i+(nchoices*ncols))],method="right-to-left")[0]

        # Check each coloumn of each row
        for ncol in range(ncols):
            cnts = temp_cnts[ncol*nchoices:((ncol+1)*nchoices)]

            bubbled = None

            cv2.drawContours(paper, cnts,-1,[0,255,255],3)
            #cv2.imshow('line_by_line_Contours',cv2.resize(paper,(1000,820)) )
            #cv2.waitKey(0)

            
            counter = 0
            choices =[]
            locations = []
            # loop over the sorted contours
            for (j, c) in enumerate(cnts):
                # construct a mask that reveals only the current
                # "bubble" for the question
                mask = np.zeros(thresh.shape, dtype="uint8")
                cv2.drawContours(mask, [c], -1, 255, -1)
                # apply the mask to the thresholded image, then
                # count the number of non-zero pixels in the
                # bubble area
                mask = cv2.bitwise_and(thresh, thresh, mask=mask)
                total = cv2.countNonZero(mask)
                # if the current total has a larger number of total
                # non-zero pixels, then we are examining the currently
                # bubbled-in answer
                
                if bubbled is None or total > bubbled[0]:
                    bubbled = (total,j)
                   
                
                # Check area of fiiled circle
                if total >= (cv2.contourArea(c)*0.85) :
                    counter +=1
                    choices.append(j+1) 
                    (x, y, w, h) = cv2.boundingRect(c)
                    locations.append([x+w/2, y+h/2])

             
            if counter >=1 :
                
                temp_Answers[str(num_question_each_col * ncol +(q+1))] = choices
                location_Answers[str(num_question_each_col * ncol +(q+1))] = locations
            
                

        
        
        
        i = i + (nchoices*ncols)
        ##### it's for sometimes last column is not a complete column
        if q+1 == num_questions_last_column :
            ncols = ncols-1
        
        ##### Updating variable
        
        q = q + 1
        
    ###################################################################################################





    #########################################  STEP 8 : Sorting question numbers ########################
    question_numbers = temp_Answers.keys()
    question_numbers_list = []
    
    for question_number in question_numbers:
        question_numbers_list.append(int(question_number))

    question_numbers_list.sort()

    
    Answers['qrcode'] = barcodeData
    Answers['papers'] = current_directory + r"/adjusted_tests/adjusted_" + file
    Answers['items'] = []

    for question_number in question_numbers_list:
        for i in range(len(temp_Answers[str(question_number)])):
            choice_dic ={}
            choice_dic['item'] = str(question_number)
            choice_dic['choices'] = temp_Answers[str(question_number)][i]
            choice_dic['location'] = location_Answers[str(question_number)][i]
            Answers['items'].append(choice_dic)    
        


    
       
    #print(Answers)
    ######################################### ######################################### #######################



    ######################################### STEP 9 :saving json file ########################################
    save_path = current_directory+ r"/Answers/Answers_" + file[:-4] + r".json"
    with open(save_path, 'w') as outfile:
        json.dump(Answers, outfile)


