from flask import Flask, request, render_template, send_file, Response
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import os
import gc
import time
import matplotlib.pyplot as plt


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024

path_img_test = "to_test"

net_path = "custom-yolov4-detector_best.weights"

cfg_path = "custom-yolov4-detector.cfg"

results_path = "results"

cropped_img_path = "cropped_img"

img_results = "final_result_img"

exp2_objNames = "darknet_Yolov4_obj_names.names"

exp1_objNames = "obj.names"
classes = []
with open(exp2_objNames,"r") as f:
    classes = [line.strip() for line in f.readlines()]

# i=0
# for c in classes:
#     print(f"{i}.-{c},",end=" ")
#     i+=1

net = cv2.dnn.readNet(net_path,cfg_path)

net.getLayerNames()

net.getUnconnectedOutLayers()

layer_names = net.getLayerNames()
outputlayers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

colors= np.random.uniform(0,255,size=(len(classes),3))

def imShow(path):
    if type(path) == str():
        image = cv2.imread(path)
    else:
        image = path

    height, width = image.shape[:2]
    resized_image = cv2.resize(image,(3*width, 3*height), interpolation = cv2.INTER_CUBIC)

    fig = plt.gcf()
    fig.set_size_inches(18, 10)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
    plt.show()


def crop_img(input_image,img_save_path, img_name ,bb_cordinate):
    dh,dw,cha = input_image.shape

    for i in range(len(bb_cordinate)):
        if bb_cordinate[i] < 0:
            bb_cordinate[i] = abs(bb_cordinate[i])

    xmin,ymin, w, h = bb_cordinate

    p = 5

    x_MIN =0
    if xmin - p >= 1 :# xmin
        X_MIN = xmin - p
    else:
        X_MIN = xmin

  
    if ymin -p >=1  :
        Y_MIN =  ymin -p
    else:
        Y_MIN = ymin


    if (ymin + h) + p > dh:
        Y_MAX = (ymin + h)
    else: 
        Y_MAX = (ymin + h) + p

    if (xmin+ w) + p > dw:
        X_MAX = (xmin+ w)
    else: 
        X_MAX = (xmin+ w) + p

    #print()       above+5    bottom+5    left+5    right+5
    #imgCrop = img1[ ymin:   ymin + h  ,  xmin:    xmin+ w  ]

    imgCrop = input_image[ Y_MIN : Y_MAX  , X_MIN : X_MAX ]

    # store the every cropped img data into dictionary
    cropped_img_dict = {}
    cropped_img_dict[ img_name ] = imgCrop

    #print(imgCrop)
    #cv2.imwrite( img_save_path, imgCrop)
    cv2.imwrite( os.path.join(img_save_path,img_name) , imgCrop)

    #return cropped_img_dict , bb_cordinate
    return img_name , imgCrop , bb_cordinate


# fun 2
def detecting_objects(img1):

    height,width,channels = img1.shape
    count = 0
    blob = cv2.dnn.blobFromImage(img1,0.00392,(416,416),(0,0,0),True,crop=False)

    net.setInput(blob)
    outs = net.forward(outputlayers)


    class_ids=[]
    confidences=[]
    boxes=[]
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]


            if confidence > 0.5:
                center_x= int(detection[0]*width)
                center_y= int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                #cv2.circle(img,(center_x,center_y),10,(0,255,0),2)
                #rectangle co-ordinaters
                x=int(center_x - w/2)
                y=int(center_y - h/2)
                #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

                boxes.append([x,y,w,h]) #put all rectangle areas
                confidences.append(float(confidence)) #how confidence was that object detected and show that percentage
                class_ids.append(class_id) #name of the object tha was detected
                #bounding_box_dict['class_ids_'+str(count)] = [center_x,center_y,w,h]
                count += 1

    # print(f"all objects Bounding Boxes,Confidences,class id's are found in : {end_time - start_time} Seconds")
    del blob, out
    gc.collect()
    return class_ids , confidences , boxes


import matplotlib.image as mpimg

def Segmentaion(cropped_img_name,cropped_img,img_cordinates):

    start_time = time.time()
    #original img
    #cv2_img=cv2.imread(cropped_img)
    cv2_img=cropped_img
    
    cv2_img_gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)

    adap_thresh = cv2.adaptiveThreshold(cv2_img_gray,100,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,15,2)

    contours,hierarchy = cv2.findContours(adap_thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    largest_areas = sorted(contours,key=cv2.contourArea)
    mask = np.zeros(cv2_img_gray.shape , np.uint8)
    img_contour = cv2.drawContours(mask,[largest_areas[-1]],0,(255,255,255,255),-1)

    #plt_img = mpimg.imread(cropped_img)
    plt_img = cropped_img

    img_bitcontour = cv2.bitwise_or(plt_img,plt_img,mask=mask)
    hsv_img = cv2.cvtColor(img_bitcontour,cv2.COLOR_BGR2HSV)
    mask_plate = cv2.inRange(hsv_img,np.array([0,0,50]),np.array([200,90,250]))

    mask_not_plate = cv2.bitwise_not(mask_plate)
    # final segmented img of object
    mask_fruit = cv2.bitwise_and(img_bitcontour,img_bitcontour,mask=mask_not_plate)

    #titles = ["original",'Output',"masked of final fruit"]
    #images = [cv2_img,mask_fruit_finger,mask_not_plate]

    rgb_img = mask_fruit.copy()
    img_gray2 = cv2.cvtColor(rgb_img,cv2.COLOR_BGR2GRAY)
    adap_thresh2 = cv2.adaptiveThreshold(img_gray2,100,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,15,2)
    cont2 ,_ = cv2.findContours(adap_thresh2,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    largest_areas2 = sorted(cont2,key=cv2.contourArea)
    print(f"total numbers of contour is: {len(largest_areas2)} for {cropped_img_name}")
    mask2 = np.zeros(img_gray2.shape , np.uint8)
    cop = mask2.copy()

    label_list = ["thumb" , "Apple", "Banana", "Orange", "Qiwi", "Tomato", "Carrot", "Onion" ]


    Thumb_img_min_rectangle = None
    if cropped_img_name.startswith(label_list[0]): # thumb
        req_contour = largest_areas2[-1]
        img_contour4 = cv2.drawContours(cop,[req_contour],0,(255,255,255,255),-1)
        req_object_area = cv2.contourArea(req_contour)

        min_rectangle = img_contour4.copy()
        rect = cv2.minAreaRect(req_contour)
        box = cv2.boxPoints(rect)
        box2 = box.copy()
        box2 = np.int0(box)

        pix_height = max(rect[1])
        pix_to_cm_multiplier = 5.0/pix_height

        min_rectangle = img_contour4.copy()
        copy_original = cv2_img.copy()
        Thumb_img_min_rectangle = cv2.drawContours(copy_original , [box2],0 ,(0,255,0), 2)

        print(f"\nrectangle is: {rect} \nbox points are: {box} \n pixel hight: {pix_height} \npixel multiplier is : {pix_to_cm_multiplier}")

        # result = [req_object_area , pix_to_cm_multiplier ]
        result = [req_contour , req_object_area , pix_to_cm_multiplier]

    elif cropped_img_name.startswith(label_list[6]): # carrot
        req_contour = largest_areas2[-2]
        img_contour4 = cv2.drawContours(cop,[req_contour],0,(255,255,255,255),-1)
        req_object_area = cv2.contourArea(req_contour)
        # result = [req_object_area]
        result = [req_contour , req_object_area ]

    else: 
        req_contour = largest_areas2[-1]
        img_contour4 = cv2.drawContours(cop,[req_contour],0,(255,255,255,255),-1)
        req_object_area = cv2.contourArea(req_contour)
        # result = [req_object_area]
        result = [req_contour , req_object_area ]

    print(f"\nthe largest_fruit_contour is : {req_object_area} which is of {cropped_img_name}")

    end_time = time.time()
    print(f"segmentation of an object is done in : {end_time - start_time} Seconds")


    img_data_list = [cv2_img, mask_fruit, img_contour,img_contour4]
    if Thumb_img_min_rectangle is not None:
        img_data_list.append(Thumb_img_min_rectangle)
    return  result , img_cordinates , img_data_list



density_dict = { "Apple":0.96, "Banana":0.94,  "Carrot":0.641,"Onion":0.513, "Orange":0.482,"Tomato":0.481 , "Qiwi" :0.575 }

calorie_dict = { "Apple":52, "Banana":80,  "Carrot":41 , "Onion":40 , "Orange":47 , "Tomato":18 , "Qiwi":44 }


skin_multiplier = 5*2.3

label_list = ["thumb" , "Apple", "Banana", "Orange", "Qiwi", "Tomato", "Carrot", "Onion" ]



def getCalorie(label, volume): #volume in cm^3
    calorie = calorie_dict[label]
    density = density_dict[label]
    mass = volume*density*1.0
    calorie_tot = (calorie/100.0)*mass
    return mass, calorie_tot, calorie #calorie per 100 grams


def getVolume(label, area, skin_area, pix_to_cm_multiplier, fruit_contour):
    area_fruit = (area/skin_area)*skin_multiplier
    volume = 100

    if label == label_list[1] or label == label_list[3] or label == label_list[4] or label == label_list[5] or label == label_list[7] : #sphere-apple,orange,kiwi,tomato,onion
        radius = np.sqrt(area_fruit/np.pi)
        volume = (4/3)*np.pi*radius*radius*radius
        #print (area_fruit, radius, volume, skin_area)

    if label == label_list[2] or label == label_list[6] and area_fruit > 30 :
        fruit_rect = cv2.minAreaRect(fruit_contour)
        height = max(fruit_rect[1])*pix_to_cm_multiplier
        radius = area_fruit/(2.0*height)
        volume = np.pi*radius*radius*height

    if (label==4 and area_fruit < 30) :
        volume = area_fruit*0.5

    return volume

def calories(segmentation_to_calorie):
    # segmentation_to_calorie == dict()
    # key =  obj name + confidence+_bbNum + .jpg
    # value = [ [ thumb/fruit Contour ,/, thumb/fruit Contour-Area , pix_cm ] , img_cordinates  ]

    # 1.-Apple, 2.-Banana, 3.-Carrot, 4.-Onion, 5.-Orange, 6.-Qiwi, 7.-Tomato, 8.-thumb,

    start_time = time.time()
    fruit_calories_dict = {}
    calo_result = " "

    skin_contour_Area = pix_cm = None

    for k in segmentation_to_calorie:
        if k.startswith("thumb"):
            skin_contour = segmentation_to_calorie[k][0][0]
            skin_contour_Area = segmentation_to_calorie[k][0][1]
            pix_cm = segmentation_to_calorie[k][0][2]

            img_cordinates = segmentation_to_calorie[k][1]
            # 'Carrot(95.66)_10.jpg'
            fruit_calories_dict[k] = [ k.split("_")[0]  , img_cordinates]

    if skin_contour_Area is None and  pix_cm is None:
        print("++++++++++++|||-------------*************** IMP NOTE ***************-------------|||++++++++++++")
        print("\n********------!***!------********\nReferance Object is not present in image please provide one to move ahead for calories\n********------!***!------********\n")
        print("++++++++++++|||-------------*************** IMP NOTE ***************-------------|||++++++++++++\n")

    for k in segmentation_to_calorie:
        if not k.startswith("thumb"):
            fruit_contour = segmentation_to_calorie[k][0][0]
            fruit_contour_Area = segmentation_to_calorie[k][0][1]

            # key =  objName(confidence)_bbNum.jpg
            # key = 'Carrot(95.66)_10.jpg'
            name = k.split("_")[0].split("(")[0]
            objName_Confidence =  k.split("_")[0]
            bbNum = k.split("_")[1][:-4]
            img_cordinates = segmentation_to_calorie[k][1]

            if skin_contour_Area is not None  and pix_cm is not None:
                volume = getVolume(name, fruit_contour_Area, skin_contour_Area, pix_cm, fruit_contour)
                mass, cal, cal_100 = getCalorie(name, volume)
                fruit_volumes=volume
                fruit_calories_100grams=cal_100
                fruit_mass=mass
                fruit_calories = cal

                #print(f"\nfruit_volumes {fruit_volumes},\nfruit_calories {fruit_calories},fruit_calories_100grams {fruit_calories_100grams}, \nfruit_mass {fruit_mass}")
                final_result_name = str(objName_Confidence) +"_"+ str(round(fruit_calories))+"kcal" +"_"+str(bbNum)
                print(final_result_name , img_cordinates )
                #fruit_calories_dict[k] = [ final_result_name , img_cordinates ]
                print(f"Calorie estimation done for--> {name} and Calorie is {round(fruit_calories,3)} kcal/g \n")
                calo_result += f"Calorie estimation done for--> {objName_Confidence} and Calorie is {round(fruit_calories,3)} kcal/g \n"
            else:
                final_result_name = str(objName_Confidence) +"_"+str(bbNum)

            fruit_calories_dict[k] = [ final_result_name , img_cordinates ]

    end_time = time.time()
    print(f"all object Calories are found in : {end_time - start_time} Seconds")

    return fruit_calories_dict, calo_result


def find_required_detections(input_image , class_ids , confidences , boxes):
    global drawn
    global required_bb_detections , whole_img_name #, history_cropped_images  #, history_final_result_img
    required_bb_detections  = {}
    history_cropped_images = {}

    #history_final_result_img = []

    start_time = time.time()

    indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.4,0.6)

    print(f"the required bb's indexes are: \n",indexes)

    '''Now using below loop over all found boxes,
    if box is appearing in indexes then only draw rectangle, color it,
    put text of class name on it.'''
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i  in indexes:
            x,y,w,h = boxes[i]
            label = str(classes[class_ids[i]])

            # whole_img_name = f"{images_list[ask_img][:-4]}_{label}({round(confidences[i]*100 ,2)})_{i}.jpg"
            whole_img_name = f"{lund_img[:-4]}_{label}({round(confidences[i]*100 ,2)})_{i}.jpg"

            img_name = f"{label}({round(confidences[i]*100 ,2)})_{i}.jpg"
            print(f"\nthe the respective lables are--> BB_No.{i} = {label}")

            required_bb_detections[img_name] = boxes[i]
    end_time = time.time()
    print(f"all required object are found in : {end_time - start_time} Seconds")

    # --------------------------*****************************--------------------------

    cropped_img_dict = {}
    segmented_img_dict = {}
    segmentation_to_calorie = {}
    fruit_calories_dict = {}
    print("\n***********------ images processing part 2 started ------***********\n")
    # start time
    start_time = time.time()
    for k in required_bb_detections:

        """
        it returns ==> cropped_img_dict
        - cropped img dict --> key= "img name" = label + confidence + requireed NBN bb number" ,
                            value = cropped img Array data --> img data
        - the bb cordinates of obj = refined bounding box(bb) cordinates
        """
        cropped_img_name ,cropped_img ,refined_obj_bb_box  = crop_img(input_image, cropped_img_path, k ,required_bb_detections[k] )
        cropped_img_dict[cropped_img_name] = cropped_img
        # history_cropped_images[images_list[ask_img][:-4]] = cropped_img_dict # global
        history_cropped_images[lund_img[:-4]] = cropped_img_dict # global
        print(f"\ncropping done for--> {cropped_img_name}")

        """
        returns ==> segmented_img_dict [img name] = [ original img , segmented img , contour drawn img]
        - key = name of cropped img with its confidence
        - value = segemted object img data
        """
        segmented_obj_contour_area_pixel, img_cordinates , segmented_images_data  = Segmentaion(cropped_img_name,cropped_img,required_bb_detections[k])
        segmented_img_dict[cropped_img_name] = segmented_images_data

        segmentation_to_calorie[cropped_img_name] = [segmented_obj_contour_area_pixel , img_cordinates]

        print(f"Segmentaion done for--> {cropped_img_name} and Area is {segmented_obj_contour_area_pixel[1]}")

    print("\n------------------***------------------\n")

    """
        returns the calories of each segmented food object through calories function
        returns: dict --> fruit_calories_dict gives==> final_data
        - key = img name+confidence+bbNum.jpg ==> objName(confidence)_bbNum.jpg
        - value = [ img name+confidence+calories+bbNum , bounding box cordinates ]
    """
    final_data, final_calo = calories(segmentation_to_calorie)
    # end time
    end_time = time.time()
    print(f"image processing part 2 completed in : {end_time - start_time} Seconds")




    print("\n***********------ images Display part started ------***********\n")
    start_time = time.time()
    print(final_data,f"\nlength of dictionary is: {len(final_data)}")

    result_img = input_image.copy()
    print("\n copy img before detection is---:\n")





    for k in final_data:
        doc_string = """
          dictionary = { "thumb / fruit" : [ objectName , [bounding box]  ] ,
                         "apple"         : [ apple(89.34)_45Kcal_5 , [372, 33, 170, 295] ],
                          ....
                        }
        """

        final_img_name, BB_Cordinates = final_data[k][0] , final_data[k][1]
        font = cv2.FONT_HERSHEY_PLAIN
        x,y,w,h = BB_Cordinates

        color = colors[2]
        cv2.rectangle(result_img,(x,y),(x+w,y+h),color,2)

        cv2.putText(result_img,final_img_name,(x,y+30),font,1,(255,255,255),2)
        print(f"Drawn part done for--> {final_img_name} ")

    end_time = time.time()
    print(f"display part done in {end_time - start_time} Seconds")
    print("\n copy img After detection is---:\n")



    # imShow(result_img)

    # if choice >=3:
    #     drawn = cv2.imwrite(os.path.join(img_results, "result_Original_"+lund_img) , result_img)
    # else:
    #     drawn = cv2.imwrite(os.path.join(img_results, "result_"+lund_img) , result_img)

    print("\n***********------ images Display part ended ------***********\n")




    return result_img , history_cropped_images , segmented_img_dict , segmentation_to_calorie , final_data, final_calo





UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        result_img = process_image(filepath)
        
        # return send_file(result_img, mimetype='image/jpeg')

        img_bytes = cv2.imencode('.jpg', result_img)[1].tobytes()
        return Response(img_bytes, mimetype='image/jpeg')

def process_image(filepath):
    img1 = cv2.imread(filepath)

    img1 = cv2.resize(img1,(416,416))


    
    height,width,channels = img1.shape
    print(height,width,channels)
    #--------------------------------------------------------------
    class_ids , confidences , boxes  =  detecting_objects(img1) # fun 2
    print("--------------------------------------------------------")

    if class_ids == []:
        print(" clould not be able to detect object inside image..!!")

    else:
        for i in range(len(class_ids)):
            print(f"No.:{i}-> {class_ids[i]} : {classes[class_ids[i]]} -> {round(confidences[i]*100 ,2)}%")

    print("--------------------------------------------------------")
    #c1+=1

    result_img , history_cropped_images , segmented_img_dict , segmentation_to_calorie , final_data, final_calo = find_required_detections(img1 , class_ids , confidences , boxes)
    
    return result_img

if __name__ == '__main__':
    app.run()
