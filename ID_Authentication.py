import cv2 as cv
import os
import numpy as np
import pickle as pk
from tkinter import *

hard_cascade = cv.CascadeClassifier('Hard_Face.xml')

name = input("Enter Name")
id = input("Enter id")

yml_dict ={}

# loading Yml_dict from file
with open("YML_dict_FILE","rb") as f:
    yml_dict=pk.load(f)
    print("Data Stored",yml_dict)


#TAKING IMAGE GETTING THE ROI AND SAVING IT ONTO A FOLDER
# Returns name,id,Parent_dir
def facestorage():
    capture = cv.VideoCapture(0)

#making individual folder for every person
    tempDir = name
    parent_Dir = "C:/Users/Linh/Desktop/"

    dir = os.path.join(parent_Dir, tempDir)
    try:
        os.mkdir(dir)
        print("Folder Created: ",dir)
    except:
        print(OSError)

    counter=0
    while True:
        _, frame = capture.read()

        cv.imshow('video',frame)
        # Convert image into grayScale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Detect face
        faces_rect = hard_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7)
        print("face:",len(faces_rect))
        counter+=1

        # Get the ROI of the faces
        for (x, y, w, h) in faces_rect:
            # Saving the ROI images(face) onto the folder
            #dir = r"C:/Users/Linh/Desktop/shlk/"
            a=cv.imwrite(dir+"/" + name + '.' + id + '.'+ str(counter)+ ".jpg", gray[y:y + h, x:x + w] )
            if a == False:
                print("Photos not being saved\nCheck folder Path")
                break
            cv.imshow("face", gray[y:y + h, x:x + w])
        if counter > 60:
            print("limit Reached\n~ Aprox ~ Number of Photos saves",counter)
            break
        elif cv.waitKey(100) == ord('q'):
            break

    capture.release()
    cv.destroyAllWindows()
    return name,id,parent_Dir


#Traning on the DATA(ROI IAMGES)
def faceTrainerID():

    # if update_data != None:
    #     print("tyeee",type(update_data))
    #     name = update_data
    #     DIR = "C:/Users/Linh/Desktop/"
    #
    # else:
    #     name, id, DIR = facestorage()


    name, id, DIR = facestorage()

    #people = ['Ansh','Biden','MakedPerson','Manjusha','Manmohan','modi','Obama','shlok','TomCruise']
    people = [name]
    #DIR = r'C:\Users\Linh\Documents\Opencv\FaceRec'

    features = []
    labels =[]

    def create_train():
    # folder path(DIR/path)
        for person in people:

            print(DIR+person)

            path = os.path.join(DIR,person)
            label = people.index(person)
            print("dir", path)
        # image path(DIR/path/image)
            for image in os.listdir(path):
                image_path = os.path.join(path,image)

                # Readingin the images
                img = cv.imread(image_path)

            # length of the image Height and Width
                h = int(img.shape[0])
                w = int(img.shape[1])

            # Resize Large images to allow faster Processing
                if h > 255 and w > 255:
                # Resize EXTRA large IMAGES
                    if h > 3264 and w > 1836:
                        # Reduce by 50%
                        img = cv.resize(img, (0, 0), fx=0.50, fy=0.50)

                    # Recude by 22%
                    img = cv.resize(img, (0, 0), fx=0.22, fy=0.22)

            # Convert image into GrayScale(optional)
                gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
                print('path:', image_path)

            # FaceDetection
                faces_ret = hard_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=4)
                for (x,y,w,h) in faces_ret:
                    # crop the image(face)
                    faces_roi = gray[y:y+h,x:x+w]
                    features.append(faces_roi)
                    labels.append(label)
                # make changes hereeeeeeeeeeee
                # @##$#$##%#%%#%
    create_train()

    print("Training DONE ------------------------")
    print('features',len(features))
    print('labes',len(labels))
    features = np.array(features,dtype='object')
    labels = np.array(labels)

    face_recognizer = cv.face.LBPHFaceRecognizer_create()

    #train the recognizer on the features list and label list
    face_recognizer.train(features,labels)

    file_name = 'face_trainned'+name+'.yml'
    print(file_name)
# Saving yml file
    face_recognizer.save(file_name)

#Addind yml file into dict(key=name & value=file_name)
    yml_dict[name]=file_name

# Saving yml_dict on a local file
    v = open('YML_dict_FILE', 'wb')
    pk.dump(yml_dict, v)
    v.close()

    np.save('features.npy',features)
    np.save('labes.npy',labels)

    cv.waitKey(0)

#Authenticating the subject(face) against the face on record.
def faceRecVideoID():
    #hard_cascade = cv.CascadeClassifier('Hard_Face.xml')
#make changes hereeeeeeeeeeee
    #@##$#$##%#%%#%
    #people = ['ansh','Biden','MakedPerson','Manjusha','Manmohan','Modi','Obama','shlok','TomCruise']
    people = [name]

    face_recognizer = cv.face.LBPHFaceRecognizer_create()
    #face_recognizer.read('face_trainned.yml')
    face_recognizer.read('face_trainned'+name+'.yml')

    capture = cv.VideoCapture(0)
    _, frame = capture.read()
#while True:
    #_, frame = capture.read()

    #flip = cv.flip(frame, 1)

    # Resize Frame for faster Processing
    Small_frame = cv.resize(frame,(0,0),fx=0.50,fy=0.50)

    # gray Scale Video
    gray = cv.cvtColor(Small_frame, cv.COLOR_BGR2GRAY)

    # edge detections
    # passing GrayScale Video(gray)
    edges = cv.Canny(gray, 125, 175)
   # cv.imshow("Edges", edges)

    # detect the face in the input image
    faces_rect = hard_cascade.detectMultiScale(gray, 1.1, minNeighbors=2)

    # Region of intest on input image
    for (x, y, w, h) in faces_rect:
        g = int((y+h)/1)
        face_roi = gray[y:g, x:x + w]
       # cv.imshow('Face roi',face_roi)

        # Analizing the RIO image for Face Recognition
        label, confidence = face_recognizer.predict(face_roi)

    # reduce false classification,Unknow person
        if confidence < 68:
            print(f"Person = {name} , Because the confidence is {confidence}")
        # Draw  Box around the Unknown  Face
            cv.putText(gray, name, (x, y), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), thickness=2, )
            cv.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
            print(f"{name} is Authenticated\nAccess Granted")
        else:
            print(f"{name} Can't be Authenticated{confidence}\nAccess Denied")
###############################################################################################################################
# final video Feed
        cv.imshow('Detected Face', gray)
        # Break out of the loop

    if cv.waitKey(20) == ord('q'):
        exit()

    capture.release()
    cv.destroyAllWindows()


def admin_control():
    Admin = input("\n\nAdmin username:==> ")
    password = input("Admin Password:==> ")

#Contains Admin name & password
    Adminstrator = {"admin":"admin","shlok":"w71acap9"}

# Authenticating Admin name & password
    if Admin in Adminstrator:

        if Adminstrator[Admin]==password:
            print(f"Access Granted\n\nWellcome {Admin.upper()} Admin")
            remove_OR_Update = input("Enter choice: \nRemove(r) Update(u):\n")

        # remove entry from yml_dict
            if remove_OR_Update == "r":
                update_name = input("\n\nEnter name to delete data: ")
            # if name in yml_dict delete it
                if update_name in yml_dict:
                    del yml_dict[update_name]
                    print(f"\n\n\n*{update_name}* Deleted Successfully\n", yml_dict)

                # Saving yml_dict on a local file
                    v = open('YML_dict_FILE', 'wb')
                    pk.dump(yml_dict, v)
                    v.close()
                else:print(f"\n\n\n*{update_name}* not present in yml_dict")

        #Updata Yml File
            elif remove_OR_Update == "u":
                update_name = input("\n\nEnter name to update data: ")
                # if name in yml_dict delete it
                if update_name in yml_dict:
                    name=update_name
                    facestorage()
                    print(f"\n\n\n*{update_name}* YML FILE Updated Successfully\n")
                else:
                    print(f"\n\n\n*{update_name}* YML FILE Not Present in System")

        else:"WRONG Password or Username"

    else:"WORONG Usernmae or Password"



#############
#this is main calling window
def Authentication():
    root = Tk()
    root.geometry("340x220")
    root.config(bg='light blue')
    root.title('ID Authentication')

    Label(root, text='Welcome\nFace Recognition System', bg='light blue', font=('caliber', 15, 'bold')).grid(row=0,column=0,columnspan=10)
    #name = Entry(root, bg='pink', borderwidth='5',)

# Buttons
    s1 = Button(root, text='Train Face\n Data', font=("helvetica", 20), height=1, width=10, pady=10, bg="black",
                fg='green', command=faceTrainerID)
    s2 = Button(root, text='Authenticate\nPerson', font=("helvetica", 20), height=1, width=10, pady=10, bg="black",
                fg='green', command=faceRecVideoID)
    s3 = Button(root, text='Admin\nControl', font=("helvetica", 20), height=1, width=10, pady=10, bg="black",
                fg='green', command=admin_control)
# add to button

    s1.grid(row=3, column=0)
    s2.grid(row=3, column=1)
    s3.grid(row=4,column=0)


    root.mainloop()