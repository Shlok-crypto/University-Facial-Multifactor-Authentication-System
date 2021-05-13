# Access_Granter
I have created a GUI Bassed **Contactless Access Control, *Facial Authentication System***.

**Input required:** 
* **Indivuatials RFID tag** (infromation)
	* Name 
	* Identifation Number
	* Unique Id Number
	* Other information
*	**Live Video**
	* Face Recognition 
	* Face Authentication

**Output:**
* **Confidence Level**
	* **Access Granted/Denied**

# **How the Software Works :)**

* ***Information Gathering =>***

	The software analyzes Live Video in **Real Time!!**
	The Person of Intrest(POI), Name, and Identification Number are harvested from the RFID Tag information.

	live video captures the image of the POI.
	Utilizing *HaarCascades*(Facial Detection), The software detects faces in the image.

	The faces are extracted as ROI(Region of interest). 
	The ROI is processed via **Local Binary Pattern Histogram(LBPH)**. Then the ROI(LBPH) is passed onto the Authentication Algrothiom.

* ***Authentifation Process=>***

	The Authentication system analysis the input ROI(LBPH) against a set of prerecorded  LBPH data of the POI. The Software outputs a confidence level. 
	The Threshold value of the confidence level determines Access Granted or Access Denied

# **Gui Based Admin Control**

The software is GUI-based, allowing a user-friendly experience.

* **Admin Controll**
	* Add New User Data 
	*	Delete User Data
	* Train New Data	
