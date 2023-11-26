#------------------------------------------------- importing necessary packages ---------------------------------------------------------#
from io import BytesIO
import cv2
import easyocr
import numpy as np
import re
from validate_email_address import validate_email
from PIL import Image
import pandas as pd
import mysql.connector
from sqlalchemy import create_engine
import pymongo
import streamlit as st
import streamlit_authenticator as sta

#----------------------------- block defining a function to create a user for authentication ---------------------------------------------------#
def signup():
    with st.form(key='signup', clear_on_submit=True):
        st.title("Sign Up")
        username=st.text_input("UserName",placeholder="Enter UserName")
        Name = st.text_input("Name", placeholder="Enter Your Name")
        Pw = st.text_input("Password", placeholder="Enter Your password",type='password')
        mongo = pymongo.MongoClient('mongodb://localhost:27017/')
        mydb = mongo["INFO"]
        mycollection = mydb['cred']
        if st.form_submit_button("Create account"):
            cred = {'User_id': username,
                    'Name': Name,
                    'Password': Pw}
            mycollection.insert_one(cred)
            st.success("Account has been successfully created!\n\n Try login with your credentials")
            st.balloons()


#------------------------------------ block of code to define function that recoginze text from image ----------------------------------------#
def data():
    #------------ defining a function to crop the image by two halves and extracting the text using cv2 if there is no text found in the centre of card -----------#
    def extract_text_and_crop_halves(upload_image, language='en'):
        # Convert the image to grayscale
        gray = cv2.cvtColor(upload_image, cv2.COLOR_BGR2GRAY)
        # Use EasyOCR to extract text regions
        reader = easyocr.Reader([language])
        results = reader.readtext(upload_image)
        # Extract bounding boxes from the OCR results
        bounding_boxes = [result[0] for result in results]
        # Create a mask for the text regions
        mask = np.zeros_like(gray)
        for box in bounding_boxes:
            pts = np.array(box, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], 255)
        # Invert the mask to get the region without text
        inverted_mask = cv2.bitwise_not(mask)
        # Find contours in the inverted mask
        contours, _ = cv2.findContours(inverted_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Assume the business card is a rectangle and find its contours
        card_contours = [contour for contour in contours if len(contour) == 4]
        if card_contours:
            # Assume the first contour is the business card
            card_contour = card_contours[0]
            # Sort the vertices of the card contour
            card_contour = cv2.convexHull(card_contour)
            sorted_card_contour = np.array(sorted(card_contour, key=lambda x: x[0][0]))
            # Split the card into left and right halves
            mid_x = (sorted_card_contour[0][0][0] + (sorted_card_contour[2][0][0]) // 2) - 35
            left_half = upload_image[:, :mid_x, :]
            right_half = upload_image[:, mid_x:, :]
            return left_half, right_half
        else:
            return None, None

    # ------------ defining a function to crop the image by two halves and extracting the text using cv2 if there is text found in the centre of card -----------#
    def crop_left_and_right(upload_image):
        # Get the width and height of the image
        height, width, _ = upload_image.shape
        # Define the coordinates for the left and right halves
        left_x, left_y = 0, 0
        right_x, right_y = (width // 2) - 35, height
        # Crop the left and right halves
        left_half = upload_image[left_y:right_y, left_x:right_x]
        right_half = upload_image[left_y:right_y, right_x:width]
        return left_half, right_half

    #line of code to fetch and convert the image as numpy array
    uploaded_image = Image.open(file)
    image_np = np.array(uploaded_image)
    left_half1, right_half1 = extract_text_and_crop_halves(image_np)

    # Display the original image, left half, and right half
    if left_half1 is not None:
        left_half = left_half1
    if right_half1 is not None:
        right_half = right_half1
    #line of code to process if the centre of the image contains text which affects during crop
    if left_half1 is None:
        left_half1, right_half1 = crop_left_and_right(image_np)
        left_half = left_half1
        right_half = right_half1

    # ------------ defining a function to recognize and extract the text from cropped images using easyocr -------------#
    def extract_text_with_easyocr(upload_image, language='en'):
        # Create an EasyOCR reader with the specified language
        reader = easyocr.Reader([language])
        # Read the text from the image
        result = reader.readtext(upload_image)
        # Extract and concatenate the recognized text
        extracted_text = '\n'.join([entry[1] for entry in result])
        # extracted_text=result
        return extracted_text
    #line of code to call the function to extract the text
    extracted_information_right = extract_text_with_easyocr(right_half)
    extracted_information_left = extract_text_with_easyocr(left_half)

    # ------------ defining a function to recognize and store the contact number from extracted text using regular expression -------------#
    def extract_contact_number(text):
        # Define a regular expression pattern for recognizing phone numbers
        phone_number_pattern = re.compile(r'''(?:\+\d{1,2}\s?)?(?:\(\d{1,3}\))?\s?\d{2,3}[\s.-]\d{2,3}[\s.-]\d{2,4}''',re.VERBOSE)
        # Search for phone numbers in the text
        contact_numbers = phone_number_pattern.findall(text)
        #storing the data into a list
        if contact_numbers:
            valid_contact_numbers = [number for number in contact_numbers]
            return valid_contact_numbers
        else:
            return None

    # ------------ defining a function to recognize and store the email from extracted text using email validate library -------------#
    def extract_email(text):
        lines = text.splitlines()
        for i in range(len(lines)):
            if validate_email(text.splitlines()[i]):
                return text.splitlines()[i]

    # ------------ defining a function to recognize and store the website from extracted text using regular expression -------------#
    def extract_urls(text):
        # Define a regex pattern for recognizing URLs
        url_pattern = re.compile(r'\b(?:https?://)?(?:www|WWW|wwW\.)?[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}(?:/[A-Za-z0-9.-]*)*\b')
        # Find all matches for URLs
        urls = url_pattern.findall(text)
        if len(urls) > 1:
            return f'www.{urls[0]}'
        else:
            return f'www.{urls}'

    # ------------ defining a function to recognize and store the address from extracted text using regular expression -------------#
    def add(text):
        # Define a regex pattern for recognizing addresses
        address_pattern = re.compile(r'\b\d{3}\s[A-Za-z0-9.,\s]+\b[,.\s]*[A-Za-z0-9.,\s]+[,.\s]*[A-Za-z0-9.,\s]+;?[,.\s]*[A-Za-z]+;?[,.\s]*\d{6,7}\b')
        # Find all matches for addresses
        address = address_pattern.findall(text)
        return address

    # ------------ defining a function to recognize and store the area,district,state and pincode from extracted address data using regular expression -------------#
    def address(address):
        pattern = re.compile(r'(\d+\s+[A-Za-z]+\s)?\W*([A-Za-z]+)?\W*([A-Za-z]+)?\W*(\d{6,7})?')
        # Extract components using regex pattern
        match = pattern.search(address)
        # Get matched values or None if not found
        area, district, state, pincode = match.groups(default=None)
        if state == None or pincode == None:
            pattern = re.compile(r'(\d+\s+[A-Za-z]+\s+[A-Za-z]+)?\W*([A-Za-z]+)?\W*([A-Za-z]+)?\W*(\d{6,7})?')
            match = pattern.search(address)
            area, district, state, pincode = match.groups(default=None)
        return area, district, state, pincode

    # ------------ defining a function to convert the image into binary data to store in sql using cv2 library -------------#
    def image_loader(file):
        _, image_binary = cv2.imencode('.png', file)
        image_binary = image_binary.tobytes()
        return image_binary

    #block of code works if the company logo is in LHS and card holders data in RHS
    if len(extracted_information_right) > len(extracted_information_left):
        info = add(extracted_information_right)
        area, district, state, pincode = address(info[0])
        # creating a dictionary for streamlit table without image data
        Biz_data1 = {"Organization": [extracted_information_left],
                     "Name": [extracted_information_right.splitlines()[0]],
                     'Designation': [extracted_information_right.splitlines()[1]],
                     'Contact Number': [extract_contact_number(extracted_information_right)[0]],
                     'E-Mail': [extract_email(extracted_information_right)],
                     'Website': [extract_urls(extracted_information_right)],
                     'Area': [area],
                     'District': [district],
                     'State': [state],
                     'Pincode': [pincode]}
        #creating a dictionary for sql insertion with image data
        Biz_data2 = {"image":[image_loader(image_np)],
                    "Organization": [extracted_information_left],
                    "Name": [extracted_information_right.splitlines()[0]],
                    'Designation': [extracted_information_right.splitlines()[1]],
                    'Contact Number': [extract_contact_number(extracted_information_right)[0]],
                    'E-Mail': [extract_email(extracted_information_right)],
                    'Website': [extract_urls(extracted_information_right)],
                    'Area': [area],
                    'District': [district],
                    'State': [state],
                    'Pincode': [pincode]}
        df2 = pd.DataFrame(Biz_data2)
        df1 = pd.DataFrame.from_dict(Biz_data1, orient='index', columns=['INFO'])
        return df1, df2

    # block of code works if the company logo is in RHS and card holders data in LHS
    else:
        info = add(extracted_information_left)
        area, district, state, pincode = address(info[0])
        # creating a dictionary for streamlit table without image data
        Biz_data1 = {"Organization": [extracted_information_right],
                    "Name": [extracted_information_left.splitlines()[0]],
                    'Designation': [extracted_information_left.splitlines()[1]],
                    'Contact Number': [extract_contact_number(extracted_information_left)[0]],
                    'E-Mail': [extract_email(extracted_information_left)],
                    'Website': [extract_urls(extracted_information_left)],
                    'Area': [area],
                    'District': [district],
                    'State': [state],
                    'Pincode': [pincode]}
        # creating a dictionary for sql insertion with image data
        Biz_data2 = {"image":[image_loader(image_np)],
                     "Organization": [extracted_information_right],
                     "Name": [extracted_information_left.splitlines()[0]],
                     'Designation': [extracted_information_left.splitlines()[1]],
                     'Contact Number': [extract_contact_number(extracted_information_left)[0]],
                     'E-Mail': [extract_email(extracted_information_left)],
                     'Website': [extract_urls(extracted_information_left)],
                     'Area': [area],
                     'District': [district],
                     'State': [state],
                     'Pincode': [pincode]}
        df2 = pd.DataFrame(Biz_data2)
        df1 = pd.DataFrame.from_dict(Biz_data1, orient='index', columns=['INFO'])
        return df1, df2

#--------------------------------------------------------- Construction of Streamlit dashboard ---------------------------------------------------------------------#

st.set_page_config(page_title="OCR", page_icon=':newspaper:', layout='wide')

#establishing connection for mongodb to store user credentials
mongo = pymongo.MongoClient('mongodb://localhost:27017/')
mydb = mongo["INFO"]
mycollection = mydb['cred']
dbcursor = mycollection.find()
logindata = list(dbcursor)

#converting data into lists
userids = [i['User_id'] for i in logindata]
names = [logindata[j]['Name'] for j in range(len(logindata))]
pw = [logindata[k]['Password'] for k in range(len(logindata))]

#converting password into hashed password
hashed_passwords = sta.Hasher(pw).generate()

#dumping username,userid and password into credentials
credentials = {'usernames': {}}
for index in range(len(logindata)):
    credentials['usernames'][userids[index]] = {'name': userids[index], 'password': hashed_passwords[index]}

#Authenticating to login after matching the credentials with database
authenticator = sta.Authenticate(credentials, cookie_name='bizcard', key='nun', cookie_expiry_days=30)
name, authentication_status, userid = authenticator.login('Login', 'main')
st.session_state['authentication_status']=authentication_status
st.session_state['Authenticator']=authenticator

#if login credentials doesn't match with database
if st.session_state.authentication_status==False:
    st.error("Check your Login credentials,it seems to be invalid")

#if login credentials haven't entered
if st.session_state.authentication_status==None:
    st.warning("Enter Login credentials")
    st.info("If you don't have an Existing account,Kindly Signup")
    if st.checkbox("Sign up"):
        signup()

#if authentication is sucess
if st.session_state.authentication_status==True:
    st.sidebar.empty()
    nav = st.sidebar.radio("Navigation panel", ['HOME', 'About Page'])
    st.session_state.Authenticator.logout('Logout','sidebar')
    #Main page for all functions
    if nav=='HOME':
        st.title("	:frame_with_picture: BizCardX: Extracting Business Card Data with OCR")
        st.subheader("_Help yourself with the dropdown options_")
        file=st.file_uploader("Upload Biz card",type=['png','jpg'])
        show_file=st.empty()
        if not file:#if no file is detected in uploader
            st.info("Please upload a file with Data Type : PNG , JPG & JPEG ")
        #process if file is detected in uploader
        if isinstance(file,BytesIO):
            show_file.image(file)
            if st.button("Convert into text"):#returns the table of data without image
                data1,data2=data()
                st.table(data1)
            #let the user to process desired actions
            sb1 = st.selectbox("SQL Options", ['Read Data from SQL', 'Export the Data to SQL', 'Delete Data from SQL'])
            #established connection to mysql
            mycon = mysql.connector.connect(host="127.0.0.1", user="root", password="12345")
            mycursor = mycon.cursor()
            mycursor.execute(f"CREATE DATABASE IF NOT EXISTS bizcard;")
            #process only if the user wish to extract the data from image and save the same to mysql database only when the image is uploaded
            if sb1=='Export the Data to SQL' and st.button("Export"):
                mycursor.execute("CREATE TABLE IF NOT EXISTS bizcard.data (id INT AUTO_INCREMENT PRIMARY KEY,image LONGBLOB,Organization VARCHAR(255),Name VARCHAR(255),Designation VARCHAR(255),`Contact Number` VARCHAR(255),`E-Mail` VARCHAR(255),Website VARCHAR(255),Area VARCHAR(255),District VARCHAR(255),State VARCHAR(255),Pincode VARCHAR(255));")
                engine = create_engine('mysql+mysqlconnector://root:12345@localhost/bizcard')
                data1, data2 = data()
                data2.to_sql('data', con=engine, if_exists='append', index=False)
                engine.dispose()
                st.success("Successfully Exported!")
                st.balloons()
            #process only if the user wish to fetch the data in table without image from mysql database
            if sb1=='Read Data from SQL' and st.button("Fetch"):
                engine = create_engine('mysql+mysqlconnector://root:12345@localhost/bizcard')
                table_name = 'data'
                df = pd.read_sql_table(table_name, con=engine)
                df = df.drop(columns=['image'])
                st.table(df)
            #process only if the user wish to fetch the image of the concered id from mysql database
            if sb1 == 'Read Data from SQL':
                if st.checkbox("Show business card image"):
                    showq = f"SELECT DISTINCT id FROM bizcard.data"
                    show_df = pd.read_sql(showq, con=mycon)
                    show_list = show_df['id'].tolist()
                    sbb=st.selectbox("Select the id of business card image to be displayed",show_list)
                    if st.button("Fetch image") and sbb:
                        imgq=f"SELECT DISTINCT * FROM bizcard.data WHERE id={sbb}"
                        imgdf = pd.read_sql(imgq, con=mycon)
                        imglist = imgdf.values.tolist()
                        imgdata=imglist[0][1]
                        image_stream = BytesIO(imgdata)
                        image_open = Image.open(image_stream)
                        st.image(image_open)
            #process only if the user wish to delete the data with respect to the name of the card holder from mysql database
            if sb1 == 'Delete Data from SQL':
                delq = f"SELECT DISTINCT Name FROM bizcard.data"
                del_df = pd.read_sql(delq, con=mycon)
                del_list = del_df['Name'].tolist()
                sb2=st.selectbox("Select Name of Data to be deleted",del_list)
                if st.button("Delete") and sb2:
                    mycon = mysql.connector.connect(host="127.0.0.1", user="root", password="12345")
                    mycursor = mycon.cursor()
                    delete_query = f"DELETE FROM bizcard.data WHERE Name = '{sb2}' LIMIT 1"
                    mycursor.execute(delete_query)
                    mycon.commit()
                    st.write(sb2)
                    st.success("Deleted Successfully!")
                    st.balloons()
                    mycon.close()
        #process if file is not detected in uploader
        else:
            sb1 = st.selectbox("SQL Options", ['Read Data from SQL', 'Export the Data to SQL', 'Delete Data from SQL'])
            mycon = mysql.connector.connect(host="127.0.0.1", user="root", password="12345")
            mycursor = mycon.cursor()
            mycursor.execute(f"CREATE DATABASE IF NOT EXISTS bizcard;")
            #displlay error message if the user wish to extract the data from image and save the same to mysql database when the image is not uploaded
            if sb1 == 'Export the Data to SQL' and st.button("Export") and data:
                st.error("No INSTANCE found to export,Kindly upload a business card image to process export")
            # process only if the user wish to fetch the data in table without image from mysql database
            if sb1 == 'Read Data from SQL' and st.button("Fetch"):
                engine = create_engine('mysql+mysqlconnector://root:12345@localhost/bizcard')
                table_name = 'data'
                df = pd.read_sql_table(table_name, con=engine)
                df = df.drop(columns=['image'])
                st.table(df)
            # process only if the user wish to fetch the image of the concered id from mysql database
            if sb1 == 'Read Data from SQL':
                if st.checkbox("Show business card image"):
                    showq = f"SELECT DISTINCT id FROM bizcard.data"
                    show_df = pd.read_sql(showq, con=mycon)
                    show_list = show_df['id'].tolist()
                    sbb = st.selectbox("Select the id of business card image to be displayed", show_list)
                    if st.button("Fetch image") and sbb:
                        imgq = f"SELECT DISTINCT * FROM bizcard.data WHERE id={sbb}"
                        imgdf = pd.read_sql(imgq, con=mycon)
                        imglist = imgdf.values.tolist()
                        imgdata = imglist[0][1]
                        image_stream = BytesIO(imgdata)
                        image_open = Image.open(image_stream)
                        st.image(image_open)
            # process only if the user wish to delete the data with respect to the name of the card holder from mysql database
            if sb1 == 'Delete Data from SQL':
                delq = f"SELECT DISTINCT Name FROM bizcard.data"
                del_df = pd.read_sql(delq, con=mycon)
                del_list = del_df['Name'].tolist()
                sb2 = st.selectbox("Select Name of Data to be deleted", del_list)
                if st.button("Delete") and sb2:
                    mycon = mysql.connector.connect(host="127.0.0.1", user="root", password="12345")
                    mycursor = mycon.cursor()
                    delete_query = f"DELETE FROM bizcard.data WHERE Name = '{sb2}' LIMIT 1"
                    mycursor.execute(delete_query)
                    mycon.commit()
                    st.write(sb2)
                    st.success("Deleted Successfully!")
                    st.balloons()
                    mycon.close()
    #block of code to define the author data
    elif nav=='About Page':
        st.title(":dart: About Page")
        st.subheader("**Crafted by :**")
        st.markdown("#### _ROHITH VIGNESH CS_")
        st.markdown("### Linked In")
        st.info("https://www.linkedin.com/in/csrv547/")
        st.markdown("### GITHUB Profile")
        st.info("https://github.com/CSRV547")

