import traceback
from pathlib import Path
from operator import itemgetter
import shlex

import os
import sys
import time
import cv2
import numpy as np
import subprocess
import requests
import json
import pydicom
from pydicom.dataset import Dataset

import pytesseract
from PIL import Image, ImageDraw, ImageFont
import pandas as pd

import socket
from  multiprocessing import Process, Queue

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import exists

from datetime import datetime
from datetime import timedelta, date

from kombu import Connection, Exchange, Queue, Producer
from kombu.pools import connections
import uuid
import threading
import math
import slidingwindow as sw

import mxnet as mx
import gluoncv as gcv

from mxnet import gluon, nd, image, autograd
from mxnet.gluon.data.vision import transforms
from gluoncv import utils
from gluoncv.model_zoo import get_model
from gluoncv.utils import viz
from matplotlib import pyplot as plt

import redis

NODE_LIST = []
NODE_LIST.append('localhost')

### uncomment and edit following parts if clustering is needed
#NODE_LIST.append('160.58.150.64')
#NODE_LIST.append('160.8.12.233')
#NODE_LIST.append('160.59.234.99')



HOST_IP_ADDRESS = '160.58.150.64'
WEB_PORT = 5002


TEXT_MODEL = './frozen_east_text_detection.pb'
ANATOMY_MODEL = './anatomy_classification.params'
MARKER_MODEL = './marker_detection.params'
DICOM_DIR = '/app_cache/dicom'

SMTP_RELAY_SERVER = 'YOUR_SMTP_RELAY_SERVER_IP'
EMAIL_RECIPIENT_LIST = ['recipient_1@test.com', 'recipient_2@test.com']

# redis_client_list = []
# local_redis = None

# for item in NODE_LIST:
#     redisClient = redis.StrictRedis(host=item, port=6379, db=0)
#     redis_client_list.append(redisClient)
#     if item == 'localhost':
#         local_redis = redisClient


Base = declarative_base()

class False_Annotation(Base):
    __tablename__ = 'false_annotation'
    
    Id = Column(Integer, autoincrement=True, primary_key=True)
    Record_Date_Time = Column(DateTime, nullable=False,default=datetime.now)
    Msg_ID = Column(String)
    Accession_Number = Column(String)
    Patient_Name = Column(String)
    Exam_Room = Column(String)
    SOP_Instance_UID = Column(String)
    Problem_Detail = Column(String)
    User_Reply = Column(String)

class Received_Image(Base):
    __tablename__ = "received_image"
 
    Id = Column(Integer, primary_key=True)
    Record_Date_Time = Column(DateTime, nullable=False,default=datetime.now)
    Station_Name = Column(String)
    Accession_Number = Column(String)
    SOP_Instance_UID = Column(String)
    Study_Description = Column(String)
    Flip = Column(String)
    Anatomy_Detected = Column(String)
    Digital_Marker_Used = Column(String)
    Physical_Marker_Used = Column(String)
    Remarks = Column(String)

class Program_Error(Base):
    __tablename__ = 'program_error'
    
    Id = Column(Integer, autoincrement=True, primary_key=True)
    Record_Date_Time = Column(DateTime, nullable=False,default=datetime.now)
    Error_Detail = Column(String)
    
class Order_Detail(Base):
    __tablename__ = 'order_detail'
    
    Id = Column(Integer, autoincrement=True, primary_key=True)
    Record_Date_Time = Column(DateTime, nullable=False,default=datetime.now)
    Accession_Number = Column(String)
    Region = Column(String)
    Laterality = Column(String)
    Radiologist_Name = Column(String)
    Urgency = Column(String)
    Order_Control = Column(String)
    Order_Number = Column(String)
    Order_Status = Column(String)
    Exam_Room = Column(String)

engine = create_engine('postgresql://rad:rad@localhost:5432/rad_db')
#engine = create_engine('sqlite:////home/rad/app/DR_Warnings/server/web_server/warning_web_server/web_server.db')
Base.metadata.create_all(bind=engine)
Session = sessionmaker(bind=engine)
session = Session()

class DicomImageAnalyzer:
    def __init__(self):
        self.img = None
        text_detection_model = TEXT_MODEL
        anatomy_classification_model = ANATOMY_MODEL
        marker_detection_model = MARKER_MODEL
        
        dir_path = Path(__file__).parent.absolute()
        self.UpperExtrimityList = ['HAND','WRIST','FOREARM','ELBOW','HUMERUS','SHOULDER','FINGER']
        self.LowerExtrimityList = ['FOOT','ANKLE','LEG','LEG','FEMUR','TOES','KNEE','HIP','CALCANEUM']
        self.TrunkList = ['CHEST', 'ABDOMEN']
        
        template_folder = str(dir_path) + '/marker_templates/'
        list_marker = []
    
        for subdir, dirs, files in os.walk(template_folder):
            for f in files:
                manufacturer_name = subdir.replace(template_folder, '')
                marker_name = os.path.splitext(f)[0]
                template_img = cv2.imread(os.path.join(subdir,f), cv2.IMREAD_GRAYSCALE)
                temp_h, temp_w = template_img.shape[:2]
                dict_template = {'manufacturer_name':manufacturer_name,'name': manufacturer_name + '_' + marker_name, 'width': temp_w, 'image': template_img}
                list_marker.append(dict_template)
                
        self.marker = sorted(list_marker, key=itemgetter('width'), reverse=True)
        
        print('Starting DicomImageAnalyzer')
        # Load text detection network with opencv
        self.text_detection_net = cv2.dnn.readNet(text_detection_model)
        
        # load anatomy classification network with gluoncv
        
        anatomy_classes_df = pd.read_csv('anatomy_classes.csv')
        anatomy_classes_df.sort_values(by='id')
        self.anatomy_class_names = anatomy_classes_df['class'].tolist()
        num_classes = len(self.anatomy_class_names)
        
        model_name = 'SE_ResNext101_32x4d'
        self.anatomy_classification_net = get_model(model_name, classes=num_classes)
        self.anatomy_classification_net.load_parameters(ANATOMY_MODEL)        
        
        #load marker detection network with gluoncv
        
        object_classes_df = pd.read_csv('classes.csv')
        object_classes_df.sort_values(by='id')
        self.object_class_names = object_classes_df['class'].tolist()
        
        self.marker_detection_net = gcv.model_zoo.get_model('ssd_512_mobilenet1.0_custom', classes=self.object_class_names, pretrained_base=False)
        self.marker_detection_net.load_parameters(MARKER_MODEL)
        
        print('All networks loaded. DicomImageAnalyzer is ready!')
        
    def post_received_image_record(self, station_name=None, access_number=None, sop_uid=None, study_description=None, flip=None, anatomy_detected=None, digi_marker_used=None, physical_marker_used=None, remarks=None):
        record = Received_Image(Station_Name=station_name, Accession_Number = access_number, SOP_Instance_UID = sop_uid, Study_Description = study_description, Flip=flip, Anatomy_Detected=anatomy_detected, Digital_Marker_Used=digi_marker_used, Physical_Marker_Used=physical_marker_used, Remarks= remarks)
        session.add(record)
        session.commit()

    def post_problem_image_record(self, access_number, sop_uid, problem_detail):
        record = False_Annotation(Accession_Number = access_number, SOP_Instance_UID = sop_uid, Problem_Detail = problem_detail)
        session.add(record)
        session.commit()
        
    def send_mail(self, SUBJECT, BODY, TO, FROM, img_path):
        """With this function we send out our html email"""
     
        # Create message container - the correct MIME type is multipart/alternative here!
        MESSAGE = MIMEMultipart()
        MESSAGE['subject'] = SUBJECT
        MESSAGE['To'] =  ', '.join(TO)
        MESSAGE['From'] = FROM
        MESSAGE.preamble = """
    Your mail reader does not support the html format email!"""
     
        # Record the MIME type text/html.
        HTML_BODY = MIMEText(BODY, 'html')
     
        # Attach parts into message container.
        # According to RFC 2046, the last part of a multipart message, in this case
        # the HTML message, is best and preferred.
        MESSAGE.attach(HTML_BODY)
        
        fp = open(str(img_path), 'rb')
        msgImage = MIMEImage(fp.read())
        fp.close()
        MESSAGE.attach(msgImage)

        # Define the image's ID as referenced above
        msgImage.add_header('Content-ID', '<image1>')
     
        # The actual sending of the e-mail
        #server = smtplib.SMTP('160.86.51.129:25')    #old SMTP server address
        server = smtplib.SMTP(SMTP_RELAY_SERVER)     #new SMTP server address
     
        # Print debugging output when testing
        #if __name__ == "__main__":
        #    server.set_debuglevel(1)
     
        # Credentials (if needed) for sending the mail
        #password = ""
     
        #server.starttls()
        #server.login(FROM,password)
        server.sendmail(FROM, TO, MESSAGE.as_string())
        server.quit()

    def predict_anatomy(self):
        nd_img = nd.array(self.img)
        
        transform_fn = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])
        
        img_detect = transform_fn(nd_img)
        
        
        pred = self.anatomy_classification_net(img_detect.expand_dims(axis=0))

        ind = nd.argmax(pred, axis=1).astype('int')
        detected_class = self.anatomy_class_names[ind.asscalar()]
        print(detected_class)
            
        return detected_class
        
        
        
    def generate_pdf(self, input_dict, img_path, msg_id):
        width, height = int(8.27 * 100), int(11.7 * 100) # A4 at 300dpi
        
        page = Image.new('RGB', (width, height), 'white')
        fileName = img_path
        pixel = Image.open(fileName)
        basewidth = int(width*0.5)
        wpercent = (basewidth / float(pixel.size[0]))
        hsize = int((float(pixel.size[1]) * float(wpercent)))
        pixel = pixel.resize((basewidth, hsize), Image.ANTIALIAS)
        
        draw = ImageDraw.Draw(page)
        font = ImageFont.truetype("tahoma.ttf", 18)
        font_barcode = ImageFont.truetype("3OF9.TTF", 40, encoding="symb")
        msg_1 = 'Dear radiographer,'
        msg_2 = 'There is false annotation found for image, please help to check.'
        msg_3 = 'Problem Detail:'
        msg_4 = input_dict['problem_detail']
        
        draw.text((70, 90), msg_1, (0,0,0), font=font)
        draw.text((70, 120), msg_2, (0,0,0), font=font)
        draw.text((70, 150), msg_3, (0,0,0), font=font)
        draw.text((70, 180), msg_4, (0,0,0), font=font)
        page.paste(pixel, (70, 180+50))
        location = 180 + 50 + hsize +20
        
        msg ='Patient: {}   \nAccession Number: {}\nExam Room: {}\n'.format(input_dict['patient_name'], input_dict['access_number'], input_dict['exam_room'])
        draw.text((70, location), msg, (0,0,0), font=font)
        location = location + 70
        msg_barcode = '*'+str(input_dict['access_number'])+'*'
        draw.text((70, location), msg_barcode, (0,0,0), font=font_barcode)
        

        
        del draw
        server_root_path = Path(__file__).resolve().parents[2]
        file_name = msg_id + '.pdf'
        outputFilename =  server_root_path/'web_server'/'warning_web_server'/'static'/'pdf'/'false_annotation'/file_name
        #outputFilename = file_name
        page.save(outputFilename, "PDF", resolution=100.0)



    def emit_message(self, routing_key, msg_id, web_url, pdf_url, num):
        def send_msg(conn, routing_key, msg_id, web_url, pdf_url, num):
            try:
                channel = conn.channel()
                exchange = Exchange("warning", type="topic")
                producer = Producer(exchange=exchange, channel=channel, routing_key=routing_key)
                
                queue = Queue(name="", exchange=exchange, routing_key=routing_key)
                queue.maybe_bind(conn)
                queue.declare()
                producer.publish("From:"+conn.as_uri()+' Routint Key: '+routing_key)
                print('Got connection: {0!r}'.format(conn.as_uri()))
                print("From:"+conn.as_uri()+' Routint Key: '+routing_key)
                
                if num == 1:
                    web_url = web_url + '&num=1'
                    data = {'msg_id':msg_id, 'web_url':web_url, 'pdf_url':pdf_url}
                    #message = json.dumps(data)
                    producer.publish(data)
                    print('Got connection: {0!r}'.format(conn.as_uri()))

                    
                elif num == 2:
                    record = session.query(False_Annotation).filter(False_Annotation.Msg_ID == msg_id).first() #check if user has any reply
                    if record is not None:
                        if record.User_Reply == None:
                            web_url = web_url + '&num=2'
                            data = {'msg_id':msg_id, 'web_url':web_url, 'pdf_url':pdf_url}
                            #message = json.dumps(data)
                            producer.publish(data)
                            print('Got connection: {0!r}'.format(conn.as_uri()))
                
            except:
                print(traceback.format_exc())
                record = Program_Error(Error_Detail = traceback.format_exc())
                session.add(record)
                session.commit()
                pass
            
        conn_list = []

        for item in NODE_LIST:
            c = Connection('amqp://rad:rad@{}:5672'.format(item))
            conn_list.append(c)
        
        for item in conn_list:
            with connections[item].acquire(block=True) as conn:
                send_msg(conn, routing_key, msg_id, web_url, pdf_url, num)
        
        


    def handle_problem_image(self, image, access_number, patient_name, sop_uid, station_name, problem_detail):
        
        print ('handle problem image!')
        
        dir_path = Path(__file__).resolve()
        parent_path = dir_path.parents[0].absolute()
        parent_2_path = dir_path.parents[2].absolute()
        
        params = list()
        params.append(cv2.IMWRITE_JPEG_QUALITY)
        params.append(20)
        

        img_path = Path.joinpath(parent_path, 'problem_images', access_number+"_"+sop_uid+".jpg")
        print('img_path is '+img_path.as_posix())
        cv2.imwrite(str(img_path.as_posix()), image, params)
        

        web_img_path = Path.joinpath(parent_2_path, 'web_server/warning_web_server/static/img/problem_images', access_number+"_"+sop_uid+".jpg")
        print('web img:'+str(web_img_path))
        cv2.imwrite(str(web_img_path.as_posix()), image, params)
        
        if 'Reminder' not in problem_detail:
            exam_room = 'N/A'
            record_for_exam_room = session.query(Order_Detail).filter(Order_Detail.Accession_Number == access_number, Order_Detail.Order_Control =='NW').order_by(Order_Detail.Id.desc()).first()
            if record_for_exam_room is not None:
                exam_room = record_for_exam_room.Exam_Room.replace(' ','') #replace space for TSH R1, R2, R3

            warning_type = 'false_annotation'
            msg_id = str(uuid.uuid4())
            img_id = access_number+"_"+sop_uid+".jpg"

            input_row = False_Annotation(Msg_ID=msg_id, Patient_Name=patient_name, Accession_Number=access_number, Exam_Room=exam_room, SOP_Instance_UID=sop_uid, Problem_Detail= problem_detail)
            session.add(input_row)
            session.commit()
            
            #myhostname = socket.getfqdn(socket.gethostname())
            #server_ip = socket.gethostbyname(myhostname)
            
            input_dict = {'patient_name': patient_name, 'access_number': access_number, 'exam_room':exam_room, 'problem_detail':problem_detail}
            
            self.generate_pdf(input_dict, img_path, msg_id)
            
            routing_key = exam_room + '.' + warning_type
            web_url ='http://{}:{}/warning?msg_id={}&warning_type={}&img_id={}'.format(HOST_IP_ADDRESS, WEB_PORT, msg_id, warning_type,img_id)
            pdf_file_path = msg_id + '.pdf'
            pdf_url ='http://{}:{}/static/pdf/false_annotation/{}'.format(HOST_IP_ADDRESS, WEB_PORT, pdf_file_path)
            
            self.emit_message(routing_key, msg_id, web_url, pdf_url, 1)
            t = threading.Timer(10*60, self.emit_message, [routing_key, msg_id, web_url, pdf_url, 2])
            t.start()
        
        warning_Message = '<b>Dear PACS Admins, <br><br>Please help to follow the following problem image.<br><br>Accession Number : <br>'+ access_number + '<br><br>Problem : <br><i>' + problem_detail + '</i> <br><br>Attached Image:<br><br></b><img src="cid:image1" style="height: 80%; width: 80%; object-fit: contain"><br><br><b>Thank you for your kind attention!</b>'
        if len(access_number)>2:
            my_hospital = access_number[:3]
        else:
            my_hospital = 'Unknown'
        if 'Reminder' not in problem_detail:
            subject_txt = 'Warning mail! -- ' + my_hospital
        else:
            subject_txt = 'Reminder mail! -- ' + my_hospital

        self.send_mail(subject_txt, warning_Message, EMAIL_RECIPIENT_LIST, 'SureSide@ha.org.hk', img_path)


    def handle_no_marker_image(self, image, access_number, sop_uid):
        print ('handle no marker image!')
        
        #dir_path = Path(__file__).parent.absolute()
        #save_directory = str(dir_path) + '/no_marker_image'
        #params = list()
        #params.append(cv2.IMWRITE_JPEG_QUALITY)
        #params.append(20)
        #img_path = os.path.join(save_directory, access_number+"_"+sop_uid+".jpg")
        #cv2.imwrite(img_path, image, params)

    def ocr_agfa_marker_preprocess(self): ## OCR agfa marker
        
        left_width_Agfa_side_marker = 187
        left_height_Agfa_side_marker = 298
        
        right_width_Agfa_side_marker = 193
        right_height_Agfa_side_marker = 266
        
        left_w_h_ratio_Agfa_side_marker = left_width_Agfa_side_marker / left_height_Agfa_side_marker
        right_w_h_ratio_Agfa_side_marker = right_width_Agfa_side_marker / right_height_Agfa_side_marker

        height, width = self.img.shape[:2]

        kernel = np.ones((3,3),np.uint8)
        
        blur = cv2.bilateralFilter(self.img,3,75,75)
        equalized_img = cv2.equalizeHist(blur)

        edge = cv2.Canny(equalized_img, 100, 200)
        
        laplacian_img = cv2.Laplacian(edge, cv2.CV_8UC1)
        converted_img = cv2.convertScaleAbs(laplacian_img)
        dilated_img = cv2.dilate(converted_img,kernel,iterations = 3)
        #cv2.imwrite(accession_number + '.png', dilated_img)
        
        ocr_result = []

        _, contours, hierarchy = cv2.findContours(dilated_img,cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        

        for cnt in contours:
            approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
            if len(approx)==4:
                x,y,w,h = cv2.boundingRect(cnt)
                area = w * h
                if area > 2000 and area < 500000: # filter out small area
                    cnt_area = cv2.contourArea(cnt)
                    area_over_cnt_area = area / cnt_area
                    if area_over_cnt_area > 0.6 and area_over_cnt_area < 1.5: # if rectangle, area of contour should be equal to area of boudingrect
                        ocr_roi = img[y : y + h,x:x + w]

                        ret2,ocr_roi = cv2.threshold(ocr_roi,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                        ocr_roi = cv2.bitwise_not(ocr_roi)
                        ocr_roi = cv2.resize(ocr_roi, None, fx=3, fy=3)
                        w_h_ratio = w/h
                        left_ratio = w_h_ratio / left_w_h_ratio_Agfa_side_marker
                        right_ratio = w_h_ratio / right_w_h_ratio_Agfa_side_marker
                        if left_ratio > 0.9 and left_ratio < 1.1: #check if roi is side marker
                            ocr_text = pytesseract.image_to_string(Image.fromarray(ocr_roi), config='--psm 10') 
                            ocr_text = ocr_text
                        elif right_ratio > 0.9 and right_ratio < 1.1: #check if roi is side marker
                            ocr_text = pytesseract.image_to_string(Image.fromarray(ocr_roi), config='--psm 10') 
                            ocr_text = ocr_text
                        else:
                            ocr_text = pytesseract.image_to_string(Image.fromarray(ocr_roi))
                        print (ocr_text)
                        
                        mask = np.zeros(self.img.shape,np.uint8)
                        self.img[y:y+h,x:x+w] = mask[y:y+h,x:x+w]
                        
                        if ocr_text.strip() != '':
                            if ocr_text.strip()=='L':
                                ocr_result.append('Left')
                            elif ocr_text.strip()=='R':
                                ocr_result.append('Right')
                            else:
                                ocr_result.append(ocr_text.strip())
        return ocr_result


    def template_match_digital_marker(self, marker_name, template): ## template matching for digital marker
        ret,th_template = cv2.threshold(template,240,255,cv2.THRESH_BINARY)
        temp_h, temp_w = template.shape[:2]
        img_h, img_w =self.img.shape[:2]
        if img_h > temp_h and img_w > temp_w:
            ret,th_img = cv2.threshold(self.img,240,255,cv2.THRESH_BINARY)
            res= cv2.matchTemplate(th_img,th_template,cv2.TM_CCOEFF_NORMED)
            NULL, max_val,NULL, max_loc = cv2.minMaxLoc(res)  
            print('marker name:'+marker_name)
            print (' Value: '+ str(max_val))
            if(max_val > 0.85):
                print (marker_name + ' is found!')
                x, y = max_loc
                w = temp_w
                h = temp_h
                mask = np.zeros(self.img.shape,np.uint8)
                self.img[y:y+h,x:x+w] = mask[y:y+h,x:x+w]
                
                return marker_name
            else:
                return 'Nil'
        else:
            return 'Nil'


    ############ Utility functions ############
    def decode(self, scores, geometry, scoreThresh):
        detections = []
        confidences = []

        ############ CHECK DIMENSIONS AND SHAPES OF geometry AND scores ############
        assert len(scores.shape) == 4, "Incorrect dimensions of scores"
        assert len(geometry.shape) == 4, "Incorrect dimensions of geometry"
        assert scores.shape[0] == 1, "Invalid dimensions of scores"
        assert geometry.shape[0] == 1, "Invalid dimensions of geometry"
        assert scores.shape[1] == 1, "Invalid dimensions of scores"
        assert geometry.shape[1] == 5, "Invalid dimensions of geometry"
        assert scores.shape[2] == geometry.shape[2], "Invalid dimensions of scores and geometry"
        assert scores.shape[3] == geometry.shape[3], "Invalid dimensions of scores and geometry"
        height = scores.shape[2]
        width = scores.shape[3]
        for y in range(0, height):

            # Extract data from scores
            scoresData = scores[0][0][y]
            x0_data = geometry[0][0][y]
            x1_data = geometry[0][1][y]
            x2_data = geometry[0][2][y]
            x3_data = geometry[0][3][y]
            anglesData = geometry[0][4][y]
            for x in range(0, width):
                score = scoresData[x]

                # If score is lower than threshold score, move to next x
                if(score < scoreThresh):
                    continue

                # Calculate offset
                offsetX = x * 4.0
                offsetY = y * 4.0
                angle = anglesData[x]

                # Calculate cos and sin of angle
                cosA = math.cos(angle)
                sinA = math.sin(angle)
                h = x0_data[x] + x2_data[x]
                w = x1_data[x] + x3_data[x]

                # Calculate offset
                offset = ([offsetX + cosA * x1_data[x] + sinA * x2_data[x], offsetY - sinA * x1_data[x] + cosA * x2_data[x]])

                # Find points for rectangle
                p1 = (-sinA * h + offset[0], -cosA * h + offset[1])
                p3 = (-cosA * w + offset[0],  sinA * w + offset[1])
                center = (0.5*(p1[0]+p3[0]), 0.5*(p1[1]+p3[1]))
                detections.append((center, (w,h), -1*angle * 180.0 / math.pi))
                confidences.append(float(score))

        # Return detections and confidences
        return [detections, confidences]

    def detect_text_area(self, net):
        height, width = self.img.shape
        # Read and store arguments
        confThreshold = 0.5
        nmsThreshold = 0.5
        inpWidth = 1280
        inpHeight = 1280

        # Create a new named window
        outNames = []
        outNames.append("feature_fusion/Conv_7/Sigmoid")
        outNames.append("feature_fusion/concat_3")

        # Get cv_im height and width
        height_ = self.img.shape[0]
        width_ = self.img.shape[1]
        rW = width_ / float(inpWidth)
        rH = height_ / float(inpHeight)
        
        input_im_color = cv2.cvtColor(self.img,cv2.COLOR_GRAY2RGB)

        # Create a 4D blob from input_im.
        blob = cv2.dnn.blobFromImage(input_im_color, 1.0, (inpWidth, inpHeight), (123.68, 116.78, 103.94), True, False)

        # Run the model
        net.setInput(blob)
        outs = net.forward(outNames)
        t, _ = net.getPerfProfile()
        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())

        # Get scores and geometry
        scores = outs[0]
        geometry = outs[1]
        [boxes, confidences] = self.decode(scores, geometry, confThreshold)
        
        result_coordinate_list = []
        # Apply NMS
        indices = cv2.dnn.NMSBoxesRotated(boxes, confidences, confThreshold,nmsThreshold)
        for i in indices:
            point_list = []
            # get 4 corners of the rotated rect
            vertices = cv2.boxPoints(boxes[i[0]])
            # scale the bounding box coordinates based on the respective ratios
            for j in range(4):
                vertices[j][0] *= rW
                vertices[j][1] *= rH
            for j in range(4):
                p1 = (vertices[j][0], vertices[j][1])
                p2 = (vertices[(j + 1) % 4][0], vertices[(j + 1) % 4][1])
                if p1 not in point_list:
                    point_list.append(p1)
                if p2 not in point_list:
                    point_list.append(p2)
                #cv2.line(input_im, p1, p2, (0, 255, 0), 1);
            point_np= np.array(point_list, dtype=np.float32)
            x,y,w,h = cv2.boundingRect(point_np)
            #cv.rectangle(input_im,(x,y),(x+w,y+h),(0,0,255),2)
            #text_roi = self.img[y:y+h, x:x+w].copy()
            result_coordinate = {'x':x, 'y':y, 'w':w, 'h':h}
            result_coordinate_list.append(result_coordinate)
        return result_coordinate_list

    def vconcat_resize_min(self, im_list, interpolation=cv2.INTER_CUBIC):
        w_min = min(im.shape[1] for im in im_list)
        im_list_resize = [cv2.resize(im, (w_min, int(im.shape[0] * w_min / im.shape[1])), interpolation=interpolation)
                          for im in im_list]
        return cv2.vconcat(im_list_resize)
    
    def cv_plot_bbox(self, img, bboxes, scores=None, labels=None, thresh=0.5,
                     class_names=None, colors=None,
                     absolute_coordinates=True, scale=1.0):
        """Visualize bounding boxes with OpenCV.
    
        Parameters
        ----------
        img : numpy.ndarray or mxnet.nd.NDArray
            Image with shape `H, W, 3`.
        bboxes : numpy.ndarray or mxnet.nd.NDArray
            Bounding boxes with shape `N, 4`. Where `N` is the number of boxes.
        scores : numpy.ndarray or mxnet.nd.NDArray, optional
            Confidence scores of the provided `bboxes` with shape `N`.
        labels : numpy.ndarray or mxnet.nd.NDArray, optional
            Class labels of the provided `bboxes` with shape `N`.
        thresh : float, optional, default 0.5
            Display threshold if `scores` is provided. Scores with less than `thresh`
            will be ignored in display, this is visually more elegant if you have
            a large number of bounding boxes with very small scores.
        class_names : list of str, optional
            Description of parameter `class_names`.
        colors : dict, optional
            You can provide desired colors as {0: (255, 0, 0), 1:(0, 255, 0), ...}, otherwise
            random colors will be substituted.
        absolute_coordinates : bool
            If `True`, absolute coordinates will be considered, otherwise coordinates
            are interpreted as in range(0, 1).
        scale : float
            The scale of output image, which may affect the positions of boxes
    
        Returns
        -------
        numpy.ndarray
            The image with detected results.
    
        """
        if labels is not None and not len(bboxes) == len(labels):
            raise ValueError('The length of labels and bboxes mismatch, {} vs {}'
                             .format(len(labels), len(bboxes)))
        if scores is not None and not len(bboxes) == len(scores):
            raise ValueError('The length of scores and bboxes mismatch, {} vs {}'
                             .format(len(scores), len(bboxes)))
    
        if isinstance(img, mx.nd.NDArray):
            img = img.asnumpy()
        if isinstance(bboxes, mx.nd.NDArray):
            bboxes = bboxes.asnumpy()
        if isinstance(labels, mx.nd.NDArray):
            labels = labels.asnumpy()
        if isinstance(scores, mx.nd.NDArray):
            scores = scores.asnumpy()
        if len(bboxes) < 1:
            return img
    
        if not absolute_coordinates:
            # convert to absolute coordinates using image shape
            height = img.shape[0]
            width = img.shape[1]
            bboxes[:, (0, 2)] *= width
            bboxes[:, (1, 3)] *= height
        else:
            bboxes *= scale
    
        result_dict = None
        subset_detect_list = []
        # use random colors if None is provided
        if colors is None:
            colors = dict()
        
        for i, bbox in enumerate(bboxes):
            if scores is not None and scores.flat[i] < thresh:
                continue
            if labels is not None and labels.flat[i] < 0:
                continue
            cls_id = int(labels.flat[i]) if labels is not None else -1
            if cls_id not in colors:
                if class_names is not None:
                    colors[cls_id] = plt.get_cmap('hsv')(cls_id / len(class_names))
                else:
                    colors[cls_id] = (random.random(), random.random(), random.random())
            xmin, ymin, xmax, ymax = [int(x) for x in bbox]
            
            bbox_coordinate = (xmin, ymin, xmax, ymax)
            bbox_area = abs(xmax - xmin)*abs(ymax - ymin)
            
            bcolor = [x * 255 for x in colors[cls_id]]
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), bcolor, 2)
    
            if class_names is not None and cls_id < len(class_names):
                class_name = class_names[cls_id]
            else:
                class_name = str(cls_id) if cls_id >= 0 else ''
            score = '{:d}%'.format(int(scores.flat[i]*100)) if scores is not None else ''
            
            if bbox_area > 500:
                if class_name or score:
                    y = ymin - 15 if ymin - 15 > 15 else ymin + 15
                    img_width = img.shape[1]
                    if xmin > img_width/2:
                        xmin = 10
                    cv2.putText(img, '{:s} {:s}'.format(class_name, score),
                                (xmin, y), cv2.FONT_HERSHEY_SIMPLEX, min(scale/2, 2),
                                bcolor, min(int(scale), 5), lineType=cv2.LINE_AA)
            
                    
                    subset_dict = {'class_name':class_name, 'score':score, 'area':bbox_area, 'coordinate':bbox_coordinate}
                    subset_detect_list.append(subset_dict)
    
            result_dict = {'img':img, 'subset_detect_list':subset_detect_list}
            
        return result_dict

    def get_classname_bbox(self, bboxes, scores=None, labels=None, thresh=0.5,
                     class_names=None):
                         
        result_list = []

        if labels is not None and not len(bboxes) == len(labels):
            raise ValueError('The length of labels and bboxes mismatch, {} vs {}'
                             .format(len(labels), len(bboxes)))
        if scores is not None and not len(bboxes) == len(scores):
            raise ValueError('The length of scores and bboxes mismatch, {} vs {}'
                             .format(len(scores), len(bboxes)))
    
        if isinstance(bboxes, mx.nd.NDArray):
            bboxes = bboxes.asnumpy()
        if isinstance(labels, mx.nd.NDArray):
            labels = labels.asnumpy()
        if isinstance(scores, mx.nd.NDArray):
            scores = scores.asnumpy()
        if len(bboxes) < 1:
            return result_list
    
        for i, bbox in enumerate(bboxes):
            if scores is not None and scores.flat[i] < thresh:
                continue
            if labels is not None and labels.flat[i] < 0:
                continue
            cls_id = int(labels.flat[i]) if labels is not None else -1
    
            if class_names is not None and cls_id < len(class_names):
                class_name = class_names[cls_id]
            else:
                class_name = str(cls_id) if cls_id >= 0 else ''
            score = '{:d}%'.format(int(scores.flat[i]*100)) if scores is not None else ''
            if class_name or score:
                final_score = scores.flat[i]
                result_dict = {'class_name':class_name, 'score':final_score}
                result_list.append(result_dict)
        return result_list

    def  detect_implant(self):

        color_input_img = cv2.cvtColor(self.img,cv2.COLOR_GRAY2RGB)


        nd_img = nd.array(color_input_img)
        x, transform_img = gcv.data.transforms.presets.ssd.transform_test(nd_img, 512)
        
        height, width = self.img.shape[:2]
        transformed_height, transformed_width = transform_img.shape[:2]
        
        transform_scale = height/transformed_height
        cid, score, bbox = self.marker_detection_net(x)
        #ax = viz.plot_bbox(transform_img, bbox[0], score[0], cid[0], class_names=classes, thresh=0.2)
        result_dict = self.cv_plot_bbox(transform_img, bbox[0], score[0], cid[0], class_names=self.object_class_names, thresh=0.5)
        
        if result_dict:
            result_list = result_dict['subset_detect_list']
            
            for item in result_list:
                if 'ortho_implant' in item['class_name']:
                    xmin, ymin, xmax, ymax = item['coordinate']
                    x = int(xmin * transform_scale)
                    y = int(ymin * transform_scale)
                    h = int(abs(ymax * transform_scale - ymin * transform_scale))
                    w = int(abs(xmax * transform_scale - xmin * transform_scale))
                    mask = np.zeros(self.img.shape,np.uint8)
                    self.img[y:y+h,x:x+w] = mask[y:y+h,x:x+w]
                    
        #cv2.imwrite('implant.png', self.img)
                
    def  detect_marker(self):
        color_input_img = cv2.cvtColor(self.img,cv2.COLOR_GRAY2RGB)

        detection_list = []
        detected_laterality = ''
        
        #windows = sw.generate(color_input_img, sw.DimOrder.HeightWidthChannel, 1024, 0.1)
        
        #for window in windows:
        #subset = color_input_img[ window.indices() ]
        
        subset = color_input_img
        nd_img = nd.array(subset)
        x, transform_img = gcv.data.transforms.presets.ssd.transform_test(nd_img, 512)
        cid, score, bbox = self.marker_detection_net(x)
        #ax = viz.plot_bbox(transform_img, bbox[0], score[0], cid[0], class_names=classes, thresh=0.2)
        result_dict = self.cv_plot_bbox(transform_img, bbox[0], score[0], cid[0], class_names=self.object_class_names, thresh=0.85)
        
        if result_dict:
            detection_list.append(result_dict)
        
        result_list = []
        final_result_list = []
        

        img_list = []
        img_list.append(color_input_img)
        for item in detection_list:
            img_list.append(item['img'])
            result_list.extend(item['subset_detect_list'])
        
        if len(result_list)>0:
            result_df = pd.DataFrame(result_list)
            
            right_df = result_df[result_df['class_name'].str.contains('right')]
            left_df = result_df[result_df['class_name'].str.contains('left')]
            right_max_area = right_df['area'].max()
            left_max_area = left_df['area'].max()
            
            
            
            right_marker_num = len(result_df[result_df['class_name'].str.contains('right')])
            
            left_marker_num = len(result_df[result_df['class_name'].str.contains('left')])
            
            print('right: ' + str(right_marker_num))
            print('right max: ' + str(right_max_area))
            
            print('left: ' + str(left_marker_num))
            print('left max: ' + str(left_max_area))
            
            if right_marker_num and left_marker_num:
                if right_max_area > left_max_area:
                    final_df = result_df[~result_df['class_name'].str.contains('left')]
                else:
                    final_df = result_df[~result_df['class_name'].str.contains('right')]
            
                result_list = final_df.to_dict(orient='records')
            
        for item in result_list:
            if item['class_name'] not in final_result_list:
                final_result_list.append(item['class_name'])

        #im_v = self.vconcat_resize_min(img_list)
        #result_filename = '/home/rad/app/DR_Warnings/image_annotation_gallery/static/images/' + str(uuid.uuid4()) + '.jpg'
        #cv2.imwrite(result_filename, im_v)
        
        return final_result_list
        

    def ocr_digital_marker(self, manufacturer_name): ## feature matching for physical marker
        ocr_result = []
        
        if 'Agfa' in manufacturer_name or 'Varian' in manufacturer_name:
            left_width_Agfa_side_marker = 187
            left_height_Agfa_side_marker = 298
            
            right_width_Agfa_side_marker = 193
            right_height_Agfa_side_marker = 266
            
            left_w_h_ratio_Agfa_side_marker = left_width_Agfa_side_marker / left_height_Agfa_side_marker
            right_w_h_ratio_Agfa_side_marker = right_width_Agfa_side_marker / right_height_Agfa_side_marker
    
            height, width = self.img.shape[:2]
    
            kernel = np.ones((3,3),np.uint8)
            
            blur = cv2.bilateralFilter(self.img,3,75,75)
            equalized_img = cv2.equalizeHist(blur)
    
            edge = cv2.Canny(equalized_img, 100, 200)
            
            laplacian_img = cv2.Laplacian(edge, cv2.CV_8UC1)
            converted_img = cv2.convertScaleAbs(laplacian_img)
            dilated_img = cv2.dilate(converted_img,kernel,iterations = 3)
            #cv2.imwrite(accession_number + '.png', dilated_img)
            
            contours, hierarchy = cv2.findContours(dilated_img,cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            
    
            for cnt in contours:
                approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
                if len(approx)==4:
                    x,y,w,h = cv2.boundingRect(cnt)
                    area = w * h
                    if area > 2000 and area < 500000: # filter out small area
                        cnt_area = cv2.contourArea(cnt)
                        area_over_cnt_area = area / cnt_area
                        if area_over_cnt_area > 0.6 and area_over_cnt_area < 1.5: # if rectangle, area of contour should be equal to area of boudingrect
                            ocr_roi = self.img[y : y + h,x:x + w]
    
                            ret2,ocr_roi = cv2.threshold(ocr_roi,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                            ocr_roi = cv2.bitwise_not(ocr_roi)
                            ocr_roi = cv2.resize(ocr_roi, None, fx=3, fy=3)
                            w_h_ratio = w/h
                            left_ratio = w_h_ratio / left_w_h_ratio_Agfa_side_marker
                            right_ratio = w_h_ratio / right_w_h_ratio_Agfa_side_marker
                            if left_ratio > 0.9 and left_ratio < 1.1: #check if roi is side marker
                                ocr_text = pytesseract.image_to_string(Image.fromarray(ocr_roi), config='--psm 10') 
                                ocr_text = ocr_text
                            elif right_ratio > 0.9 and right_ratio < 1.1: #check if roi is side marker
                                ocr_text = pytesseract.image_to_string(Image.fromarray(ocr_roi), config='--psm 10') 
                                ocr_text = ocr_text
                            else:
                                ocr_text = pytesseract.image_to_string(Image.fromarray(ocr_roi))
                            print (ocr_text)
                            
                            mask = np.zeros(self.img.shape,np.uint8)
                            self.img[y:y+h,x:x+w] = mask[y:y+h,x:x+w]
                            
                            if ocr_text.strip() != '':
                                if ocr_text.strip()=='L':
                                    ocr_result.append('Left')
                                elif ocr_text.strip()=='R':
                                    ocr_result.append('Right')
                                else:
                                    ocr_result.append(ocr_text.strip())
            return ocr_result
        
        else:
            result_list = self.detect_text_area(self.text_detection_net)
            if len(result_list) > 0:
                for i, item in enumerate(result_list):
                    x, y, w, h = item['x'], item['y'], item['w'], item['h']
                    text_roi = self.img[y:y+h, x:x+w].copy()
                    
                    if h > 0 and w >0:
                        resized_item = cv2.resize(text_roi, None, fx=3, fy=3)
                        resized_item = cv2.copyMakeBorder(resized_item,100,100,100,100,cv2.BORDER_CONSTANT,value=(0,0,0))
                        ret,th_img = cv2.threshold(resized_item,250,255,cv2.THRESH_BINARY)
                        th_img = cv2.bitwise_not(th_img)
                        
                        #cv2.imwrite('ocr'+str(i)+'.jpg', resized_item)
    
                        ocr_text = pytesseract.image_to_string(Image.fromarray(th_img))
                        ocr_text = ocr_text.strip()
                        print('ocr:'+ocr_text)
                        if ocr_text != '':
                            if ocr_text == 'L':
                                ocr_text = ocr_text.replace('L', '')
                            elif ocr_text == 'R':
                                ocr_text = ocr_text.replace('R', '')
                            ocr_result.append(ocr_text.strip())
                            if len(ocr_text) > 1:
                                mask = np.zeros(self.img.shape,np.uint8)
                                self.img[y:y+h,x:x+w] = mask[y:y+h,x:x+w]
            return ocr_result 



    def new_instance_received(self,dicom_path, input_img):
    ##
    ## The following function is called each time a new instance is
    ## received.
    ##
        print('**************** new instance ****************\n\n')
        input_dataset = pydicom.dcmread(dicom_path)

        keys_list = ['ImageType', 'PatientName', 'SOPInstanceUID', 'StudyDescription', 'StationName', 'AccessionNumber', 'Manufacturer']
        if all(key in input_dataset for key in keys_list):
            ImageType = str(input_dataset.ImageType)
            PatientName = str(input_dataset.PatientName)
            SOPInstanceUID = str(input_dataset.SOPInstanceUID)
            StudyDescription = str(input_dataset.StudyDescription)
            
            StationName = str(input_dataset.StationName)
            
            AccessionNumber = str(input_dataset.AccessionNumber)
            Manufacturer = str(input_dataset.Manufacturer)

            ## Remove the possible trailing characters due to DICOM padding
            PatientName = PatientName.strip()
            Manufacturer = Manufacturer.strip()
            SOPInstanceUID = SOPInstanceUID.strip()
            StudyDescription = StudyDescription.strip()
            StationName = StationName.strip()
            
            if 'KODAK' in Manufacturer:
                Manufacturer = 'CARESTREAM HEALTH'

            print ('Patient Name: ' + PatientName)
            print ('Manufacturer: ' + Manufacturer)
            print ('Station Name: ' + StationName)
            print ('Accession Number: ' + AccessionNumber)


            # image_exist_list = []
            # for redis_client in redis_client_list:
            #     try:
            #         redis_exist =redis_client.sismember('scan_instance_uid_set', SOPInstanceUID)
            #         image_exist_list.append(redis_exist)
            #     except:
            #         print('Redis connection error')
            #         pass
            # print(image_exist_list)

            # if any(image_exist_list):
            #     print("Skip cmove, SOP Instance UID exists in redis record!")
            # else:
            #     local_redis.sadd('scan_instance_uid_set', SOPInstanceUID)
            if True:
                self.img = input_img
                
                img_height, img_width = self.img.shape[:2]
                original_img = self.img
    
                ### Preform Anatomy Prediction with Machine Learning
    
                predicted_anatomy_result = self.predict_anatomy()
                
                print('Predicted Class: ' + predicted_anatomy_result)
                
                grey_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
                self.img = cv2.copyMakeBorder(grey_img,100,100,100,100,cv2.BORDER_CONSTANT,value=(0,0,0))
                #cv2.imwrite(SOPInstanceUID+'.png',img)
    
                final_marker_result = []
                digital_marker_result = []
                physical_marker_result = []
                
                for item in self.marker:
                    if item['manufacturer_name'] in Manufacturer.upper():
                        scan_digital_result = self.template_match_digital_marker(item['name'],item['image'])
                        if scan_digital_result != 'Nil':
                            digital_marker_result.append(scan_digital_result)
                
                ### ocr_digital_marker cause false warning for L and R detection so remove the detected L and R from OCR
                ocr_result = self.ocr_digital_marker(Manufacturer)
                digital_marker_result.extend(ocr_result)
                
                #self.detect_implant()
                
                object_detect_result = self.detect_marker()
                
                for item in object_detect_result:
                    if 'physical' in item:
                        physical_marker_result.append(item)        
                    elif 'digital' in item:
                        left_in_digital =  any("left" in s.lower() for s in digital_marker_result)
                        right_in_digital =  any("right" in s.lower() for s in digital_marker_result)
                        if not left_in_digital and not right_in_digital:
                            digital_marker_result.append(item)
                
                digital_marker_list_txt = ','.join(digital_marker_result)
                physical_marker_list_txt = ','.join(physical_marker_result)
                
                
                print('digital marker:'+digital_marker_list_txt)
                print('physical marker:'+physical_marker_list_txt)
                
                final_marker_result = digital_marker_result + physical_marker_result
    
                flip_status = 'UnKnown'
                
                if 'CARESTREAM' in Manufacturer.upper():
                    carestream_flip_private_tag = str(input_dataset[0x0029,0x1019].value)
                    if carestream_flip_private_tag.replace(' ', '') == '1':
                        flip_status = 'PA'
                    else:
                        flip_status = 'AP'
    
                # elif 'KODAK' in Manufacturer.upper():
                #     carestream_flip_private_tag = str(input_dataset[0x0029,0x1019].value)
                #     if carestream_flip_private_tag.replace(' ', '') == '1':
                #         flip_status = 'PA'
                #     else:
                #         flip_status = 'AP'
                                
                elif 'AGFA' in Manufacturer.upper() and 'ViewPosition' in input_dataset:
                    ViewPosition = str(input_dataset.ViewPosition)
                    ViewPosition = ViewPosition.strip()
                    flip_status = ViewPosition
                    
                elif 'VARIAN' in Manufacturer.upper() and 'ViewPosition' in input_dataset:
                    ViewPosition = str(input_dataset.ViewPosition)
                    ViewPosition = ViewPosition.strip()
                    flip_status = ViewPosition
                    
                elif 'FUJI' in Manufacturer.upper() and 'ViewPosition' in input_dataset:
                    ViewPosition = str(input_dataset.ViewPosition)
                    ViewPosition = ViewPosition.strip()
                    flip_status = ViewPosition
    
                elif 'GE' in Manufacturer.upper() and 'PatientOrientation' in input_dataset and 'ViewPosition' in input_dataset:
                    PatientOrientation = str(input_dataset.PatientOrientation)
                    PatientOrientation = PatientOrientation.strip()
                    ViewPosition = str(input_dataset.ViewPosition)
                    ViewPosition = ViewPosition.strip()
                    flip_status = ViewPosition
                    if flip_status == 'AP' and 'R' in PatientOrientation:
                        flip_status = 'PA'
                    elif flip_status == 'PA' and 'R' in PatientOrientation:
                        flip_status = 'AP'
                    
                if 'CANON' in Manufacturer.upper() and any('original' in result for result in final_marker_result):
                    flip_status = 'AP'
                elif 'CANON' in Manufacturer.upper() and any('flipped' in result for result in final_marker_result):
                    flip_status = 'PA'
                
    
                self.post_received_image_record(StationName, AccessionNumber, SOPInstanceUID, StudyDescription, flip_status, predicted_anatomy_result, digital_marker_list_txt, physical_marker_list_txt, '')
                
                if StudyDescription.upper() in self.TrunkList:                                    
                        
                    #if (all('sitting' not in result.lower() for result in final_marker_result) and all('sit' not in result.lower() for result in final_marker_result) and all('sup' not in result.lower() for result in final_marker_result) and all('standing' not in result.lower() for result in final_marker_result) and all('erect' not in result.lower() for result in final_marker_result)) and 'Chest' in StudyDescription and flip_status == 'AP':
                        #self.handle_no_marker_image(self.original_img, AccessionNumber, SOPInstanceUID)
                    #if (all('pa' not in result.lower() for result in final_marker_result)) and 'Abdomen' in StudyDescription and flip_status == 'PA':
                        #self.handle_no_marker_image(self.original_img, AccessionNumber, SOPInstanceUID)
                    if (any('AP' in result for result in final_marker_result) or any('sup' in result.lower() for result in final_marker_result)) and flip_status == 'PA':
                        self.handle_problem_image(original_img, AccessionNumber,PatientName , SOPInstanceUID, StationName, 'Marker is AP but image is flipped')
                    if any('PA' in result for result in final_marker_result) and flip_status == 'AP':
                        self.handle_problem_image(original_img, AccessionNumber,PatientName , SOPInstanceUID, StationName, 'Marker is PA, but image is NOT flipped')
                    # if 'CHEST' in StudyDescription.upper() and 'AP' not in final_marker_result and 'Sup' not in final_marker_result and flip_status == 'AP':
                        # handle_problem_image(original_img, AccessionNumber, SOPInstanceUID, StationName, 'Chest image. Image is NOT flipped, but no AP sit or Supine marker applied')
                    if 'ABDOMEN' in StudyDescription.upper() and not any('PA' in result for result in final_marker_result) and flip_status == 'PA' and 'Gallbladder' not in StudyDescription and 'Enema' not in StudyDescription:
                        if 'abdomen' in predicted_anatomy_result:
                            self.handle_problem_image(original_img, AccessionNumber, PatientName , SOPInstanceUID, StationName, 'Abdomen image. Image is flipped, but no PA Erect marker applied')
                        else:
                            self.handle_problem_image(original_img, AccessionNumber, PatientName, SOPInstanceUID, StationName, 'Image is not Abdomen but stored in Abdomen folder. Image is flipped, but no PA Erect marker applied')
                    # if any('L.' in result for result in final_marker_result):
                        # x = str(result.split('.')[1])
                        # if x > img_width/2:
                            # self.handle_problem_image(img, AccessionNumber, SOPInstanceUID, StationName, 'Left marker is placed on the Right side')
                    # elif any('R.' in result for result in final_marker_result):
                        # x = str(result.split('.')[1])
                        # if x < img_width/2:
                            # self.handle_problem_image(img, AccessionNumber, SOPInstanceUID, StationName, 'Right marker is placed on the Left side')
    
    
                elif StudyDescription.upper() in self.UpperExtrimityList or StudyDescription.upper() in self.LowerExtrimityList:
                    record = session.query(Order_Detail).filter(Order_Detail.Accession_Number == AccessionNumber).first()
                    if record is not None:
                        record_laterality =record.Laterality
                        if StudyDescription.upper() == 'CALCANEUM' or StudyDescription.upper() == 'HIP' or StudyDescription.upper() == 'KNEE':
                            if any('LEFT' in result.upper() for result in final_marker_result) and ('Right' in record_laterality):
                                if 'both' not in predicted_anatomy_result:
                                    self.handle_problem_image(original_img, AccessionNumber, PatientName, SOPInstanceUID, StationName, 'Marker is Left, but GCRS request is Right')
                                #else:
                                    #self.handle_problem_image(original_img, AccessionNumber, PatientName  , SOPInstanceUID, StationName, 'Reminder: Both side image. Marker is Left, but GCRS request is Right')
                            elif any('RIGHT' in result.upper() for result in final_marker_result) and ('Left'  in record_laterality):
                                if 'both' not in predicted_anatomy_result:
                                    self.handle_problem_image(original_img, AccessionNumber, PatientName, SOPInstanceUID, StationName, 'Marker is Right, but GCRS request is Left')
                                #else:
                                    #self.handle_problem_image(original_img, AccessionNumber, PatientName, SOPInstanceUID, StationName, 'Reminder: Both side image. Marker is Right, but GCRS request is Left')
                        else:
                            if any('LEFT' in result.upper() for result in final_marker_result) and ('Right' in record_laterality):
                                self.handle_problem_image(original_img, AccessionNumber, PatientName, SOPInstanceUID, StationName, 'Marker is Left, but GCRS request is Right')
                            elif any('RIGHT' in result.upper() for result in final_marker_result) and ('Left'  in record_laterality):
                                self.handle_problem_image(original_img, AccessionNumber, PatientName, SOPInstanceUID, StationName, 'Marker is Right, but GCRS request is Left')
        
                row_count = session.query(Order_Detail).count()
                if row_count > 5000:
                    first_record = session.query(Order_Detail).order_by(Order_Detail.Id.asc()).first()
                    session.delete(first_record)
                
        else:
            now = datetime.now()
            time_string = now.strftime("%Y-%m-%d %H:%M:%S")
            print('Essential DICOM tag not found!!!')
            self.post_received_image_record(remarks='Essentail dicom tag not existed! '+ time_string)

                    
        print('\n\n**************** finish scanning ****************\n\n')                    


def main():
    dir_path = Path(__file__).parent.absolute()
    os.chdir(str(dir_path))
    
    if not os.path.exists(str(dir_path) + "/no_marker_images/"):
        os.makedirs(str(dir_path) + "/no_marker_images/")
    if not os.path.exists(str(dir_path) + "/problem_images/"):
        os.makedirs(str(dir_path) + "/problem_images/")
    if not os.path.exists(str(dir_path) + '/evaluate_result/'):
        os.makedirs(str(dir_path) + '/evaluate_result/')

    dicom_analyzer = DicomImageAnalyzer()

    try:
        while True:
            dicom_pathlist = Path(DICOM_DIR).glob('**/*')
            for path in dicom_pathlist:
                dicom_path = str(path)
                print(dicom_path)
                img_path = dicom_path.replace('dicom', 'image')+'.png'
                input_img = cv2.imread(img_path)
                if 'KO' not in str(path.stem):
                    if Path(dicom_path).is_file() and input_img is not None:
                        dicom_analyzer.new_instance_received(dicom_path, input_img)
                        os.remove(dicom_path)
                        os.remove(img_path)
                    else:
                        print('execute dcmj2pnm')
                        dcmj2pnm_command ='dcmj2pnm --write-png {} {}'.format(dicom_path, img_path)
                        print(dcmj2pnm_command)
                        args = shlex.split(dcmj2pnm_command)
                        p = subprocess.Popen(args)
                        
                else:
                    os.remove(dicom_path)
                if 'PS' in str(path.stem) or 'SR' in str(path.stem):
                    os.remove(dicom_path)
            time.sleep(3)
    except KeyboardInterrupt:
        raise
    except:
        print(traceback.format_exc())
        record = Program_Error(Error_Detail = traceback.format_exc())
        session.add(record)
        session.commit()
        pass
        
if __name__ == '__main__':
    main()
    





