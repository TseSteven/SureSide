# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from hl7apy.core import Message
from hl7apy.parser import parse_message
from hl7apy.mllp import AbstractHandler
from hl7apy.mllp import AbstractErrorHandler
from hl7apy.mllp import MLLPServer
from hl7apy.mllp import UnsupportedMessageType

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker
from sqlalchemy import exc

from os.path import dirname, realpath

from datetime import datetime
import traceback

engine_app = create_engine('postgresql://rad:rad@localhost:5432/rad_db')

Base = declarative_base()

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
    

class GCRS_Laterality(Base):
    __tablename__ = 'gcrs_laterality'
    
    Id = Column(Integer, autoincrement=True, primary_key=True)
    Record_Date_Time = Column(DateTime, nullable=False,default=datetime.now)
    Accession_Number = Column(String)
    Region = Column(String)
    Laterality = Column(String)
    
class Medical_Report(Base):
    __tablename__ = "medical_report"
 
    Id = Column(Integer, primary_key=True)
    Record_Date_Time = Column(DateTime, nullable=False,default=datetime.now)
    Order_Number = Column(String)
    Accession_Number = Column(String)
    Report_ID = Column(String)
    Report_Text = Column(String)
    Result_Status = Column(String)

Base.metadata.create_all(bind=engine_app)


class ORM_O01_Handler(AbstractHandler):
    def reply(self):
        msg = parse_message(self.incoming_message, find_groups=False)
        # do something with the message
        
        now_time = datetime.now()
        now_str = now_time.strftime("%Y-%m-%d %H:%M:%S")
        print(now_str + ' # '+'Received new ORM O01 message!!')
        
        order_control = ''
        order_number = ''
        order_status = ''
        accession_number = ''
        region_code = ''
        laterality = ''
        radiologist_name = ''
        urgency = ''
        order_control = msg.orc.orc_1.value
        print('Order Control:'+order_control)
        if order_control == 'NW':
            for child in msg.obr:
                accession_number = child.obr_18.value + '-' + child.obr_1.value
                region_code = child.obr_4.value
                laterality = child.obr_15.value
                
                if len(laterality) >0:
                    record = GCRS_Laterality(Accession_Number = accession_number, Region = region_code, Laterality = laterality)
                    Session_app = sessionmaker(bind=engine_app)
                    session_app = Session_app()
                    session_app.add(record)
                    session_app.commit()
                    session_app.close()
                            
        order_number = msg.orc.orc_2.value
        print('Order Number:'+order_number)
        
        for child in msg.obr:
            accession_number = child.obr_18.value + '-' + child.obr_1.value
            print('Acc Num:'+accession_number)
            region_code = child.obr_4.value
            print('Reg Code:'+region_code)
            laterality = child.obr_15.value
            print('Lat:'+laterality)
            radiologist_name = child.obr_32.value
            print('Rad Name:'+radiologist_name)
            urgency = child.obr_5.value
            print('Urgency:'+urgency)
            exam_room = child.obr_19.value
            print('Exam Room:'+exam_room)
            
            record_order_detial = Order_Detail(Accession_Number = accession_number, Region = region_code, Laterality = laterality, Radiologist_Name=radiologist_name, Urgency=urgency,Order_Control = order_control,Order_Number = order_number,Order_Status = order_status, Exam_Room = exam_room)
            #if accession_number.startswith('POH'):
            
            Session_app = sessionmaker(bind=engine_app)
            session_app = Session_app()
            session_app.add(record_order_detial)
            session_app.commit()
            session_app.close()

        res = Message('ACK')
        # populate the message
        return res.to_mllp()

class ORU_R01_Handler(AbstractHandler):
    def reply(self):
        obx_start_pos = self.incoming_message.find('OBX|')
        if obx_start_pos > 0:
            new_string = self.incoming_message[obx_start_pos:].replace('\r', '@#@')
            message = self.incoming_message[:obx_start_pos]+new_string
        else:
            message = self.incoming_message
        msg = parse_message(message, find_groups=False)
        # do something with the message
        now_time = datetime.now()
        now_str = now_time.strftime("%Y-%m-%d %H:%M:%S")
        print(now_str + ' # '+'Received new ORU R01 message!!')
        
        accession_number = msg.obr.obr_18.to_er7()
        report_id = msg.obr.obr_26_to_er7()
        report_text = msg.obx.obx_5.to_er7()
        result_status = msg.obx.obx_11.to_er7()
        
        record = Medical_Report(Accession_Number = accession_number, Report_ID = report_id ,Report_Text = report_text, Result_Status = result_status)
        
        Session_app = sessionmaker(bind=engine_app)
        session_app = Session_app()
        session_app.add(record)
        session_app.commit()
        session_app.close()
                            
        
        #~ with open('ORU_R01' + msg.msh.msh_7.to_er7() + ".txt", "w") as text_file:
            #~ text_file.write(msg.to_er7())
            
        res = Message('ACK')
        # populate the message
        return res.to_mllp()
        
class ADT_A08_Handler(AbstractHandler):
    def reply(self):
        msg = parse_message(self.incoming_message, find_groups=False)
        # do something with the message
        
        now_time = datetime.now()
        now_str = now_time.strftime("%Y-%m-%d %H:%M:%S")
        print(now_str + ' # '+'Received new ADT A08 message!!')

        #hkid = msg.pid.pid_3.to_er7()
        #allergen_type = msg.al1.al1_2.to_er7()
        #allergy_reaction = msg.al1.al1_5.to_er7()
        
        #record = ADT_A08(HKID = hkid, Allergen_Type = allergen_type, Allergy_Reaction = allergy_reaction)
        #session.add(record)
        #session.commit()
        
            
        res = Message('ACK')
        # populate the message
        return res.to_mllp()
        
class ADT_A47_Handler(AbstractHandler):
    def reply(self):
        msg = parse_message(self.incoming_message, find_groups=False)
        # do something with the message
        now_time = datetime.now()
        now_str = now_time.strftime("%Y-%m-%d %H:%M:%S")
        print(now_str + ' # '+'Received new ADT A47 message!!')
        
        #with open('ADT_A47_' + msg.msh.msh_7.to_er7() + ".txt", "w") as text_file:
            #text_file.write(msg.to_er7())

        
        res = Message('ACK')
        # populate the message
        return res.to_mllp()

class ADT_A40_Handler(AbstractHandler):
    def reply(self):
        msg = parse_message(self.incoming_message, find_groups=False)
        # do something with the message

        now_time = datetime.now()
        now_str = now_time.strftime("%Y-%m-%d %H:%M:%S")
        print(now_str + ' # '+'Received new ADT A40 message!!')
        
        #with open('ADT_A40_' + msg.msh.msh_7.to_er7() + ".txt", "w") as text_file:
            #text_file.write(msg.to_er7())
            
        #with open('ADT_A40_' + now_str + ".txt", "w") as text_file:
            #text_file.write(msg.to_er7())
            
        res = Message('ACK')
        # populate the message
        return res.to_mllp()
        
class HL7ErrorHandler(AbstractErrorHandler):
    def __init__(self, exc, incoming_message):
        super(HL7ErrorHandler, self).__init__(incoming_message)
        self.exc = exc

    def reply(self):

        if isinstance(self.exc, UnsupportedMessageType):
            err_code, err_msg = 101, 'Unsupported message'
            print('Unsupported message')
        elif isinstance(self.exc, ParserError):
            err_code, err_msg = 101, 'Parser Error'
            print('Parser Error')
        elif isinstance(self.exc, InvalidHL7Message):
            err_code, err_msg = 102, 'Incoming message is not an HL7 valid message'
            print('Incoming message is not an HL7 valid message')
        else:
            err_code, err_msg = 100, 'Unknown error occurred'
            print('Unknown error occurred')

        parsed_message = parse_message(self.incoming_message)

        m = Message("ACK")
        m.MSH.MSH_9 = "ACK^ACK"
        m.MSA.MSA_1 = "AR"
        m.MSA.MSA_2 = parsed_message.MSH.MSH_10
        m.ERR.ERR_1 = "%s" % err_code
        m.ERR.ERR_2 = "%s" % err_msg

        return m.to_mllp()

    def reply(self):
        print (str(self.exc))
        if isinstance(self.exc, ParserError):
            # return your custom response for unsupported message
            res = Message('ACK')
            pass
        return res.to_mllp()
        

handlers = {
    'ORM^O01^ORM_O01': (ORM_O01_Handler,), # Appointment and Order
    'ORU^R01^ORU_R01': (ORU_R01_Handler,), # Report
    'ADT^A08': (ADT_A08_Handler,), # Patient information update e.g. Allergy
    'ADT^A47': (ADT_A47_Handler,), # HKID update
    'ADT^A40': (ADT_A40_Handler,),  # Merge Patient ID
    'ERR': (HL7ErrorHandler,)
}

print('HL7 MLLP server started!')
server = MLLPServer('0.0.0.0', 2311, handlers)
#server = MLLPServer('0.0.0.0', 2310, handlers)
server.serve_forever()
