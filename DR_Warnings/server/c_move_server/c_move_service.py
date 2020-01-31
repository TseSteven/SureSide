import os
import sys
import time
import socket
import random

from pydicom.dataset import Dataset

from pynetdicom import (AE, QueryRetrievePresentationContexts)

from datetime import datetime
from datetime import timedelta, date

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker

import redis

import traceback

AE_TITLE = 'SURESIDE'
PEER_AE_TITLE = 'YOUR_DICOM_SERVER_AE_TITLE'
PEER_IP_ADDRESS = 'YOUR_DICOM_SERVER_IP'

PEER_PORT = 104
CMOVE_AE_TITLE = 'SURESIDE'

NODE_LIST = []
NODE_LIST.append('localhost')

### uncomment and edit following parts if clustering is needed
#NODE_LIST.append('160.58.150.64')
#NODE_LIST.append('160.8.12.233')
#NODE_LIST.append('160.59.234.99')

redis_client_list = []
local_redis = None

for item in NODE_LIST:
    redisClient = redis.StrictRedis(host=item, port=6379, db=0)
    redis_client_list.append(redisClient)
    if item == 'localhost':
        local_redis = redisClient

Base = declarative_base()

class Program_Error(Base):
    __tablename__ = 'program_error'
    
    Id = Column(Integer, autoincrement=True, primary_key=True)
    Record_Date_Time = Column(DateTime, nullable=False,default=datetime.now)
    Error_Detail = Column(String)

engine = create_engine('postgresql://rad:rad@localhost:5432/rad_db')

Base.metadata.create_all(bind=engine)
Session = sessionmaker(bind=engine)
session = Session()

def c_find_pacs(date_of_query, time_of_query):
    # QueryRetrieveSOPClassList contains the SOP Classes supported
    #   by the Query/Retrieve Service Class (see PS3.4 Annex C.6)
    #ae = AE(ae_title=my_title,scu_sop_class=QueryRetrieveSOPClassList)
    
    ae = AE(ae_title=AE_TITLE)
    ae.requested_contexts = QueryRetrievePresentationContexts

    # Try and associate with the peer AE
    #   Returns the Association thread
    print('Requesting Association with the peer for C-FIND')
    assoc = ae.associate(PEER_IP_ADDRESS, int(PEER_PORT), ae_title=PEER_AE_TITLE)

    result_list = []

    if assoc.is_established:
        print('Association accepted by the peer')

        # Creat a new DICOM dataset with the attributes to match against
        #   In this case match any patient's name at the PATIENT query
        #   level. See PS3.4 Annex C.6 for the complete list of possible
        #   attributes and query levels.
        dataset = Dataset()
        dataset.ModalitiesInStudy = 'DX\CR'
        dataset.AccessionNumber = ''
        dataset.StudyInstanceUID = ''
        #dataset.SeriesInstanceUID = ''
        #dataset.SOPInstanceUID = ''
        
        dataset.StudyDate = date_of_query
        dataset.StudyTime = time_of_query
        dataset.QueryRetrieveLevel = "STUDY"

        # Send a DIMSE C-FIND request to the peer
        #   query_model is the Query/Retrieve Information Model to use
        #   and is one of 'W', 'P', 'S', 'O'
        #       'W' - Modality Worklist (1.2.840.10008.5.1.4.31)
        #       'P' - Patient Root (1.2.840.10008.5.1.4.1.2.1.1)
        #       'S' - Study Root (1.2.840.10008.5.1.4.1.2.2.1)
        #       'O' - Patient/Study Only (1.2.840.10008.5.1.4.1.2.3.1)
        responses = assoc.send_c_find(dataset, query_model='S')
        
        for (status, dataset) in responses:
            print (str(status))
            #if 'Pending' in str(status):
            if dataset:
                if dataset.ModalitiesInStudy and dataset.AccessionNumber and dataset.StudyInstanceUID and dataset.StudyDate and dataset.StudyTime:
                    print ('Modality:' +str(dataset.ModalitiesInStudy))
                    print ('Accession Number:' +str(dataset.AccessionNumber))
                    print ('Study Instance UID:' +str(dataset.StudyInstanceUID))
                    print ('Study Date:' +str(dataset.StudyDate))
                    print ('Study Time:' +str(dataset.StudyTime))
                    #print ('Series Instance UID:' +str(dataset.SeriesInstanceUID))
                    #print('SOP Instance UID:'+ str(dataset.SOPInstanceUID))
                    
                    
                    result_list.append(dataset.StudyInstanceUID)
                        
            #except Exception as e:
                #print('Exception: '+ str(e))
                #pass

        # Release the association
        assoc.release()
    return result_list

def c_find_pacs_series(study_uid):
    # QueryRetrieveSOPClassList contains the SOP Classes supported
    #   by the Query/Retrieve Service Class (see PS3.4 Annex C.6)
    #ae = AE(ae_title=my_title,scu_sop_class=QueryRetrieveSOPClassList)
    
    ae = AE(ae_title=AE_TITLE)
    ae.requested_contexts = QueryRetrievePresentationContexts
    result_list = []

    # Try and associate with the peer AE
    #   Returns the Association thread
    print('Requesting Association with the peer for C-FIND')
    assoc = ae.associate(PEER_IP_ADDRESS, int(PEER_PORT), ae_title=PEER_AE_TITLE)

    if assoc.is_established:
        print('Association accepted by the peer')

        # Creat a new DICOM dataset with the attributes to match against
        #   In this case match any patient's name at the PATIENT query
        #   level. See PS3.4 Annex C.6 for the complete list of possible
        #   attributes and query levels.
        dataset = Dataset()
        dataset.StudyInstanceUID = study_uid
        dataset.SeriesInstanceUID = ''
        #dataset.SOPInstanceUID = ''
        
        dataset.QueryRetrieveLevel = "SERIES"

        # Send a DIMSE C-FIND request to the peer
        #   query_model is the Query/Retrieve Information Model to use
        #   and is one of 'W', 'P', 'S', 'O'
        #       'W' - Modality Worklist (1.2.840.10008.5.1.4.31)
        #       'P' - Patient Root (1.2.840.10008.5.1.4.1.2.1.1)
        #       'S' - Study Root (1.2.840.10008.5.1.4.1.2.2.1)
        #       'O' - Patient/Study Only (1.2.840.10008.5.1.4.1.2.3.1)
        responses = assoc.send_c_find(dataset, query_model='S')
        
        for (status, dataset) in responses:
            #print (str(status))
            #if 'Pending' in str(status):
            if dataset:
                if dataset.SeriesInstanceUID:
                    #print ('Series Instance UID:' +str(dataset.SeriesInstanceUID))
                    #print('SOP Instance UID:'+ str(dataset.SOPInstanceUID))
                    
                    result_dict = {'study_uid': study_uid, 'series_uid': dataset.SeriesInstanceUID}
                    result_list.append(result_dict)
                        
            #except Exception as e:
                #print('Exception: '+ str(e))
                #pass

        # Release the association
        assoc.release()
    return result_list

def c_find_pacs_image(study_uid, series_uid):
    # QueryRetrieveSOPClassList contains the SOP Classes supported
    #   by the Query/Retrieve Service Class (see PS3.4 Annex C.6)
    #ae = AE(ae_title=my_title,scu_sop_class=QueryRetrieveSOPClassList)
    
    ae = AE(ae_title=AE_TITLE)
    ae.requested_contexts = QueryRetrievePresentationContexts
    result_list = []

    # Try and associate with the peer AE
    #   Returns the Association thread
    print('Requesting Association with the peer for C-FIND')
    assoc = ae.associate(PEER_IP_ADDRESS, int(PEER_PORT), ae_title=PEER_AE_TITLE)

    if assoc.is_established:
        print('Association accepted by the peer')

        # Creat a new DICOM dataset with the attributes to match against
        #   In this case match any patient's name at the PATIENT query
        #   level. See PS3.4 Annex C.6 for the complete list of possible
        #   attributes and query levels.
        dataset = Dataset()
        dataset.StudyInstanceUID = study_uid
        dataset.SeriesInstanceUID = series_uid
        dataset.SOPInstanceUID = ''
        
        dataset.QueryRetrieveLevel = "IMAGE"

        # Send a DIMSE C-FIND request to the peer
        #   query_model is the Query/Retrieve Information Model to use
        #   and is one of 'W', 'P', 'S', 'O'
        #       'W' - Modality Worklist (1.2.840.10008.5.1.4.31)
        #       'P' - Patient Root (1.2.840.10008.5.1.4.1.2.1.1)
        #       'S' - Study Root (1.2.840.10008.5.1.4.1.2.2.1)
        #       'O' - Patient/Study Only (1.2.840.10008.5.1.4.1.2.3.1)
        responses = assoc.send_c_find(dataset, query_model='S')
        
        for (status, dataset) in responses:
            #print (str(status))
            #if 'Pending' in str(status):
            if dataset:
                if dataset.SOPInstanceUID:
                    #print('SOP Instance UID:'+ str(dataset.SOPInstanceUID))
                    result_dict = {'study_uid': study_uid, 'series_uid': dataset.SeriesInstanceUID, 'img_uid':dataset.SOPInstanceUID}
                    result_list.append(result_dict)
                        
            #except Exception as e:
                #print('Exception: '+ str(e))
                #pass

        # Release the association
        assoc.release()
    return result_list

def c_move_pacs(study_uid, series_uid, img_uid):
    
    ae = AE(ae_title=AE_TITLE)
    ae.requested_contexts = QueryRetrievePresentationContexts
    print('Requesting Association with the peer for C-MOVE')
    assoc = ae.associate(PEER_IP_ADDRESS, int(PEER_PORT),ae_title=PEER_AE_TITLE)

    if assoc.is_established:
        print('Association accepted by the peer')

        dataset = Dataset()
        dataset.StudyInstanceUID = study_uid
        dataset.SeriesInstanceUID = series_uid
        dataset.SOPInstanceUID = img_uid
        dataset.QueryRetrieveLevel = "IMAGE"
        
        
        responses = assoc.send_c_move(dataset, CMOVE_AE_TITLE,query_model='S')

        for (status, dataset) in responses:
            #print (str(status))
            pass
            
        # Release the association
        assoc.release()

def main():
    ##
    ## Main loop for C-Move.
    ## 


    while True:
        try:
            now = datetime.now()
            print(now)
            fifteenth_min_before_time = now - timedelta(minutes=15)
            query_date = fifteenth_min_before_time.strftime("%Y%m%d")
            query_time = fifteenth_min_before_time.strftime("%H%M%S") + '-' + now.strftime("%H%M%S")
            print(query_date)
            print(query_time)
            
            series_uid_list = []
            img_uid_list = []
            study_result = c_find_pacs(query_date, query_time)
            for item in study_result:
                series_result = c_find_pacs_series(item)
                series_uid_list.extend(series_result)
            for item in series_uid_list:
                img_result = c_find_pacs_image(item['study_uid'], item['series_uid'])
                img_uid_list.extend(img_result)
            
            
            for item in img_uid_list:
                image_exist_list = []
                for redis_client in redis_client_list:
                    try:
                        redis_exist =redis_client.sismember('instance_uid_set', str(item['img_uid']))
                        image_exist_list.append(redis_exist)
                    except:
                        print('Redis connection error')
                        pass
                print(image_exist_list)

                if any(image_exist_list):
                    print("Skip cmove, SOP Instance UID exists in redis record!")
                else:
                    c_move_pacs(item['study_uid'], item['series_uid'], item['img_uid'])
                    local_redis.sadd('instance_uid_set', str(item['img_uid']))
                
            # cmove_list = []
    
            # for key, value in cfind_result.items():
            #     study_instance_uid = key
            #     number_of_image = value
            #     combined = study_instance_uid + '_' + number_of_image
                
            #     study_exist_list = []
                
            #     for item in session_list:
            #         session = item['session']
            #         exists_at_record = session.query(exists().where(CFIND_Study_UID.Study_UID == combined)).scalar()
            #         study_exist_list.append(exists_at_record)
            #         session.close()
    
            #     if not any(study_exist_list):
            #         cmove_list.append(study_instance_uid)
                    
            # for item in cmove_list:
            #     c_move_pacs(item)
            
    
            # for key, value in cfind_result.items():
            #     study_instance_uid = key
            #     number_of_image = value
            #     combined = study_instance_uid + '_' + number_of_image
            #     for item in session_list:
            #         session =item['session']
            #         if item['ip_address'] == 'localhost':
            #             input_row = CFIND_Study_UID(Study_UID=combined)
            #             session.add(input_row)
            #             session.commit()
        except:
            print(traceback.format_exc())
            record = Program_Error(Error_Detail = traceback.format_exc())
            session.add(record)
            session.commit()
            pass
        sleep_time = random.randint(10, 60) 
        time.sleep(sleep_time)
        
if __name__ == '__main__':
    main()
