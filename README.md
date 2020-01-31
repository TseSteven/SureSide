# SureSide
Wrong side marker detection system for radiographic image

Easy way to test the application:

1. Download the Virtualbox ova file DR_QC_buster GitHub.ova from  https://sourceforge.net/projects/sureside/files/
  
2. Import the ova and start the virtual server.

3. Push DICOM image to virtual DICOM server with AE Title "SURESIDE"

4. See result from YOUR_VIRTUAL_IP:5002


Difficult way

Download the git repository with command git clone.

Download the files from https://sourceforge.net/projects/sureside/files/

1. anatomy_classification.params
2. frozen_east_text_detection.pb
3. marker_detection.params

and place the files inside folder DR_Warnings/server/app_server/false_annotation/

You may download the client for the alert window pop up with file DR Warnings Client.zip from https://sourceforge.net/projects/sureside/files/




