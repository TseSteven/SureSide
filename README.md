# SureSide - Wrong side marker detection system for radiographic image


A simple radiological quality control applcation to detect falsely applied annotation on radiological image.
SureSide is a client server application with DICOM capability. 
By setting up auto DICOM routing from your hospital DICOM server to SureSdie server, radiological image will be analyzed and false marker image will be spotted out.

The following open source projects are used for SureSide project

- DCMTK
- OpenCV
- GluonCV
- RabbitMQ
- NextGen Connect
- Postegresql
- Redis
- Cockpit
- Cloud9

### [Server Installation]

1. Download the Virtualbox ova file 'DR_QC_buster GitHub.ova' from  https://sourceforge.net/projects/sureside/files/
  
2. Import the ova and start the virtual server.

3. Push DICOM image to virtual DICOM server with AE Title "SURESIDE"

4. See result from YOUR_VIRTUAL_SERVER_IP:5002


### [Client Installation]

Windows python clients with filename 'DR Warnings Client.zip' can be downloaded from https://sourceforge.net/projects/sureside/files/

Portable windows python are included inside 'DR Warnings Client.zip', just unzip and run the batch script 'DR Warnings client.bat' to start.




