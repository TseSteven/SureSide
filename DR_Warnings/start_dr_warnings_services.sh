#!/bin/sh

python3 /home/rad/app/DR_Warnings/server/app_server/false_annotation/check_false_annotation.py
python3 /home/rad/app/DR_Warnings/server/web_server/warning_web_server/web_server.py


## uncomment following line to enable c_move service
#python3 /home/rad/app/DR_Warnings/server/c_move_server/c_move_service.py