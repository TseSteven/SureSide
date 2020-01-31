import os
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory
from flask_sqlalchemy import SQLAlchemy

from datetime import timedelta, date
from datetime import datetime

app = Flask(__name__)
app.config['DEBUG'] = True

PROGRAM_DIRECTORY = '/home/rad/app/DR_Warnings/server/web_server/warning_web_server/static/program'
app.config["PROGRAM"] = PROGRAM_DIRECTORY
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://rad:rad@localhost:5432/rad_db'
#app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////home/rad/app/DR_Warnings/server/web_server/warning_web_server/web_server.db'
db = SQLAlchemy(app)

class Program_Error(db.Model):
    __tablename__ = 'program_error'
    
    Id = db.Column(db.Integer, primary_key=True)
    Record_Date_Time = db.Column(db.DateTime, nullable=False,default=datetime.now)
    Error_Detail = db.Column(db.String)

class Received_Image(db.Model):
    __tablename__ = "received_image"
 
    Id = db.Column(db.Integer, primary_key=True)
    Record_Date_Time = db.Column(db.DateTime, nullable=False,default=datetime.now)
    Station_Name = db.Column(db.String)
    Accession_Number = db.Column(db.String)
    SOP_Instance_UID = db.Column(db.String)
    Study_Description = db.Column(db.String)
    Flip = db.Column(db.String)
    Anatomy_Detected = db.Column(db.String)
    Digital_Marker_Used = db.Column(db.String)
    Physical_Marker_Used = db.Column(db.String)
    Remarks = db.Column(db.String)

class Delay_Upload(db.Model):
    __tablename__ = 'delay_upload'
    
    Id = db.Column(db.Integer, autoincrement=True, primary_key=True)
    Record_Date_Time = db.Column(db.DateTime, nullable=False,default=datetime.now)
    Msg_ID = db.Column(db.String)
    Accession_Number = db.Column(db.String)
    Patient_Name = db.Column(db.String)
    Exam_Room = db.Column(db.String)
    User_Reply = db.Column(db.String)
    
class False_Annotation(db.Model):
    __tablename__ = 'false_annotation'
    
    Id = db.Column(db.Integer, autoincrement=True, primary_key=True)
    Record_Date_Time = db.Column(db.DateTime, nullable=False,default=datetime.now)
    Msg_ID = db.Column(db.String)
    Accession_Number = db.Column(db.String)
    Patient_Name = db.Column(db.String)
    Exam_Room = db.Column(db.String)
    SOP_Instance_UID = db.Column(db.String)
    Problem_Detail = db.Column(db.String)
    User_Reply = db.Column(db.String)

class Order_Detail(db.Model):
    __tablename__ = 'order_detail'
    
    Id = db.Column(db.Integer, autoincrement=True, primary_key=True)
    Record_Date_Time = db.Column(db.DateTime, nullable=False,default=datetime.now)
    Accession_Number = db.Column(db.String)
    Region = db.Column(db.String)
    Laterality = db.Column(db.String)
    Radiologist_Name = db.Column(db.String)
    Urgency = db.Column(db.String)
    Order_Control = db.Column(db.String)
    Order_Number = db.Column(db.String)
    Order_Status = db.Column(db.String)
    Exam_Room = db.Column(db.String)

db.create_all()

# input_row = Delay_Upload(Msg_ID='msg-123', Patient_Name='Chan Tai Man', Accession_Number='poh-123')
# db.session.add(input_row)
# input_row_2 = Delay_Upload(Msg_ID='msg-232', Patient_Name='Lee Siu Man', Accession_Number='poh-664')
# db.session.add(input_row_2)

# db.session.commit()

@app.route('/', methods=['GET', 'POST'])
def home_page():
    if request.method == 'GET':
        return render_template('/app/index.html')
    else:
        if request.form['mybutton'] == 'received_image_record':
            return redirect(url_for('received_image_record'))
        elif request.form['mybutton'] == 'false_annotation_record':
            return redirect(url_for('false_annotation_record'))
        elif request.form['mybutton'] == 'delay_upload_record':
            return redirect(url_for('delay_upload_record'))
        elif request.form['mybutton'] == 'hl7_record':
            return redirect(url_for('hl7_record'))          
        elif request.form['mybutton'] == 'problem_record':
            return redirect(url_for('problem_record'))
        else:
            return redirect(url_for('home_page'))

@app.route('/warning', methods=['GET', 'POST'])
def warning():
    if request.method == 'POST':
        user_reply = request.form.get('user_reply')
        msg_id = request.form.get('msg_id')
        warning_type = request.form.get('warning_type')
        if warning_type == 'delay_upload':
            results = Delay_Upload.query.filter_by(Msg_ID=msg_id).all()
        elif warning_type == 'false_annotation':
            results = False_Annotation.query.filter_by(Msg_ID=msg_id).all()
            
        for item in results:
            item.User_Reply = user_reply
        db.session.commit()
        
        return render_template('user_reply.html')
    else:
        msg_id = request.args.get('msg_id')
        warning_type = request.args.get('warning_type')
        num = request.args.get('num')
        if warning_type == 'delay_upload':
            title = 'Image Upload Delayed'
            results= Delay_Upload.query.filter_by(Msg_ID=msg_id).all()
        elif warning_type == 'false_annotation':
            img_id = request.args.get('img_id')
            title = 'Image Falsely Annotated'
            results= False_Annotation.query.filter_by(Msg_ID=msg_id).all()
        data=[]
        for item in results:
            data.append({'patient_name':item.Patient_Name,'acc_num':item.Accession_Number, 'exam_room':item.Exam_Room})
        columns = [{"field": "patient_name", "title": "Patient Name"}, {"field": "acc_num", "title": "Accession Number"},{"field": "exam_room", "title": "Exam Room"}]
        
        if warning_type =='delay_upload':
            return render_template('/app/warning.html',data=data, columns=columns, msg_id=msg_id, num=num, title=title, warning_type = warning_type)
        elif warning_type == 'false_annotation':
            return render_template('/app/warning.html',data=data, columns=columns, msg_id=msg_id, num=num, title=title, warning_type = warning_type, img_id = img_id)

@app.route('/delay_upload_record', methods=['GET'])
def delay_upload_record():
    results= Delay_Upload.query.order_by(Delay_Upload.Id.desc()).limit(500).all()
    data=[]
    columns = []
    def row_to_dict(row):
        result = {}
        for column in row.__table__.columns:
            result[column.name] = str(getattr(row, column.name))
        return result
    for item in results:
        data.append(row_to_dict(item))
    
    columns_name =Delay_Upload.__table__.columns.keys()
    for item in columns_name:
        columns.append({"field": item, "title": item, "sortable": True})
    return render_template('/record/delay_upload_record.html',data=data, columns=columns)
    
@app.route('/received_image_record', methods=['GET'])
def received_image_record():
    results= Received_Image.query.order_by(Received_Image.Id.desc()).limit(500).all()
    data=[]
    columns = []
    def row_to_dict(row):
        result = {}
        for column in row.__table__.columns:
            result[column.name] = str(getattr(row, column.name))
        return result
    for item in results:
        data.append(row_to_dict(item))
    
    columns_name =Received_Image.__table__.columns.keys()
    for item in columns_name:
        if item != 'SOP_Instance_UID':
            columns.append({"field": item, "title": item, "sortable": True})
    return render_template('/record/received_image_record.html',data=data, columns=columns)

@app.route('/hl7_record', methods=['GET'])
def hl7_record():
    results= Order_Detail.query.order_by(Order_Detail.Id.desc()).limit(500).all()
    data=[]
    columns = []
    def row_to_dict(row):
        result = {}
        for column in row.__table__.columns:
            result[column.name] = str(getattr(row, column.name))
        return result
    for item in results:
        data.append(row_to_dict(item))
    
    columns_name =Order_Detail.__table__.columns.keys()
    for item in columns_name:
        columns.append({"field": item, "title": item, "sortable": True})
    return render_template('/record/hl7_record.html',data=data, columns=columns)

@app.route('/false_annotation_record', methods=['GET'])
def false_annotation_record():
    results= False_Annotation.query.order_by(False_Annotation.Id.desc()).limit(500).all()
    data=[]
    columns = []
    def row_to_dict(row):
        result = {}
        for column in row.__table__.columns:
            result[column.name] = str(getattr(row, column.name))
        return result
    for item in results:
        data.append(row_to_dict(item))
    
    columns_name =False_Annotation.__table__.columns.keys()
    for item in columns_name:
        if item != 'SOP_Instance_UID':
            columns.append({"field": item, "title": item, "sortable": True})
    return render_template('/record/false_annotation_record.html',data=data, columns=columns)

@app.route('/problem', methods=['GET'])
def problem_record():
    results= Program_Error.query.order_by(Program_Error.Id.desc()).limit(500).all()
    data=[]
    columns = []
    def row_to_dict(row):
        result = {}
        for column in row.__table__.columns:
            result[column.name] = str(getattr(row, column.name))
        return result
    for item in results:
        data.append(row_to_dict(item))
    
    columns_name =Program_Error.__table__.columns.keys()
    for item in columns_name:
        columns.append({"field": item, "title": item, "sortable": True})
    return render_template('/record/problem_record.html',data=data, columns=columns)


@app.route("/files")
def list_files():
    """Endpoint to list files on the server."""
    files = []
    for filename in os.listdir(PROGRAM_DIRECTORY):
        path = os.path.join(PROGRAM_DIRECTORY, filename)
        if os.path.isfile(path):
            files.append(filename)
    return jsonify(files)


@app.route("/get-program/<file_name>")
def get_program(file_name):

    try:
        return send_from_directory(app.config["PROGRAM"], filename=file_name, as_attachment=True)
    except FileNotFoundError:
        abort(404)

if __name__ == '__main__':
    pdf_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),"static","pdf")
    if not os.path.exists(pdf_path):
        os.makedirs(pdf_path)
        
    delay_upload_pdf_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),"static","pdf", "delay_upload")
    if not os.path.exists(delay_upload_pdf_path):
        os.makedirs(delay_upload_pdf_path)
        
    false_annotation_pdf_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),"static","pdf", "false_annotation")
    if not os.path.exists(false_annotation_pdf_path):
        os.makedirs(false_annotation_pdf_path)
        
    #app.run()
    app.run(host='0.0.0.0',port=5002, threaded=True)
    #app.run(host='0.0.0.0',port=80, threaded=True)
