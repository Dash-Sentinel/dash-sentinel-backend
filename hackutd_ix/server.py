from flask import Flask
import db_ops

app = Flask(__name__)

@app.get("/alerts")
def hello_world():
    conn = db_ops.connect()
    cur = conn.cursor()
    cur.execute('SELECT * FROM alerts ORDER BY id DESC')
    cur.execute('''SELECT json_build_object(
    'id',          id,
    'person_name', person_name,
    'gender',      gender,
    'race',        race,
    'age',         age,
    'car_color',   car_color,
    'car_plate',   car_plate,
    'car_make',    car_make,
    'car_model',   car_model,
    'car_year',    car_year,
    'created_at',  created_at,
    'location',    location,
    'geometry',    ST_AsGeoJSON(geog)::json
 )
 FROM alerts;''')
    return cur.fetchall()
