import psycopg2
import os
from dotenv import load_dotenv

'''
CREATE TABLE alerts (
    id SERIAL PRIMARY KEY,
    person_name VARCHAR,
    gender VARCHAR,
    race VARCHAR,
    age INT,
    car_color VARCHAR,
    car_plate VARCHAR,
    car_make VARCHAR,
    car_model VARCHAR,
    car_year VARCHAR,
    created_at TIMESTAMP NOT NULL,
    location VARCHAR,
    geog GEOGRAPHY
)
'''

def connect():
    load_dotenv()
    conn = psycopg2.connect(os.getenv('DB_URL'))
    return conn
