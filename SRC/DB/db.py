import mysql.connector
from mysql.connector import Error
import pandas as pd
from config import Config
import os
import csv

class DB:
  def create_server_connection():
      connection = None
      try:
          connection = mysql.connector.connect(
              pool_name=Config.DB_POOL,
              host=Config.DB_HOST,
              user=Config.DB_USER,
              passwd=os.environ.get("DB_PASSWORD"),
              database=Config.DB_database_name
          )
          print("MySQL Database connection successful")
      except Error as err:
          print(f"Error: '{err}'")

      return connection

  def create_database(connection, query):
      cursor = connection.cursor()
      try:
          cursor.execute(query)
          print("Database created successfully")
      except Error as err:
          print(f"Error: '{err}'")

  def execute_query(query):
    connection = DB.create_server_connection()
    try:
      with connection.cursor() as cursor:
        #query = "INSERT INTO posts(post_details, sentiment_value) VALUES (%s,%d)"
        # Read all records
        cursor.execute(query)
        connection.commit()
        print("Query successful")

    finally:
      connection.close()

  def create_table(connection):
    create_posts_table = """
      CREATE TABLE posts (
        post_id INT PRIMARY KEY,
        details VARCHAR(40) NOT NULL,
        date_of_post DATE
        );
      """
    DB.execute_query(connection, create_posts_table) # Execute our defined query

  def seed_data(data):
    pop_posts = ("""
    INSERT INTO posts(post_details, sentiment_value) VALUES (%s,%s)"
    """, (data[0], data[1]))

    DB.execute_query(pop_posts)

  def read_all_data():
    connection = DB.create_server_connection()
    try:
      with connection.cursor() as cursor:
        # Read all records
        sql = "SELECT * FROM `posts`"
        cursor.execute(sql)
        return cursor.fetchall()

      connection.commit()
    finally:
      connection.close()

  def seed_all_data():
    connection = DB.create_server_connection()
    try:
      cursor = connection.cursor()
      with open('data/stock_data.csv', encoding="utf8") as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        skipFirtsLine = True
        for row in readCSV:
            data = []
            if row and not skipFirtsLine:
              
                data.append(row[0])
                data.append(int(row[1]))
             
              

                cursor.execute("INSERT INTO posts(post_details, sentiment_value) VALUES (%s,%s)", (data[0], data[1]))


            else:
                skipFirtsLine = False
      connection.commit()
    finally:
      connection.close()

        