# Used in GCC AWS
class Config():
    FLASK_APP = "main"
    FLASK_ENV = "development"
    DEBUG = True
    APP_PORT = 8080
    SESSION_LIFETIME = 60

    DB_conn_string = ""
    DB_POOL = "db_pool"
    DB_HOST = "localhost"
    DB_USER = "root"

    DB_database_name = "stocks"
    DB_collection = "Data"