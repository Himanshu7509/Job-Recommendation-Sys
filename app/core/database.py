import os
from pymongo import MongoClient
import dotenv 

dotenv.load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")

if not MONGO_URI:
    raise ValueError("MONGO_URI environment variable not set")

DB_NAME = "jobsDB"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]

jobs_collection = db["jobs"]
