from motor.motor_asyncio import AsyncIOMotorClient
from fastapi import Depends
import os
from dotenv import load_dotenv

load_dotenv()

MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
client = AsyncIOMotorClient(MONGODB_URL)
db = client.seniorjobgo


async def get_database():
    try:
        await client.admin.command("ping")
        print("MongoDB Connection Success")
        return db
    except Exception as e:
        print(f"MongoDB Connection Error: {e}")
        raise e


__all__ = ["get_database"]
