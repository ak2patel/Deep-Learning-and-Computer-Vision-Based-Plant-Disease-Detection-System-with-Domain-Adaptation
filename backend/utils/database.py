"""
Database client management for Prisma ORM.

Provides a singleton async Prisma client that connects on app startup
and disconnects on shutdown via FastAPI lifespan events.
"""
from prisma import Prisma

# Singleton Prisma client instance
db = Prisma()


async def connect_db():
    """Connect to the database. Call this on app startup."""
    await db.connect()
    print("✅ Connected to PostgreSQL via Prisma")


async def disconnect_db():
    """Disconnect from the database. Call this on app shutdown."""
    await db.disconnect()
    print("🔌 Disconnected from PostgreSQL")
