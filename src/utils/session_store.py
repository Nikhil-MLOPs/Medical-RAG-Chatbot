import json
import os
import redis
from src.utils.logger import logger


REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
REDIS_DB = int(os.getenv("REDIS_DB", 0))
SESSION_TTL = int(os.getenv("SESSION_TTL_SECONDS", 86400))  # 24 hours


class SessionStore:
    def __init__(self):
        try:
            self.client = redis.Redis(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", 6379)),
                password=os.getenv("REDIS_PASSWORD", None),

                # CRITICAL PART
                socket_connect_timeout=3,   # if Redis not reachable → fail fast
                socket_timeout=3,
                retry_on_timeout=False,
                ssl=os.getenv("REDIS_SSL", "false").lower() == "true"
            )

            self.client.ping()
            logger.info("Connected to Redis successfully")

        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            raise

    def _key(self, session_id):
        return f"medical_chat:{session_id}"

    def get_history(self, session_id):
        key = self._key(session_id)

        data = self.client.get(key)
        if not data:
            return []

        return json.loads(data)

    def save_turn(self, session_id, user_message, assistant_message):
        key = self._key(session_id)

        history = self.get_history(session_id)
        history.append({"role": "user", "message": user_message})
        history.append({"role": "assistant", "message": assistant_message})

        self.client.set(key, json.dumps(history))
        self.client.expire(key, SESSION_TTL)
        return history