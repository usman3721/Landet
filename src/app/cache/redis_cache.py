import redis

from src.app.core.config import settings

redis_client = redis.Redis.from_url(
    settings.REDIS_URL, decode_responses=True, socket_connect_timeout=1
)


# The cache only makes things faster, so if Redis is down we skip it instead of failing.
def get_cached_prediction(key: str):
    try:
        return redis_client.get(key)
    except redis.RedisError:
        return None


def set_cached_prediction(key: str, value: str):
    try:
        redis_client.set(key, value)
    except redis.RedisError:
        pass
