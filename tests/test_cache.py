import redis

from src.app.cache import redis_cache


class FakeRedis:
    def __init__(self):
        self.store = {}

    def get(self, key):
        return self.store.get(key)

    def set(self, key, value):
        self.store[key] = value


class BrokenRedis:
    def get(self, key):
        raise redis.ConnectionError

    def set(self, key, value):
        raise redis.ConnectionError


def test_cache_returns_what_was_stored(monkeypatch):
    monkeypatch.setattr(redis_cache, "redis_client", FakeRedis())
    redis_cache.set_cached_prediction("hello", "English")
    assert redis_cache.get_cached_prediction("hello") == "English"


def test_cache_miss_returns_none(monkeypatch):
    monkeypatch.setattr(redis_cache, "redis_client", FakeRedis())
    assert redis_cache.get_cached_prediction("unknown") is None


def test_redis_being_down_is_not_an_error(monkeypatch):
    monkeypatch.setattr(redis_cache, "redis_client", BrokenRedis())
    redis_cache.set_cached_prediction("hello", "English")
    assert redis_cache.get_cached_prediction("hello") is None
