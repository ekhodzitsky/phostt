//! Redis-backed rate limiter for multi-instance deployments.
//!
//! Enabled via the `redis-rate-limit` feature. Falls back to the in-memory
//! [`RateLimiter`](super::RateLimiter) when `REDIS_URL` is not configured.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};



/// Redis-backed token-bucket rate limiter.
///
/// Uses a Lua script for atomic check-and-deduct so concurrent requests from
/// the same IP never over-count.
pub struct RedisRateLimiter {
    conn: redis::aio::MultiplexedConnection,
    refill_per_ms: f64,
    burst: u64,
    script: redis::Script,
}

impl RedisRateLimiter {
    /// Connect to Redis and create a new limiter.
    pub async fn new(redis_url: &str, rpm: u64, burst: u64) -> anyhow::Result<Arc<Self>> {
        let client = redis::Client::open(redis_url)?;
        let conn = client.get_multiplexed_async_connection().await?;

        let script = redis::Script::new(
            r#"
            local key = KEYS[1]
            local refill = tonumber(ARGV[1])
            local burst = tonumber(ARGV[2])
            local now = tonumber(ARGV[3])
            local cost = tonumber(ARGV[4])

            local data = redis.call('HMGET', key, 'tokens', 'last_refill')
            local tokens = tonumber(data[1]) or burst
            local last = tonumber(data[2]) or now

            tokens = math.min(burst, tokens + (now - last) * refill)

            if tokens >= cost then
                tokens = tokens - cost
                redis.call('HSET', key, 'tokens', tokens, 'last_refill', now)
                redis.call('PEXPIRE', key, 60000)
                return 1
            else
                redis.call('HSET', key, 'tokens', tokens, 'last_refill', now)
                redis.call('PEXPIRE', key, 60000)
                return 0
            end
            "#,
        );

        Ok(Arc::new(Self {
            conn,
            refill_per_ms: rpm as f64 / 60_000.0,
            burst,
            script,
        }))
    }

    /// Check whether `ip` has enough tokens for one request.
    /// Returns `true` if allowed, `false` if throttled.
    pub async fn check(&self, ip: &str) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let key = format!("rate_limit:{ip}");
        self
            .script
            .key(&key)
            .arg(self.refill_per_ms)
            .arg(self.burst)
            .arg(now)
            .arg(1)
            .invoke_async(&mut self.conn.clone())
            .await
            .unwrap_or(true)
    }

    /// Milliseconds to wait before the next token is available.
    pub fn retry_after_ms(&self) -> u64 {
        ((1.0 / self.refill_per_ms) as u64).max(1000)
    }
}
