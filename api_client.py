"""
Recall API Client with resilience features
"""
import asyncio
import json
from datetime import datetime, timezone
from typing import Optional, Dict

import aiohttp
from aiohttp import ClientTimeout, TCPConnector

from config import config
from models import CircuitBreakerOpenError, CircuitState
from logging_manager import get_logger

logger = get_logger("RecallAPI")


class CircuitBreaker:
    """Circuit breaker pattern for API resilience"""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        name: str = "default"
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.name = name
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.success_count = 0
        self._lock = asyncio.Lock()
    
    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        async with self._lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                    logger.info(f"🔄 Circuit breaker '{self.name}' entering HALF_OPEN")
                else:
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker '{self.name}' is OPEN. "
                        f"Retry after {self._time_until_reset():.0f}s"
                    )
        
        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except Exception as e:
            await self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        if self.last_failure_time is None:
            return True
        elapsed = (datetime.now(timezone.utc) - self.last_failure_time).total_seconds()
        return elapsed >= self.recovery_timeout
    
    def _time_until_reset(self) -> float:
        if self.last_failure_time is None:
            return 0
        elapsed = (datetime.now(timezone.utc) - self.last_failure_time).total_seconds()
        return max(0, self.recovery_timeout - elapsed)
    
    async def _on_success(self):
        async with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= 2:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    self.success_count = 0
                    logger.info(f"✅ Circuit breaker '{self.name}' CLOSED (recovered)")
            else:
                self.failure_count = 0
    
    async def _on_failure(self):
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now(timezone.utc)
            self.success_count = 0
            
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
                logger.warning(
                    f"⚠️ Circuit breaker '{self.name}' OPEN after {self.failure_count} failures"
                )


class ResilientHTTPClient:
    """HTTP client with retry logic and circuit breaker"""
    
    def __init__(
        self,
        base_url: str,
        headers: Dict[str, str],
        retry_count: int = 5,
        timeout: int = 30,
        circuit_breaker: Optional[CircuitBreaker] = None
    ):
        self.base_url = base_url.rstrip('/')
        self.headers = headers
        self.retry_count = retry_count
        self.timeout = ClientTimeout(total=timeout)
        self.circuit_breaker = circuit_breaker or CircuitBreaker(name="http_client")
        self._session: Optional[aiohttp.ClientSession] = None
        self._connector: Optional[TCPConnector] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session"""
        if self._session is None or self._session.closed:
            self._connector = TCPConnector(
                limit=10,
                limit_per_host=5,
                ttl_dns_cache=300
            )
            self._session = aiohttp.ClientSession(
                connector=self._connector,
                timeout=self.timeout,
                headers=self.headers
            )
        return self._session
    
    async def close(self):
        """Close HTTP session"""
        if self._session and not self._session.closed:
            await self._session.close()
        if self._connector:
            await self._connector.close()
    
    async def request(self, method: str, endpoint: str, **kwargs) -> Dict:
        """Make HTTP request with retry and circuit breaker"""
        return await self.circuit_breaker.call(
            self._request_with_retry, method, endpoint, **kwargs
        )
    
    async def _request_with_retry(self, method: str, endpoint: str, **kwargs) -> Dict:
        """Internal request with exponential backoff retry"""
        url = f"{self.base_url}{endpoint}"
        last_error = None
        
        for attempt in range(self.retry_count):
            try:
                session = await self._get_session()
                
                async with session.request(method, url, **kwargs) as response:
                    text = await response.text()
                    
                    if response.status == 429:
                        retry_after = int(response.headers.get('Retry-After', 60))
                        logger.warning(f"⏳ Rate limited. Waiting {retry_after}s")
                        await asyncio.sleep(retry_after)
                        continue
                    
                    if response.status >= 500:
                        raise aiohttp.ClientResponseError(
                            response.request_info,
                            response.history,
                            status=response.status,
                            message=text
                        )
                    
                    if response.status >= 400:
                        logger.error(f"API Error ({response.status}): {text[:500]}")
                        response.raise_for_status()
                    
                    return json.loads(text) if text else {}
                    
            except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError) as e:
                last_error = e
                if attempt < self.retry_count - 1:
                    delay = min(
                        config.API_BACKOFF_BASE ** attempt,
                        config.API_BACKOFF_MAX
                    )
                    logger.warning(
                        f"⚠️ Request failed (attempt {attempt + 1}/{self.retry_count}): {e}. "
                        f"Retrying in {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)
        
        raise last_error or Exception("Request failed after all retries")


class RecallAPIClient:
    """Recall Network API client"""
    
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "TradingAgent/4.0"
        }
        
        self.http = ResilientHTTPClient(
            base_url=base_url,
            headers=headers,
            retry_count=config.API_RETRY_COUNT,
            timeout=config.API_TIMEOUT_SECONDS,
            circuit_breaker=CircuitBreaker(
                failure_threshold=config.CIRCUIT_FAILURE_THRESHOLD,
                recovery_timeout=config.CIRCUIT_RECOVERY_TIMEOUT,
                name="recall_api"
            )
        )
        
        env = "SANDBOX" if "sandbox" in base_url else "PRODUCTION"
        logger.info(f"✅ Recall API Client initialized ({env})")
    
    async def close(self):
        """Close client connections"""
        await self.http.close()
    
    async def get_portfolio(self, competition_id: str) -> Dict:
        """Get agent balances"""
        return await self.http.request(
            "GET",
            f"/api/agent/balances?competitionId={competition_id}"
        )
    
    async def get_token_price(self, token_address: str, chain: str = "eth") -> float:
        """Get token price"""
        result = await self.http.request(
            "GET",
            f"/api/price?token={token_address}&chain={chain}"
        )
        return float(result.get("price", 0.0))
    
    async def execute_trade(
        self,
        competition_id: str,
        from_token: str,
        to_token: str,
        amount: str,
        reason: str = "AI trading decision",
        from_chain: Optional[str] = None,
        to_chain: Optional[str] = None
    ) -> Dict:
        """Execute a trade"""
        payload = {
            "competitionId": competition_id,
            "fromToken": from_token,
            "toToken": to_token,
            "amount": amount,
            "reason": reason[:500]
        }
        
        if from_chain:
            payload["fromChain"] = from_chain
        if to_chain:
            payload["toChain"] = to_chain
        
        return await self.http.request("POST", "/api/trade/execute", json=payload)
    
    async def get_trade_history(self, competition_id: str) -> Dict:
        """Get trade history"""
        return await self.http.request(
            "GET",
            f"/api/agent/trades?competitionId={competition_id}"
        )
    
    async def get_leaderboard(self, competition_id: Optional[str] = None) -> Dict:
        """Get leaderboard"""
        endpoint = "/api/leaderboard"
        if competition_id:
            endpoint += f"?competitionId={competition_id}"
        return await self.http.request("GET", endpoint)
    
    async def get_competitions(self) -> Dict:
        """Get all competitions"""
        return await self.http.request("GET", "/api/competitions")
    
    async def get_user_competitions(self) -> Dict:
        """Get user's competitions"""
        return await self.http.request("GET", "/api/user/competitions")