"""
Self-Thinking Market Scanner
Automatically discovers and analyzes trading opportunities
"""
import asyncio
import aiohttp
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set
from collections import deque
from aiohttp import ClientTimeout, TCPConnector

from config import config
from models import (
    DiscoveredToken, MarketSnapshot, SignalType, 
    Conviction, InsufficientLiquidityError
)
from logging_manager import get_logger

logger = get_logger("MarketScanner")


class MarketScanner:
    """
    Autonomous market scanner that discovers tokens and opportunities
    WITHOUT manual token selection
    """
    
    BASE_URL = "https://api.dexscreener.com/latest/dex"
    
    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        self._cache: Dict[str, any] = {}
        self._discovered_tokens: Dict[str, DiscoveredToken] = {}
        self._volume_history: Dict[str, deque] = {}
        self._blacklist: Set[str] = set()  # Tokens to avoid
        
        # Rate limiting
        self._semaphore = asyncio.Semaphore(10)
        self._last_request_time = datetime.now()
        self._request_count = 0
        
        logger.info("🔍 Self-Thinking Market Scanner initialized")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session"""
        if self._session is None or self._session.closed:
            connector = TCPConnector(
                limit=20,
                limit_per_host=10,
                ttl_dns_cache=300
            )
            self._session = aiohttp.ClientSession(
                timeout=ClientTimeout(total=20),
                connector=connector
            )
        return self._session
    
    async def close(self):
        """Close HTTP session"""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def scan_market(
        self, 
        chains: Optional[List[str]] = None,
        min_liquidity: Optional[float] = None,
        min_volume: Optional[float] = None
    ) -> List[DiscoveredToken]:
        """
        Scan market for trading opportunities
        
        Args:
            chains: List of chains to scan (default: all major chains)
            min_liquidity: Minimum liquidity in USD
            min_volume: Minimum 24h volume in USD
        
        Returns:
            List of discovered tokens meeting criteria
        """
        if chains is None:
            chains = ["ethereum", "polygon", "arbitrum", "base", "optimism"]
        
        min_liquidity = min_liquidity or config.MIN_LIQUIDITY_USD
        min_volume = min_volume or config.MIN_VOLUME_24H_USD
        
        logger.info(f"🔍 Scanning {len(chains)} chains for opportunities...")
        logger.info(f"   Min Liquidity: ${min_liquidity:,.0f}")
        logger.info(f"   Min Volume: ${min_volume:,.0f}")
        
        all_tokens = []
        
        for chain in chains:
            try:
                tokens = await self._scan_chain(chain, min_liquidity, min_volume)
                all_tokens.extend(tokens)
                logger.info(f"   ✅ {chain}: Found {len(tokens)} tokens")
            except Exception as e:
                logger.warning(f"   ⚠️ {chain}: Failed to scan - {e}")
        
        # Filter and score
        filtered_tokens = self._filter_tokens(all_tokens)
        scored_tokens = self._score_opportunities(filtered_tokens)
        
        # Update cache
        for token in scored_tokens:
            self._discovered_tokens[f"{token.symbol}_{token.chain}"] = token
        
        logger.info(f"✅ Scan complete: {len(scored_tokens)} opportunities found")
        
        return scored_tokens
    
    async def _scan_chain(
        self, 
        chain: str, 
        min_liquidity: float, 
        min_volume: float
    ) -> List[DiscoveredToken]:
        """Scan a specific chain for tokens"""
        async with self._semaphore:
            session = await self._get_session()
            
            # DexScreener trending endpoint
            url = f"{self.BASE_URL}/search"
            
            # Search for high-volume pairs on this chain
            try:
                await asyncio.sleep(0.1)  # Rate limiting
                
                async with session.get(
                    url, 
                    params={"q": chain},
                    timeout=15
                ) as resp:
                    if resp.status != 200:
                        return []
                    
                    data = await resp.json()
                    pairs = data.get("pairs", [])
                    
                    tokens = []
                    seen_addresses = set()
                    
                    for pair in pairs[:50]:  # Limit to top 50 pairs
                        try:
                            # Extract token info
                            chain_id = pair.get("chainId", "").lower()
                            if chain_id != chain.lower():
                                continue
                            
                            base_token = pair.get("baseToken", {})
                            quote_token = pair.get("quoteToken", {})
                            
                            # Skip if quote isn't a stablecoin
                            if quote_token.get("symbol", "").upper() not in ["USDC", "USDT", "DAI", "WETH"]:
                                continue
                            
                            address = base_token.get("address", "").lower()
                            symbol = base_token.get("symbol", "UNKNOWN")
                            
                            # Skip duplicates
                            if address in seen_addresses:
                                continue
                            seen_addresses.add(address)
                            
                            # Extract metrics
                            price = float(pair.get("priceUsd", 0))
                            liquidity = float(pair.get("liquidity", {}).get("usd", 0))
                            volume = float(pair.get("volume", {}).get("h24", 0))
                            change_24h = float(pair.get("priceChange", {}).get("h24", 0))
                            market_cap = pair.get("fdv")  # Fully diluted valuation
                            
                            # Filter by liquidity and volume
                            if liquidity < min_liquidity or volume < min_volume:
                                continue
                            
                            # Create discovered token
                            token = DiscoveredToken(
                                symbol=symbol,
                                address=address,
                                chain=chain,
                                price=price,
                                liquidity_usd=liquidity,
                                volume_24h=volume,
                                change_24h_pct=change_24h,
                                market_cap=float(market_cap) if market_cap else None
                            )
                            
                            tokens.append(token)
                            
                        except Exception as e:
                            logger.debug(f"Failed to parse pair: {e}")
                            continue
                    
                    return tokens
                    
            except Exception as e:
                logger.warning(f"Failed to scan {chain}: {e}")
                return []
    
    def _filter_tokens(self, tokens: List[DiscoveredToken]) -> List[DiscoveredToken]:
        """Apply strict filters to discovered tokens - PRODUCTION SAFETY"""
        filtered = []
        
        for token in tokens:
            # Skip blacklisted
            if token.address in self._blacklist:
                continue
            
            # ENHANCED: More comprehensive scam detection
            suspicious_patterns = [
                "test", "xxx", "scam", "rug", "fake", "moon", "safe",
                "elon", "doge", "shib", "floki", "inu", "pepe",  # High-risk meme patterns
                "token", "coin", "swap", "finance",  # Generic names
                "v2", "2.0", "new"  # Fork indicators
            ]
            symbol_lower = token.symbol.lower()
            if any(pattern in symbol_lower for pattern in suspicious_patterns):
                logger.debug(f"⏭️ Skipping suspicious token: {token.symbol}")
                continue
            
            # Check liquidity/volume ratio (prevents wash trading)
            if token.volume_24h > 0:
                lv_ratio = token.liquidity_usd / token.volume_24h
                
                # TIGHTENED: More conservative ratio
                if lv_ratio < 0.1:  # Volume is 10x liquidity (suspicious)
                    logger.debug(f"⏭️ {token.symbol}: Suspicious liquidity/volume ratio ({lv_ratio:.3f})")
                    continue
                
                # Also check if liquidity is TOO high vs volume (stale pool)
                if lv_ratio > 20:  # Liquidity 20x volume (no trading activity)
                    logger.debug(f"⏭️ {token.symbol}: Stale pool (high liquidity, low volume)")
                    continue
            
            # NEW: Minimum market cap check (avoid micro-caps)
            if token.market_cap and token.market_cap < 1_000_000:  # $1M minimum
                logger.debug(f"⏭️ {token.symbol}: Market cap too low (${token.market_cap:,.0f})")
                continue
            
            # NEW: Symbol length check (scam tokens often have long symbols)
            if len(token.symbol) > 10:
                logger.debug(f"⏭️ {token.symbol}: Symbol too long")
                continue
            
            # NEW: Price sanity check
            if token.price <= 0 or token.price > 1_000_000:
                logger.debug(f"⏭️ {token.symbol}: Invalid price (${token.price})")
                continue
            
            filtered.append(token)
        
        return filtered
    
    def _score_opportunities(self, tokens: List[DiscoveredToken]) -> List[DiscoveredToken]:
        """
        Score tokens based on opportunity potential
        PRODUCTION: More conservative scoring
        """
        for token in tokens:
            score = 0.0
            
            # Volume surge component (REDUCED weight for safety)
            volume_surge = self._calc_volume_surge(
                f"{token.symbol}_{token.chain}", 
                token.volume_24h
            )
            score += volume_surge * 2.0  # Reduced from 3.0
            
            # Price momentum component (MORE SELECTIVE)
            abs_change = abs(token.change_24h_pct)
            if abs_change > 5:  # Only significant moves
                score += abs_change * 0.3  # Reduced from 0.5
            
            # Liquidity safety component (HIGHER requirements)
            if token.liquidity_usd > 1_000_000:  # $1M+
                score += 3.0
            elif token.liquidity_usd > 500_000:  # $500k+
                score += 2.0
            elif token.liquidity_usd > 200_000:  # $200k+
                score += 1.0
            else:
                score -= 2.0  # PENALTY for low liquidity
            
            # Volume component (HIGHER requirements)
            if token.volume_24h > 5_000_000:  # $5M+
                score += 3.0
            elif token.volume_24h > 2_000_000:  # $2M+
                score += 2.0
            elif token.volume_24h > 1_000_000:  # $1M+
                score += 1.0
            else:
                score -= 1.0  # PENALTY for low volume
            
            # Market cap component (STRONG preference for mid-caps)
            if token.market_cap:
                if 50_000_000 < token.market_cap < 500_000_000:  # $50M-$500M sweet spot
                    score += 3.0
                elif 10_000_000 < token.market_cap < 50_000_000:  # $10M-$50M
                    score += 1.5
                elif token.market_cap < 5_000_000:  # <$5M (risky micro-cap)
                    score -= 3.0  # HEAVY PENALTY
            
            # NEW: Stability bonus (prefer established tokens)
            # Lower volatility = more stable = bonus
            if abs_change < 3:  # Less than 3% move
                score += 1.0
            
            # NEW: Volume consistency check
            if volume_surge < 0.5:  # Volume is stable (not just a spike)
                score += 0.5
            
            token.opportunity_score = max(0.0, score)  # Can't be negative
        
        # Sort by score descending
        tokens.sort(key=lambda t: t.opportunity_score, reverse=True)
        
        # PRODUCTION: Only return tokens with score > 8.0 (high quality)
        quality_tokens = [t for t in tokens if t.opportunity_score >= 8.0]
        
        return quality_tokens
    
    def _calc_volume_surge(self, identifier: str, volume: float) -> float:
        """Calculate volume surge score"""
        if volume <= 0:
            return 0.0
        
        if identifier not in self._volume_history:
            self._volume_history[identifier] = deque(maxlen=20)
        
        hist = self._volume_history[identifier]
        hist.append(volume)
        
        if len(hist) < 3:
            return 0.1
        
        if len(hist) < 7:
            avg = sum(hist) / len(hist)
            if avg == 0:
                return 0.3
            ratio = volume / avg
            return max(0.0, (ratio - 0.5) * 0.5)
        
        # EMA calculation
        ema_alpha = 0.3
        ema = hist[0]
        for v in list(hist)[1:]:
            ema = ema_alpha * v + (1 - ema_alpha) * ema
        
        if ema == 0:
            return 0.5
        
        ratio = volume / ema
        surge_score = max(0.0, ratio - 0.7)
        
        return surge_score
    
    def blacklist_token(self, address: str):
        """Add token to blacklist"""
        self._blacklist.add(address.lower())
        logger.info(f"🚫 Blacklisted token: {address}")
    
    def get_discovered_tokens(self) -> Dict[str, DiscoveredToken]:
        """Get all currently discovered tokens"""
        return self._discovered_tokens
    
    def get_top_opportunities(self, n: int = 10) -> List[DiscoveredToken]:
        """Get top N opportunities by score"""
        tokens = sorted(
            self._discovered_tokens.values(),
            key=lambda t: t.opportunity_score,
            reverse=True
        )
        return tokens[:n]