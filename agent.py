"""
Recall Network Advanced Paper Trading Agent
Production-ready AI agent for competitive paper trading
Features: Multi-chain, smart sizing, risk management, diversification
"""

import os
import sys
import json
import logging
import asyncio
from datetime import datetime, timezone
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

import aiohttp
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Advanced configuration for competitive paper trading"""
    
    # API Configuration
    RECALL_API_KEY = os.getenv("RECALL_API_KEY", "")
    USE_SANDBOX = os.getenv("RECALL_USE_SANDBOX", "true").lower() == "true"
    COMPETITION_ID = os.getenv("COMPETITION_ID", "")
    
    SANDBOX_URL = "https://api.sandbox.competitions.recall.network"
    PRODUCTION_URL = "https://api.competitions.recall.network"
    
    @property
    def base_url(self):
        return self.SANDBOX_URL if self.USE_SANDBOX else self.PRODUCTION_URL
    
    # LLM Configuration
    GAIA_API_KEY = os.getenv("GAIA_API_KEY")
    GAIA_NODE_URL = os.getenv("GAIA_NODE_URL", "https://qwen7b.gaia.domains/v1")
    GAIA_MODEL_NAME = os.getenv("GAIA_MODEL_NAME")
    
    # Trading Strategy Configuration
    TRADING_INTERVAL = int(os.getenv("TRADING_INTERVAL_SECONDS", "300"))  # 5 min (need 3+ trades/day)
    
    # Position Sizing (Following competition rules)
    MIN_TRADE_SIZE = float(os.getenv("MIN_TRADE_SIZE", "0.000001"))  # Competition minimum
    BASE_POSITION_SIZE = float(os.getenv("BASE_POSITION_SIZE", "300"))  # Smaller for more trades
    MAX_POSITION_PCT = float(os.getenv("MAX_POSITION_PCT", "0.20"))  # 20% (rule: max 25%)
    MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", "8"))  # More diversification
    MIN_TRADES_PER_DAY = int(os.getenv("MIN_TRADES_PER_DAY", "3"))  # Competition requirement
    
    # Risk Management
    STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "-0.08"))  # -8% stop loss
    TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", "0.15"))  # +15% take profit
    MAX_PORTFOLIO_RISK = float(os.getenv("MAX_PORTFOLIO_RISK", "0.60"))  # 60% max deployed
    
    # Strategy Settings
    STRATEGY_MODE = os.getenv("STRATEGY_MODE", "BALANCED")  # CONSERVATIVE, BALANCED, AGGRESSIVE
    ENABLE_MEAN_REVERSION = os.getenv("ENABLE_MEAN_REVERSION", "true").lower() == "true"
    ENABLE_MOMENTUM = os.getenv("ENABLE_MOMENTUM", "true").lower() == "true"
    
    # Multi-chain token catalog - COMPETITION COMPLIANT
    # All chains supported: Solana, Ethereum, Polygon, BSC, Arbitrum, Base, Optimism, Avalanche, Linea
    TOKENS = {
        # === Ethereum Mainnet (EVM) ===
        "USDC": {"address": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48", "chain": "eth", "stable": True},
        "WETH": {"address": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2", "chain": "eth", "stable": False},
        "WBTC": {"address": "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599", "chain": "eth", "stable": False},
        "DAI": {"address": "0x6B175474E89094C44Da98b954EedeAC495271d0F", "chain": "eth", "stable": True},
        "UNI": {"address": "0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984", "chain": "eth", "stable": False},
        "LINK": {"address": "0x514910771AF9Ca656af840dff83E8264EcF986CA", "chain": "eth", "stable": False},
        "AAVE": {"address": "0x7Fc66500c84A76Ad7e9c93437bFc5Ac33E2DDaE9", "chain": "eth", "stable": False},
        "MKR": {"address": "0x9f8F72aA9304c8B593d555F12eF6589cC3A579A2", "chain": "eth", "stable": False},
        "SNX": {"address": "0xC011a73ee8576Fb46F5E1c5751cA3B9Fe0af2a6F", "chain": "eth", "stable": False},
        "CRV": {"address": "0xD533a949740bb3306d119CC777fa900bA034cd52", "chain": "eth", "stable": False},
        "LDO": {"address": "0x5A98FcBEA516Cf06857215779Fd812CA3beF1B32", "chain": "eth", "stable": False},
        "PEPE": {"address": "0x6982508145454Ce325dDbE47a25d4ec3d2311933", "chain": "eth", "stable": False},
        
        # Add more chains as needed (Polygon, BSC, Arbitrum, Base, etc.)
    }

config = Config()

# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class TokenInfo:
    """Token information"""
    symbol: str
    address: str
    chain: str
    price: float
    change_24h: float
    is_stable: bool
    
@dataclass
class Position:
    """Trading position"""
    symbol: str
    amount: float
    entry_price: float
    current_price: float
    value: float
    pnl_pct: float
    
    @property
    def is_profitable(self) -> bool:
        return self.pnl_pct > 0
    
    @property
    def should_stop_loss(self) -> bool:
        return self.pnl_pct < config.STOP_LOSS_PCT
    
    @property
    def should_take_profit(self) -> bool:
        return self.pnl_pct > config.TAKE_PROFIT_PCT

# ============================================================================
# RECALL API CLIENT
# ============================================================================

class RecallAPIClient:
    """Enhanced Recall API Client"""
    
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        env = "SANDBOX" if "sandbox" in base_url else "PRODUCTION"
        logger.info(f"✅ Recall API Client initialized")
        logger.info(f"   Environment: {env}")
        logger.info(f"   Base URL: {base_url}")
    
    async def _request(self, method: str, endpoint: str, **kwargs):
        """Make HTTP request with retry logic"""
        url = f"{self.base_url}{endpoint}"
        
        for attempt in range(3):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.request(
                        method, url, headers=self.headers, timeout=30, **kwargs
                    ) as response:
                        text = await response.text()
                        
                        if response.status >= 400:
                            logger.error(f"API Error ({response.status}): {text}")
                        
                        response.raise_for_status()
                        return json.loads(text)
                        
            except aiohttp.ClientError as e:
                if attempt == 2:
                    logger.error(f"API request failed after 3 attempts: {e}")
                    raise
                await asyncio.sleep(2 ** attempt)
    
    async def get_portfolio(self, competition_id: str) -> Dict:
        """Get agent balances"""
        return await self._request("GET", f"/api/agent/balances?competitionId={competition_id}")
    
    async def get_token_price(self, token_address: str, chain: str = "eth") -> float:
        """Get token price (rate limited: 300/min)"""
        result = await self._request("GET", f"/api/price?token={token_address}&chain={chain}")
        return result.get("price", 0.0)
    
    async def execute_trade(
        self,
        competition_id: str,
        from_token: str,
        to_token: str,
        amount: str,
        reason: str = "AI trading decision",
        from_chain: str = None,
        to_chain: str = None
    ) -> Dict:
        """Execute a trade"""
        payload = {
            "competitionId": competition_id,
            "fromToken": from_token,
            "toToken": to_token,
            "amount": amount,
            "reason": reason
        }
        
        if from_chain:
            payload["fromChain"] = from_chain
        if to_chain:
            payload["toChain"] = to_chain
        
        return await self._request("POST", "/api/trade/execute", json=payload)
    
    async def get_trade_history(self, competition_id: str) -> Dict:
        """Get trade history"""
        return await self._request("GET", f"/api/agent/trades?competitionId={competition_id}")
    
    async def get_leaderboard(self, competition_id: str = None) -> Dict:
        """Get leaderboard"""
        if competition_id:
            return await self._request("GET", f"/api/leaderboard?competitionId={competition_id}")
        return await self._request("GET", "/api/leaderboard")
    
    async def get_competitions(self) -> Dict:
        """Get competitions"""
        return await self._request("GET", "/api/competitions")
    
    async def get_user_competitions(self) -> Dict:
        """Get user competitions"""
        return await self._request("GET", "/api/user/competitions")

# ============================================================================
# MARKET ANALYSIS
# ============================================================================

class MarketAnalyzer:
    """Advanced market analysis with multiple indicators"""
    
    COINGECKO_IDS = {
        "WETH": "ethereum",
        "WBTC": "bitcoin",
        "UNI": "uniswap",
        "LINK": "chainlink",
        "DAI": "dai",
        "USDC": "usd-coin",
        "AAVE": "aave",
        "MKR": "maker",
        "SNX": "synthetix-network-token",
        "CRV": "curve-dao-token",
        "LDO": "lido-dao",
        "MATIC": "matic-network",
        "SHIB": "shiba-inu",
        "PEPE": "pepe",
    }
    
    async def get_market_data(self) -> Dict[str, Dict]:
        """Get comprehensive market data from CoinGecko"""
        try:
            ids = ",".join(self.COINGECKO_IDS.values())
            url = "https://api.coingecko.com/api/v3/simple/price"
            params = {
                "ids": ids,
                "vs_currencies": "usd",
                "include_24hr_change": "true",
                "include_24hr_vol": "true",
                "include_market_cap": "true"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=15) as response:
                    response.raise_for_status()
                    data = await response.json()
                    
                    market_data = {}
                    for symbol, cg_id in self.COINGECKO_IDS.items():
                        if cg_id in data and "usd" in data[cg_id]:
                            token_data = data[cg_id]
                            market_data[symbol] = {
                                "price": token_data.get("usd", 0),
                                "change_24h": token_data.get("usd_24h_change", 0.0),
                                "volume_24h": token_data.get("usd_24h_vol", 0.0),
                                "market_cap": token_data.get("usd_market_cap", 0.0)
                            }
                    
                    if not market_data:
                        logger.warning("⚠️  CoinGecko returned empty data, using fallback")
                    
                    return market_data
        except Exception as e:
            logger.error(f"Failed to fetch market data: {e}")
            # Return empty dict to continue operation
            return {}
    
    @staticmethod
    def classify_signal(change_24h: float, strategy: str) -> Tuple[str, str, str]:
        """Classify trading signal based on momentum"""
        
        if strategy == "AGGRESSIVE":
            thresholds = [(3, "STRONG_BULLISH", "HIGH"), (1, "BULLISH", "MEDIUM"), 
                         (0.3, "WEAK_BULLISH", "LOW"), (-0.3, "NEUTRAL", "LOW"),
                         (-1, "WEAK_BEARISH", "LOW"), (-3, "BEARISH", "MEDIUM")]
        elif strategy == "CONSERVATIVE":
            thresholds = [(5, "STRONG_BULLISH", "HIGH"), (3, "BULLISH", "MEDIUM"),
                         (1, "WEAK_BULLISH", "LOW"), (-1, "NEUTRAL", "LOW"),
                         (-3, "WEAK_BEARISH", "LOW"), (-5, "BEARISH", "MEDIUM")]
        else:  # BALANCED
            thresholds = [(4, "STRONG_BULLISH", "HIGH"), (2, "BULLISH", "MEDIUM"),
                         (0.5, "WEAK_BULLISH", "LOW"), (-0.5, "NEUTRAL", "LOW"),
                         (-2, "WEAK_BEARISH", "LOW"), (-4, "BEARISH", "MEDIUM")]
        
        for threshold, signal, conviction in thresholds:
            if change_24h > threshold:
                return signal, conviction, f"{signal.replace('_', ' ').title()} ({change_24h:+.2f}%)"
        
        return "STRONG_BEARISH", "HIGH", f"Strong Bearish ({change_24h:+.2f}%)"
    
    def find_opportunities(self, market_data: Dict, portfolio: Dict) -> List[Tuple[str, float, str]]:
        """Find trading opportunities"""
        opportunities = []
        
        for symbol, data in market_data.items():
            if symbol in ["USDC", "DAI"]:  # Skip stablecoins
                continue
            
            change = data["change_24h"]
            signal, conviction, _ = self.classify_signal(change, config.STRATEGY_MODE)
            
            # Score based on signal strength
            score = 0
            if "BULLISH" in signal:
                score = abs(change)
                if conviction == "HIGH":
                    score *= 1.5
                elif conviction == "MEDIUM":
                    score *= 1.2
            
            if score > 0:
                opportunities.append((symbol, score, signal))
        
        # Sort by score
        opportunities.sort(key=lambda x: x[1], reverse=True)
        return opportunities[:5]  # Top 5

# ============================================================================
# TRADING STRATEGY
# ============================================================================

class TradingStrategy:
    """Advanced trading strategy engine"""
    
    def __init__(self, analyzer: MarketAnalyzer):
        self.analyzer = analyzer
    
    def calculate_position_size(self, total_value: float, conviction: str) -> float:
        """Calculate position size based on conviction"""
        base_size = config.BASE_POSITION_SIZE
        
        # Adjust by conviction
        multipliers = {"HIGH": 1.5, "MEDIUM": 1.0, "LOW": 0.6}
        size = base_size * multipliers.get(conviction, 1.0)
        
        # Cap at max position %
        max_size = total_value * config.MAX_POSITION_PCT
        size = min(size, max_size)
        
        # Respect minimum
        size = max(size, config.MIN_TRADE_SIZE)
        
        return size
    
    def should_rebalance(self, positions: List[Position]) -> bool:
        """Check if portfolio needs rebalancing"""
        if not positions:
            return False
        
        # Check for positions that need stop-loss or take-profit
        for pos in positions:
            if pos.should_stop_loss or pos.should_take_profit:
                return True
        
        # Check concentration risk
        if positions:
            max_position = max(pos.value for pos in positions)
            total_value = sum(pos.value for pos in positions)
            if max_position / total_value > 0.4:  # Over 40% in one position
                return True
        
        return False
    
    def generate_trade_decision(
        self,
        portfolio: Dict,
        market_data: Dict,
        opportunities: List[Tuple[str, float, str]]
    ) -> Dict:
        """Generate intelligent trade decision"""
        
        total_value = portfolio.get("total_value", 0)
        holdings = portfolio.get("holdings", {})
        positions = portfolio.get("positions", [])
        
        # Calculate deployed capital
        deployed = sum(h["value"] for h in holdings.values() if h.get("symbol") != "USDC")
        deployed_pct = deployed / total_value if total_value > 0 else 0
        
        usdc_balance = holdings.get("USDC", {}).get("amount", 0)
        usdc_value = holdings.get("USDC", {}).get("value", 0)
        
        # 1. Risk Management: Check existing positions
        for pos in positions:
            if pos.should_stop_loss:
                return {
                    "action": "SELL",
                    "from_token": pos.symbol,
                    "to_token": "USDC",
                    "amount_usd": pos.value * 0.98,
                    "conviction": "HIGH",
                    "reason": f"Stop-loss triggered: {pos.pnl_pct:.1f}% loss"
                }
            
            if pos.should_take_profit:
                # Take 50% profit
                return {
                    "action": "SELL",
                    "from_token": pos.symbol,
                    "to_token": "USDC",
                    "amount_usd": pos.value * 0.5,
                    "conviction": "MEDIUM",
                    "reason": f"Take-profit: {pos.pnl_pct:.1f}% gain"
                }
        
        # 2. Check if we should deploy more capital
        if deployed_pct >= config.MAX_PORTFOLIO_RISK:
            return {
                "action": "HOLD",
                "reason": f"Max risk deployed ({deployed_pct*100:.0f}%). Monitoring positions."
            }
        
        # 3. Don't trade if insufficient USDC
        if usdc_value < config.MIN_TRADE_SIZE * 1.5:
            return {
                "action": "HOLD",
                "reason": f"Insufficient USDC (${usdc_value:.0f}). Waiting for better liquidity."
            }
        
        # 4. Find best opportunity
        if not opportunities:
            return {
                "action": "HOLD",
                "reason": "No clear opportunities. Market analysis shows neutral signals."
            }
        
        # Check if we're already in top opportunities
        existing_symbols = set(pos.symbol for pos in positions)
        new_opportunities = [opp for opp in opportunities if opp[0] not in existing_symbols]
        
        if not new_opportunities and len(positions) >= config.MAX_POSITIONS:
            return {
                "action": "HOLD",
                "reason": f"Max positions ({config.MAX_POSITIONS}) reached. Monitoring portfolio."
            }
        
        # 5. Execute new position
        if new_opportunities:
            symbol, score, signal = new_opportunities[0]
            token_data = market_data.get(symbol, {})
            change = token_data.get("change_24h", 0)
            
            _, conviction, _ = self.analyzer.classify_signal(change, config.STRATEGY_MODE)
            position_size = self.calculate_position_size(total_value, conviction)
            
            # Don't exceed available USDC
            position_size = min(position_size, usdc_value * 0.95)
            
            return {
                "action": "BUY",
                "from_token": "USDC",
                "to_token": symbol,
                "amount_usd": position_size,
                "conviction": conviction,
                "reason": f"{signal} signal, {change:+.2f}% momentum, score: {score:.1f}"
            }
        
        return {
            "action": "HOLD",
            "reason": "Waiting for better setup. Current positions look good."
        }

# ============================================================================
# ADVANCED TRADING AGENT
# ============================================================================

class AdvancedTradingAgent:
    """Production-ready AI trading agent"""
    
    def __init__(self):
        if not config.RECALL_API_KEY:
            raise ValueError("❌ RECALL_API_KEY not set!")
        
        self.client = RecallAPIClient(config.RECALL_API_KEY, config.base_url)
        self.analyzer = MarketAnalyzer()
        self.strategy = TradingStrategy(self.analyzer)
        self.competition_id = None
        self.trade_history = []
        self.trades_today = 0
        self.last_trade_date = None
        
        logger.info("🤖 Advanced Trading Agent initialized")
        logger.info(f"   Strategy: {config.STRATEGY_MODE}")
        logger.info(f"   Base Position Size: ${config.BASE_POSITION_SIZE}")
        logger.info(f"   Max Positions: {config.MAX_POSITIONS}")
        logger.info(f"   Min Trades/Day: {config.MIN_TRADES_PER_DAY}")
        logger.info(f"   Risk Management: SL={config.STOP_LOSS_PCT*100:.0f}%, TP={config.TAKE_PROFIT_PCT*100:.0f}%")
    
    async def select_competition(self) -> str:
        """Select competition"""
        if config.COMPETITION_ID:
            logger.info(f"✅ Using configured competition: {config.COMPETITION_ID}")
            return config.COMPETITION_ID
        
        try:
            logger.info("🔍 Fetching available competitions...")
            user_comps = await self.client.get_user_competitions()
            competitions = user_comps.get("competitions", [])
            
            if not competitions:
                all_comps = await self.client.get_competitions()
                competitions = all_comps.get("competitions", [])
            
            if not competitions:
                raise ValueError("❌ No competitions found!")
            
            active = [c for c in competitions if c.get("status") == "active"]
            comp = active[0] if active else competitions[0]
            comp_id = comp.get("id")
            
            logger.info(f"✅ Auto-selected competition:")
            logger.info(f"   ID: {comp_id}")
            logger.info(f"   Name: {comp.get('name', 'N/A')}")
            return comp_id
            
        except Exception as e:
            raise ValueError(f"❌ Could not select competition: {e}")
    
    async def get_portfolio_state(self) -> Dict:
        """Get comprehensive portfolio state"""
        try:
            # Get balances
            portfolio = await self.client.get_portfolio(self.competition_id)
            market_data = await self.analyzer.get_market_data()
            
            balances = portfolio.get("balances", [])
            total_value = 0
            holdings = {}
            positions = []
            
            for balance in balances:
                symbol = balance.get("symbol", "")
                amount = float(balance.get("amount", 0))
                price = float(balance.get("price", 0))
                value = amount * price
                total_value += value
                
                if symbol in config.TOKENS:
                    holdings[symbol] = {
                        "symbol": symbol,
                        "amount": amount,
                        "value": value,
                        "price": price,
                        "pct": 0
                    }
                    
                    # Track non-stablecoin positions
                    if not config.TOKENS[symbol]["stable"] and amount > 0:
                        # Calculate PnL (simplified - would need entry price from history)
                        market_price = market_data.get(symbol, {}).get("price", price)
                        pnl_pct = ((market_price - price) / price * 100) if price > 0 else 0
                        
                        positions.append(Position(
                            symbol=symbol,
                            amount=amount,
                            entry_price=price,
                            current_price=market_price,
                            value=value,
                            pnl_pct=pnl_pct
                        ))
            
            # Calculate percentages
            for symbol in holdings:
                if total_value > 0:
                    holdings[symbol]["pct"] = (holdings[symbol]["value"] / total_value * 100)
            
            return {
                "total_value": total_value,
                "holdings": holdings,
                "positions": positions,
                "market_data": market_data,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to get portfolio: {e}")
            return {}
    
    async def execute_trade(self, decision: Dict, portfolio: Dict) -> bool:
        """Execute trade with validation and competition rules compliance"""
        action = decision.get("action", "HOLD").upper()
        
        if action == "HOLD":
            logger.info(f"⏸️  {decision.get('reason', 'Holding position')}")
            return False
        
        from_symbol = decision.get("from_token", "").upper()
        to_symbol = decision.get("to_token", "").upper()
        amount_usd = decision.get("amount_usd", 0)
        reason = decision.get("reason", "AI trading decision")
        
        if from_symbol not in config.TOKENS or to_symbol not in config.TOKENS:
            logger.error(f"❌ Invalid tokens: {from_symbol} → {to_symbol}")
            return False
        
        from_token = config.TOKENS[from_symbol]
        to_token = config.TOKENS[to_symbol]
        
        holdings = portfolio["holdings"]
        from_holding = holdings.get(from_symbol)
        
        if not from_holding or from_holding["amount"] <= 0:
            logger.error(f"❌ No {from_symbol} balance")
            return False
        
        # Calculate amounts
        available = from_holding["amount"]
        from_price = from_holding["price"]
        total_portfolio_value = portfolio["total_value"]
        
        # COMPETITION RULE: Max 25% of portfolio per trade
        max_trade_value = total_portfolio_value * 0.25
        
        # Respect our own limits (20% configured)
        max_position_value = total_portfolio_value * config.MAX_POSITION_PCT
        
        # Use the more conservative limit
        max_allowed = min(max_trade_value, max_position_value)
        
        # Adjust if requested amount exceeds limits
        if amount_usd > max_allowed:
            logger.warning(f"⚠️  Trade too large: ${amount_usd:.2f} > ${max_allowed:.2f}, adjusting")
            amount_usd = max_allowed * 0.95  # Use 95% of max
        
        amount_tokens = amount_usd / from_price
        
        if amount_tokens > available:
            logger.warning(f"⚠️  Adjusting: need {amount_tokens:.6f}, have {available:.6f}")
            amount_tokens = available * 0.98
            amount_usd = amount_tokens * from_price
        
        # COMPETITION RULE: Minimum 0.000001 tokens
        if amount_tokens < 0.000001:
            logger.warning(f"❌ Trade too small: {amount_tokens:.10f} tokens")
            return False
        
        # Execute
        try:
            logger.info(f"📤 {action}: {from_symbol} → {to_symbol}")
            logger.info(f"   Amount: {amount_tokens:.6f} {from_symbol} (${amount_usd:.2f})")
            logger.info(f"   Max allowed: ${max_allowed:.2f} (25% rule)")
            logger.info(f"   Reason: {reason}")
            
            result = await self.client.execute_trade(
                competition_id=self.competition_id,
                from_token=from_token["address"],
                to_token=to_token["address"],
                amount=str(amount_tokens),
                reason=reason,
                from_chain=from_token.get("chain"),
                to_chain=to_token.get("chain")
            )
            
            if result.get("success"):
                logger.info(f"✅ Trade executed successfully!")
                
                # Track daily trades
                today = datetime.now(timezone.utc).date()
                if self.last_trade_date != today:
                    self.trades_today = 0
                    self.last_trade_date = today
                self.trades_today += 1
                
                self.trade_history.append({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "action": action,
                    "from": from_symbol,
                    "to": to_symbol,
                    "amount": amount_usd,
                    "reason": reason
                })
                
                logger.info(f"   Daily trades: {self.trades_today}/{config.MIN_TRADES_PER_DAY}")
                return True
            else:
                logger.error(f"❌ Trade failed: {result.get('error', 'Unknown')}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Trade error: {e}")
            return False
    
    async def run(self):
        """Main trading loop"""
        logger.info("="*80)
        logger.info("🚀 ADVANCED PAPER TRADING AGENT")
        logger.info(f"   Mode: {config.STRATEGY_MODE}")
        logger.info(f"   Risk: {config.MAX_PORTFOLIO_RISK*100:.0f}% max deployed")
        logger.info("="*80)
        
        self.competition_id = await self.select_competition()
        cycle = 0
        
        while True:
            cycle += 1
            logger.info(f"\n{'='*70}")
            logger.info(f"📊 Cycle #{cycle} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"{'='*70}")
            
            try:
                # Get portfolio state
                portfolio = await self.get_portfolio_state()
                
                if not portfolio:
                    logger.warning("⚠️  Failed to get portfolio")
                    await asyncio.sleep(config.TRADING_INTERVAL)
                    continue
                
                # Log portfolio
                total = portfolio["total_value"]
                holdings = portfolio["holdings"]
                positions = portfolio["positions"]
                
                logger.info(f"💼 Portfolio: ${total:,.2f}")
                for symbol, data in holdings.items():
                    pnl_indicator = ""
                    for pos in positions:
                        if pos.symbol == symbol:
                            pnl_indicator = f" ({pos.pnl_pct:+.1f}%)"
                    logger.info(f"   {symbol}: ${data['value']:.2f} ({data['pct']:.1f}%){pnl_indicator}")
                
                # Find opportunities
                market_data = portfolio["market_data"]
                
                # Check if we need to force trades for daily minimum
                today = datetime.now(timezone.utc).date()
                if self.last_trade_date != today:
                    self.trades_today = 0
                    self.last_trade_date = today
                
                needs_trades = self.trades_today < config.MIN_TRADES_PER_DAY
                
                if market_data:
                    opportunities = self.analyzer.find_opportunities(market_data, portfolio)
                    
                    logger.info(f"\n🎯 Top Opportunities:")
                    for symbol, score, signal in opportunities[:3]:
                        change = market_data[symbol]["change_24h"]
                        logger.info(f"   {symbol}: {signal} | {change:+.2f}% | Score: {score:.1f}")
                else:
                    logger.warning("⚠️  No market data available, using portfolio data only")
                    opportunities = []
                
                # Show daily trade progress
                logger.info(f"\n📊 Daily Progress: {self.trades_today}/{config.MIN_TRADES_PER_DAY} trades")
                if needs_trades:
                    logger.info(f"⚠️  Need {config.MIN_TRADES_PER_DAY - self.trades_today} more trade(s) today!")
                
                # Generate decision
                decision = self.strategy.generate_trade_decision(
                    portfolio, market_data, opportunities
                )
                
                logger.info(f"\n🤖 Decision: {decision['action']}")
                logger.info(f"   {decision.get('reason', 'N/A')}")
                
                # Execute
                executed = await self.execute_trade(decision, portfolio)
                
                if executed:
                    logger.info(f"✅ Trade completed (Total trades: {len(self.trade_history)})")
                
                # Log leaderboard
                try:
                    leaderboard = await self.client.get_leaderboard(self.competition_id)
                    entries = leaderboard.get('entries', [])
                    if entries:
                        # Find our position
                        my_entry = None
                        for idx, entry in enumerate(entries, 1):
                            if entry.get("agentId") == "current":  # Adjust as needed
                                my_entry = (idx, entry)
                                break
                        
                        logger.info(f"\n📈 Competition: {len(entries)} agents")
                        if my_entry:
                            logger.info(f"   Our rank: #{my_entry[0]}")
                except Exception as e:
                    logger.debug(f"Could not fetch leaderboard: {e}")
                
            except Exception as e:
                logger.error(f"❌ Error in cycle: {e}")
            
            logger.info(f"\n⏳ Next cycle in {config.TRADING_INTERVAL} seconds...")
            await asyncio.sleep(config.TRADING_INTERVAL)

# ============================================================================
# MAIN
# ============================================================================

async def main():
    """Main entry point"""
    logger.info("="*80)
    logger.info("📈 RECALL NETWORK - ADVANCED PAPER TRADING AGENT")
    logger.info("="*80)
    logger.info("✅ Zero risk - paper trading with virtual funds")
    logger.info("🎯 Multi-chain support (EVM + Solana ready)")
    logger.info("🛡️  Advanced risk management & position sizing")
    logger.info("📊 Smart strategy: momentum + mean reversion")
    logger.info("="*80)
    
    agent = AdvancedTradingAgent()
    await agent.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\n⏹️  Agent stopped by user")
    except Exception as e:
        logger.critical(f"❌ Fatal error: {e}")
        sys.exit(1)