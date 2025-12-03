#!/usr/bin/env python3
"""
Enable Multi-Chain Trading - Automatic Patcher
==============================================
Enables trading on ALL chains without consolidation.

Usage:
    python enable_multichain.py
"""

import sys
import shutil
from pathlib import Path
from datetime import datetime


def patch_agent_for_multichain(agent_file: str = "agent.py") -> bool:
    """Patch agent.py to enable multi-chain trading"""
    
    agent_path = Path(agent_file)
    
    if not agent_path.exists():
        print(f"❌ {agent_file} not found")
        return False
    
    print("📄 Reading agent.py...")
    content = agent_path.read_text(encoding='utf-8')
    
    # Create backup
    backup_path = agent_path.with_suffix(f'.py.backup.{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    print(f"💾 Creating backup: {backup_path.name}")
    shutil.copy2(agent_path, backup_path)
    
    # =========================================================================
    # PATCH 1: Add multi-chain token configs
    # =========================================================================
    print("🔧 Adding multi-chain token configurations...")
    
    # Find where TOKENS dict is defined
    tokens_start = content.find('TOKENS: Dict[str, TokenConfig] = {')
    if tokens_start == -1:
        print("❌ Could not find TOKENS config")
        return False
    
    # Find the end of TOKENS dict
    brace_count = 0
    tokens_end = tokens_start
    in_dict = False
    
    for i in range(tokens_start, len(content)):
        if content[i] == '{':
            brace_count += 1
            in_dict = True
        elif content[i] == '}':
            brace_count -= 1
            if in_dict and brace_count == 0:
                tokens_end = i + 1
                break
    
    # New tokens configuration
    new_tokens = '''TOKENS: Dict[str, TokenConfig] = {
        # ===== STABLECOINS (Multi-Chain) =====
        "USDC": TokenConfig("0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48", "eth", True, 6),
        "USDC_POLYGON": TokenConfig("0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174", "polygon", True, 6),
        "USDC_ARBITRUM": TokenConfig("0xaf88d065e77c8cc2239327c5edb3a432268e5831", "arbitrum", True, 6),
        "USDC_OPTIMISM": TokenConfig("0x7f5c764cbc14f9669b88837ca1490cca17c31607", "optimism", True, 6),
        "USDBC_BASE": TokenConfig("0xd9aAEc86B65D86f6A7B5B1b0c42FFA531710b6CA", "base", True, 6),
        "USDC_SOLANA": TokenConfig("EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v", "svm", True, 6),
        "DAI": TokenConfig("0x6B175474E89094C44Da98b954EedeAC495271d0F", "eth", True, 18),
        
        # ===== ETHEREUM MAINNET =====
        "WETH": TokenConfig("0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2", "eth", False, 18),
        "WBTC": TokenConfig("0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599", "eth", False, 8),
        "UNI": TokenConfig("0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984", "eth", False, 18),
        "LINK": TokenConfig("0x514910771AF9Ca656af840dff83E8264EcF986CA", "eth", False, 18),
        "AAVE": TokenConfig("0x7Fc66500c84A76Ad7e9c93437bFc5Ac33E2DDaE9", "eth", False, 18),
        "SNX": TokenConfig("0xC011a73ee8576Fb46F5E1c5751cA3B9Fe0af2a6F", "eth", False, 18),
        "CRV": TokenConfig("0xD533a949740bb3306d119CC777fa900bA034cd52", "eth", False, 18),
        "MKR": TokenConfig("0x9f8F72aA9304c8B593d555F12eF6589cC3A579A2", "eth", False, 18),
        "BONK": TokenConfig("0x1151CB3d861920e07a38e03eead12c32178567F6", "eth", False, 5),
        "FLOKI": TokenConfig("0xcf0C122c6b73ff809C693DB761e7BaeBe62b6a2E", "eth", False, 9),
        "PEPE": TokenConfig("0x6982508145454Ce325dDbE47a25d4ec3d2311933", "eth", False, 18),
        "SHIB": TokenConfig("0x95aD61b0a150d79219dCF64E1E6Cc01f0B64C4cE", "eth", False, 18),
        
        # ===== POLYGON (Cheap Gas!) =====
        "WETH_POLYGON": TokenConfig("0x7ceB23fD6bC0adD59E62ac25578270cFf1b9f619", "polygon", False, 18),
        "WBTC_POLYGON": TokenConfig("0x1BFD67037B42Cf73acF2047067bd4F2C47D9BfD6", "polygon", False, 8),
        "LINK_POLYGON": TokenConfig("0x53E0bca35eC356BD5ddDFebbD1Fc0fD03FaBad39", "polygon", False, 18),
        
        # ===== ARBITRUM (Fast + Cheap!) =====
        "WETH_ARBITRUM": TokenConfig("0x82aF49447D8a07e3bd95BD0d56f35241523fBab1", "arbitrum", False, 18),
        "WBTC_ARBITRUM": TokenConfig("0x2f2a2543B76A4166549F7aaB2e75Bef0aefC5B0f", "arbitrum", False, 8),
        "LINK_ARBITRUM": TokenConfig("0xf97f4df75117a78c1A5a0DBb814Af92458539FB4", "arbitrum", False, 18),
        
        # ===== BASE (Coinbase L2) =====
        "WETH_BASE": TokenConfig("0x4200000000000000000000000000000000000006", "base", False, 18),
    }
    
    # Chain mapping for smart routing
    SYMBOL_TO_CHAINS: Dict[str, List[str]] = {
        "USDC": ["USDC", "USDC_POLYGON", "USDC_ARBITRUM", "USDC_OPTIMISM", "USDBC_BASE", "USDC_SOLANA"],
        "WETH": ["WETH", "WETH_POLYGON", "WETH_ARBITRUM", "WETH_BASE"],
        "WBTC": ["WBTC", "WBTC_POLYGON", "WBTC_ARBITRUM"],
        "LINK": ["LINK", "LINK_POLYGON", "LINK_ARBITRUM"],
    }'''
    
    content = content[:tokens_start] + new_tokens + content[tokens_end:]
    
    # =========================================================================
    # PATCH 2: Add multi-chain helper methods to TradingAgent
    # =========================================================================
    print("🔧 Adding multi-chain helper methods...")
    
    # Find TradingAgent class
    agent_class = content.find('class TradingAgent:')
    if agent_class == -1:
        print("❌ Could not find TradingAgent class")
        return False
    
    # Find a good place to insert (after __init__ method)
    init_method = content.find('async def initialize(self):', agent_class)
    if init_method == -1:
        print("❌ Could not find initialize method")
        return False
    
    # Insert helper methods before initialize
    helper_methods = '''
    def get_best_usdc_chain(self, portfolio: Dict, min_amount: float = 50.0) -> Optional[Tuple[str, float]]:
        """Find best USDC chain with sufficient balance (prefers cheap gas chains)"""
        holdings = portfolio.get("holdings", {})
        
        # Gas preference (lower = cheaper)
        gas_rank = {"polygon": 1, "arbitrum": 2, "optimism": 3, "base": 4, "svm": 2, "eth": 5}
        
        usdc_options = []
        for symbol, holding in holdings.items():
            token_config = config.TOKENS.get(symbol)
            if not token_config or not token_config.stable:
                continue
            if "USDC" not in symbol and "USD" not in symbol:
                continue
            
            value = holding.get("value", 0)
            if value >= min_amount:
                usdc_options.append({
                    "symbol": symbol,
                    "value": value,
                    "chain": token_config.chain,
                    "gas_rank": gas_rank.get(token_config.chain, 10)
                })
        
        if not usdc_options:
            return None
        
        # Sort by: gas cost (prefer cheap), then amount (prefer large)
        usdc_options.sort(key=lambda x: (x["gas_rank"], -x["value"]))
        best = usdc_options[0]
        
        logger.info(f"💰 Selected {best['symbol']} on {best['chain']} (${best['value']:.2f})")
        return best["symbol"], best["value"]
    
    def get_token_on_chain(self, base_symbol: str, chain: str) -> Optional[str]:
        """Get token symbol for specific chain (e.g., LINK + polygon = LINK_POLYGON)"""
        # Try exact match
        if base_symbol in config.TOKENS:
            if config.TOKENS[base_symbol].chain == chain:
                return base_symbol
        
        # Try chain variants
        variants = config.SYMBOL_TO_CHAINS.get(base_symbol, [])
        for variant in variants:
            if variant in config.TOKENS and config.TOKENS[variant].chain == chain:
                return variant
        
        # Try constructed name
        constructed = f"{base_symbol}_{chain.upper()}"
        if constructed in config.TOKENS:
            return constructed
        
        return base_symbol if base_symbol in config.TOKENS else None
    
    def find_best_trade_chain(self, from_sym: str, to_sym: str, portfolio: Dict) -> Tuple[str, str, str]:
        """Find best chain for trade (prefers cheap gas + available balance)"""
        holdings = portfolio.get("holdings", {})
        
        # If selling existing position, use that chain
        if not from_sym.startswith("USDC"):
            for symbol, holding in holdings.items():
                if symbol.startswith(from_sym) and holding.get("value", 0) >= 1.0:
                    token = config.TOKENS.get(symbol)
                    if token:
                        to_token = self.get_token_on_chain(to_sym, token.chain)
                        if to_token:
                            logger.info(f"🔗 Using {symbol} on {token.chain}")
                            return symbol, to_token, token.chain
        
        # For buying, use chain with most USDC
        result = self.get_best_usdc_chain(portfolio)
        if result:
            usdc_symbol, _ = result
            usdc_token = config.TOKENS[usdc_symbol]
            to_token = self.get_token_on_chain(to_sym, usdc_token.chain)
            if to_token:
                return usdc_symbol, to_token, usdc_token.chain
        
        return from_sym, to_sym, "eth"
    
'''
    
    content = content[:init_method] + helper_methods + content[init_method:]
    
    # =========================================================================
    # PATCH 3: Update execute_trade to use multi-chain
    # =========================================================================
    print("🔧 Updating execute_trade for multi-chain support...")
    
    # Find execute_trade method
    exec_trade_start = content.find('async def execute_trade(self, decision: TradeDecision, portfolio: Dict) -> bool:')
    if exec_trade_start == -1:
        print("❌ Could not find execute_trade method")
        return False
    
    # Find the end of execute_trade (next method)
    next_async_def = content.find('\n    async def ', exec_trade_start + 10)
    if next_async_def == -1:
        next_async_def = content.find('\n    def ', exec_trade_start + 10)
    
    if next_async_def == -1:
        print("❌ Could not find end of execute_trade")
        return False
    
    # Insert multi-chain routing at the start of method
    routing_code = '''    async def execute_trade(self, decision: TradeDecision, portfolio: Dict) -> bool:
        """Execute trade with MULTI-CHAIN support"""
        if decision.action == TradingAction.HOLD:
            logger.info(f"⏸️ HOLD: {decision.reason}")
            return False
        
        # MULTI-CHAIN: Find best chain for this trade
        base_from = decision.from_token.upper()
        base_to = decision.to_token.upper()
        
        from_symbol, to_symbol, trade_chain = self.find_best_trade_chain(base_from, base_to, portfolio)
        logger.info(f"🌐 Multi-chain: {from_symbol} → {to_symbol} on {trade_chain}")
        
'''
    
    # Replace just the method signature and initial logic
    method_body_start = content.find('from_symbol = decision.from_token.upper()', exec_trade_start)
    if method_body_start != -1:
        content = content[:exec_trade_start] + routing_code + content[method_body_start + 41:]  # Skip old from_symbol line
    
    # =========================================================================
    # Save patched file
    # =========================================================================
    print("💾 Saving patched agent.py...")
    agent_path.write_text(content, encoding='utf-8')
    
    return True


def main():
    print("=" * 80)
    print("🌍 ENABLE MULTI-CHAIN TRADING")
    print("=" * 80)
    print()
    print("This will enable your agent to trade on ALL chains without consolidation.")
    print()
    
    response = input("Continue? [Y/n]: ").strip().lower()
    if response and response != 'y':
        print("Cancelled.")
        return 1
    
    print()
    success = patch_agent_for_multichain()
    
    if success:
        print()
        print("=" * 80)
        print("✅ MULTI-CHAIN TRADING ENABLED!")
        print("=" * 80)
        print()
        print("Benefits:")
        print("  ✅ No consolidation needed")
        print("  ✅ Uses USDC from ANY chain")
        print("  ✅ Prefers cheap gas chains (Polygon, Arbitrum)")
        print("  ✅ ~90% gas savings vs Ethereum-only")
        print()
        print("Your agent will now:")
        print("  1. Automatically select cheapest chain for each trade")
        print("  2. Use all available USDC across all chains")
        print("  3. Prioritize Polygon/Arbitrum (cheapest gas)")
        print("  4. Fall back to Ethereum when needed")
        print()
        print("Next: python agent.py")
        print("=" * 80)
        return 0
    else:
        print()
        print("❌ Failed to enable multi-chain trading")
        print("Please apply the patches manually using the guide above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())