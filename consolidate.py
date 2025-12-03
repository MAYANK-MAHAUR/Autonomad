#!/usr/bin/env python3
"""
USDC Chain Consolidator for Recall Network Trading Agent
========================================================
Consolidates all USDC balances from multiple chains to Ethereum mainnet
before starting the trading agent.

Run this before starting agent.py to ensure all capital is available for trading.

Usage:
    python consolidate_usdc.py [--dry-run] [--min-amount MIN]
"""

import os
import sys
import json
import asyncio
import argparse
from typing import List, Dict, Optional
from dataclasses import dataclass
from decimal import Decimal

import aiohttp
from aiohttp import ClientTimeout
from dotenv import load_dotenv

load_dotenv()


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ChainConfig:
    """Configuration for a specific chain"""
    name: str
    chain_id: str
    usdc_address: str
    usdc_symbol: str
    decimals: int = 6


# Supported chains and their USDC token addresses
CHAIN_CONFIGS = {
    "eth": ChainConfig(
        name="Ethereum",
        chain_id="eth",
        usdc_address="0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
        usdc_symbol="USDC",
        decimals=6
    ),
    "polygon": ChainConfig(
        name="Polygon",
        chain_id="polygon",
        usdc_address="0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174",
        usdc_symbol="USDC",
        decimals=6
    ),
    "arbitrum": ChainConfig(
        name="Arbitrum",
        chain_id="arbitrum",
        usdc_address="0xaf88d065e77c8cc2239327c5edb3a432268e5831",
        usdc_symbol="USDC",
        decimals=6
    ),
    "optimism": ChainConfig(
        name="Optimism",
        chain_id="optimism",
        usdc_address="0x7f5c764cbc14f9669b88837ca1490cca17c31607",
        usdc_symbol="USDC",
        decimals=6
    ),
    "base": ChainConfig(
        name="Base",
        chain_id="base",
        usdc_address="0xd9aAEc86B65D86f6A7B5B1b0c42FFA531710b6CA",
        usdc_symbol="USDbC",
        decimals=6
    ),
    "svm": ChainConfig(
        name="Solana",
        chain_id="svm",
        usdc_address="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
        usdc_symbol="USDC",
        decimals=6
    ),
}

TARGET_CHAIN = "eth"  # Consolidate everything to Ethereum


# ============================================================================
# RECALL API CLIENT
# ============================================================================

class ConsolidationClient:
    """Lightweight API client for USDC consolidation"""
    
    def __init__(self, api_key: str, use_sandbox: bool = True):
        self.api_key = api_key
        self.base_url = (
            "https://api.sandbox.competitions.recall.network"
            if use_sandbox
            else "https://api.competitions.recall.network"
        )
        
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=ClientTimeout(total=30),
            headers=self.headers
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_portfolio(self, competition_id: str) -> Dict:
        """Get agent portfolio"""
        async with self.session.get(
            f"{self.base_url}/api/agent/balances",
            params={"competitionId": competition_id}
        ) as response:
            response.raise_for_status()
            return await response.json()
    
    async def execute_trade(
        self,
        competition_id: str,
        from_token: str,
        to_token: str,
        amount: str,
        from_chain: str,
        to_chain: str,
        reason: str = "USDC consolidation to Ethereum"
    ) -> Dict:
        """Execute a trade"""
        payload = {
            "competitionId": competition_id,
            "fromToken": from_token,
            "toToken": to_token,
            "amount": amount,
            "fromChain": from_chain,
            "toChain": to_chain,
            "reason": reason
        }
        
        async with self.session.post(
            f"{self.base_url}/api/trade/execute",
            json=payload
        ) as response:
            response.raise_for_status()
            return await response.json()
    
    async def get_user_competitions(self) -> Dict:
        """Get user's competitions"""
        async with self.session.get(
            f"{self.base_url}/api/user/competitions"
        ) as response:
            response.raise_for_status()
            return await response.json()


# ============================================================================
# CONSOLIDATION LOGIC
# ============================================================================

class USDCConsolidator:
    """Main consolidation coordinator"""
    
    def __init__(
        self,
        api_key: str,
        competition_id: Optional[str] = None,
        use_sandbox: bool = True,
        min_amount: float = 1.0,
        dry_run: bool = False
    ):
        self.api_key = api_key
        self.competition_id = competition_id
        self.use_sandbox = use_sandbox
        self.min_amount = min_amount
        self.dry_run = dry_run
    
    async def run(self):
        """Execute consolidation"""
        print("=" * 80)
        print("🔄 USDC CHAIN CONSOLIDATOR")
        print("=" * 80)
        print(f"Environment: {'SANDBOX' if self.use_sandbox else 'PRODUCTION'}")
        print(f"Target Chain: {CHAIN_CONFIGS[TARGET_CHAIN].name}")
        print(f"Minimum Amount: ${self.min_amount:.2f}")
        print(f"Mode: {'DRY RUN' if self.dry_run else 'LIVE'}")
        print("=" * 80)
        print()
        
        async with ConsolidationClient(self.api_key, self.use_sandbox) as client:
            # Get competition ID
            competition_id = await self._get_competition_id(client)
            
            # Get portfolio
            print("📊 Fetching portfolio...")
            portfolio = await client.get_portfolio(competition_id)
            
            if not portfolio.get("success"):
                print(f"❌ Failed to fetch portfolio: {portfolio.get('error')}")
                return False
            
            # Analyze balances
            balances = portfolio.get("balances", [])
            consolidation_plan = self._analyze_balances(balances)
            
            if not consolidation_plan:
                print("✅ All USDC already on Ethereum mainnet!")
                return True
            
            # Display plan
            self._display_plan(consolidation_plan)
            
            # Execute consolidation
            if self.dry_run:
                print("\n🔍 DRY RUN MODE - No trades executed")
                return True
            
            print("\n🚀 Executing consolidation trades...")
            success = await self._execute_consolidation(client, competition_id, consolidation_plan)
            
            if success:
                print("\n✅ Consolidation complete!")
                await self._display_final_state(client, competition_id)
            else:
                print("\n⚠️ Consolidation completed with some errors")
            
            return success
    
    async def _get_competition_id(self, client: ConsolidationClient) -> str:
        """Get competition ID"""
        if self.competition_id:
            print(f"✅ Using competition ID: {self.competition_id}\n")
            return self.competition_id
        
        print("🔍 Finding active competition...")
        user_comps = await client.get_user_competitions()
        competitions = user_comps.get("competitions", [])
        
        if not competitions:
            raise ValueError("No competitions found")
        
        # Prefer active competitions
        active = [c for c in competitions if c.get("status") == "active"]
        comp = active[0] if active else competitions[0]
        
        comp_id = comp.get("id")
        print(f"✅ Selected: {comp.get('name', comp_id)}\n")
        return comp_id
    
    def _analyze_balances(self, balances: List[Dict]) -> List[Dict]:
        """Analyze balances and create consolidation plan"""
        consolidation_plan = []
        
        for balance in balances:
            symbol = balance.get("symbol", "")
            chain = balance.get("specificChain", "")
            address = balance.get("tokenAddress", "")
            amount = float(balance.get("amount", 0))
            value = float(balance.get("value", 0))
            
            # Skip if not USDC/USDbC
            if symbol not in ["USDC", "USDbC"]:
                continue
            
            # Skip if already on target chain
            if chain == TARGET_CHAIN:
                continue
            
            # Skip if below minimum
            if value < self.min_amount:
                continue
            
            # Find chain config
            chain_config = CHAIN_CONFIGS.get(chain)
            if not chain_config:
                print(f"⚠️ Warning: Unknown chain '{chain}', skipping")
                continue
            
            consolidation_plan.append({
                "from_chain": chain,
                "from_chain_name": chain_config.name,
                "from_token": address,
                "from_symbol": symbol,
                "amount": amount,
                "value": value,
                "decimals": chain_config.decimals
            })
        
        return consolidation_plan
    
    def _display_plan(self, plan: List[Dict]):
        """Display consolidation plan"""
        print("\n📋 CONSOLIDATION PLAN")
        print("─" * 80)
        
        total_value = sum(item["value"] for item in plan)
        
        for i, item in enumerate(plan, 1):
            print(f"{i}. {item['from_chain_name']:12} → Ethereum")
            print(f"   Amount: {item['amount']:,.6f} {item['from_symbol']}")
            print(f"   Value:  ${item['value']:,.2f}")
            print()
        
        print("─" * 80)
        print(f"Total to consolidate: ${total_value:,.2f}")
        print("─" * 80)
    
    async def _execute_consolidation(
        self,
        client: ConsolidationClient,
        competition_id: str,
        plan: List[Dict]
    ) -> bool:
        """Execute consolidation trades"""
        target_chain_config = CHAIN_CONFIGS[TARGET_CHAIN]
        all_success = True
        
        for i, item in enumerate(plan, 1):
            print(f"\n[{i}/{len(plan)}] Consolidating {item['from_chain_name']}...")
            
            # Format amount with proper decimals (leave 0.5% buffer for fees)
            amount = item["amount"] * 0.995
            amount_str = f"{amount:.{item['decimals']}f}"
            
            print(f"   From: {item['from_symbol']} on {item['from_chain_name']}")
            print(f"   To:   USDC on Ethereum")
            print(f"   Amount: {amount_str} (~${item['value'] * 0.995:.2f})")
            
            try:
                result = await client.execute_trade(
                    competition_id=competition_id,
                    from_token=item["from_token"],
                    to_token=target_chain_config.usdc_address,
                    amount=amount_str,
                    from_chain=item["from_chain"],
                    to_chain=TARGET_CHAIN,
                    reason=f"Consolidating USDC from {item['from_chain_name']} to Ethereum"
                )
                
                if result.get("success"):
                    print("   ✅ Success!")
                else:
                    error = result.get("error", "Unknown error")
                    print(f"   ❌ Failed: {error}")
                    all_success = False
                
                # Rate limiting - wait between trades
                if i < len(plan):
                    await asyncio.sleep(2)
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                all_success = False
        
        return all_success
    
    async def _display_final_state(self, client: ConsolidationClient, competition_id: str):
        """Display final portfolio state"""
        print("\n📊 FINAL PORTFOLIO STATE")
        print("─" * 80)
        
        portfolio = await client.get_portfolio(competition_id)
        balances = portfolio.get("balances", [])
        
        usdc_balances = []
        for balance in balances:
            symbol = balance.get("symbol", "")
            if symbol in ["USDC", "USDbC"]:
                usdc_balances.append({
                    "chain": balance.get("specificChain", ""),
                    "symbol": symbol,
                    "amount": float(balance.get("amount", 0)),
                    "value": float(balance.get("value", 0))
                })
        
        # Sort by value descending
        usdc_balances.sort(key=lambda x: x["value"], reverse=True)
        
        total_usdc = sum(b["value"] for b in usdc_balances)
        
        for balance in usdc_balances:
            if balance["value"] >= 0.01:  # Only show balances > $0.01
                chain_name = CHAIN_CONFIGS.get(balance["chain"], ChainConfig("Unknown", "", "", "")).name
                indicator = "✅" if balance["chain"] == TARGET_CHAIN else "⚠️"
                print(f"{indicator} {chain_name:12} | ${balance['value']:>10,.2f} | {balance['amount']:,.6f} {balance['symbol']}")
        
        print("─" * 80)
        print(f"Total USDC: ${total_usdc:,.2f}")
        print("─" * 80)


# ============================================================================
# CLI
# ============================================================================

async def main():
    parser = argparse.ArgumentParser(
        description="Consolidate USDC from multiple chains to Ethereum mainnet"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing trades"
    )
    parser.add_argument(
        "--min-amount",
        type=float,
        default=1.0,
        help="Minimum USDC amount to consolidate (default: $1.00)"
    )
    parser.add_argument(
        "--competition-id",
        type=str,
        help="Specific competition ID to use"
    )
    parser.add_argument(
        "--production",
        action="store_true",
        help="Use production environment (default: sandbox)"
    )
    
    args = parser.parse_args()
    
    # Get API key
    api_key = os.getenv("RECALL_API_KEY")
    if not api_key:
        print("❌ Error: RECALL_API_KEY not found in environment")
        print("Please set it in your .env file or environment variables")
        return 1
    
    # Check competition ID
    competition_id = args.competition_id or os.getenv("COMPETITION_ID")
    
    # Create consolidator
    consolidator = USDCConsolidator(
        api_key=api_key,
        competition_id=competition_id,
        use_sandbox=not args.production,
        min_amount=args.min_amount,
        dry_run=args.dry_run
    )
    
    # Run consolidation
    try:
        success = await consolidator.run()
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))