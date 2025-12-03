#!/usr/bin/env python3
"""
Portfolio Diagnostic Tool
=========================
Shows exactly what's in your portfolio across all chains.
"""

import os
import json
import asyncio
import aiohttp
from dotenv import load_dotenv

load_dotenv()


async def diagnose_portfolio():
    """Show detailed portfolio breakdown"""
    
    api_key = os.getenv("RECALL_API_KEY")
    use_sandbox = os.getenv("RECALL_USE_SANDBOX", "true").lower() == "true"
    competition_id = os.getenv("COMPETITION_ID", "")
    
    base_url = (
        "https://api.sandbox.competitions.recall.network" 
        if use_sandbox 
        else "https://api.competitions.recall.network"
    )
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    print("=" * 80)
    print("🔍 PORTFOLIO DIAGNOSTIC")
    print("=" * 80)
    print()
    
    # Get competitions if no ID provided
    if not competition_id:
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.get(f"{base_url}/api/user/competitions") as resp:
                data = await resp.json()
                competitions = data.get("competitions", [])
                if competitions:
                    competition_id = competitions[0]["id"]
                    print(f"Using competition: {competitions[0].get('name', competition_id)}")
    
    # Get portfolio
    async with aiohttp.ClientSession(headers=headers) as session:
        async with session.get(
            f"{base_url}/api/agent/balances",
            params={"competitionId": competition_id}
        ) as resp:
            portfolio = await resp.json()
    
    if not portfolio.get("success"):
        print(f"❌ Failed to get portfolio: {portfolio.get('error')}")
        return
    
    balances = portfolio.get("balances", [])
    
    print()
    print("📊 ALL BALANCES:")
    print("─" * 80)
    
    total_value = 0
    usdc_total = 0
    trading_assets_total = 0
    
    # Group by type
    stablecoins = []
    trading_assets = []
    
    for balance in balances:
        symbol = balance.get("symbol", "")
        amount = float(balance.get("amount", 0))
        price = float(balance.get("price", 0))
        value = float(balance.get("value", 0))
        chain = balance.get("specificChain", "")
        address = balance.get("tokenAddress", "")
        
        total_value += value
        
        # Categorize
        is_stable = symbol in ["USDC", "USDbC", "DAI", "USDT"]
        
        if is_stable:
            stablecoins.append({
                "symbol": symbol,
                "chain": chain,
                "amount": amount,
                "value": value,
                "address": address
            })
            usdc_total += value
        else:
            trading_assets.append({
                "symbol": symbol,
                "chain": chain,
                "amount": amount,
                "price": price,
                "value": value,
                "address": address
            })
            trading_assets_total += value
    
    # Display stablecoins
    print()
    print("💵 STABLECOINS (Trading Capital):")
    print("─" * 80)
    
    stablecoins.sort(key=lambda x: x["value"], reverse=True)
    for coin in stablecoins:
        if coin["value"] >= 0.01:
            print(f"  {coin['symbol']:8} on {coin['chain']:10} | ${coin['value']:>12,.2f} | {coin['amount']:>15,.6f}")
            print(f"           Address: {coin['address'][:42]}...")
    
    print(f"  {'─' * 76}")
    print(f"  {'TOTAL':8}                  | ${usdc_total:>12,.2f}")
    
    # Display trading assets
    if trading_assets:
        print()
        print("📈 TRADING ASSETS:")
        print("─" * 80)
        
        trading_assets.sort(key=lambda x: x["value"], reverse=True)
        for asset in trading_assets:
            if asset["value"] >= 0.01:
                print(f"  {asset['symbol']:8} on {asset['chain']:10} | ${asset['value']:>12,.2f} | {asset['amount']:>15,.6f} @ ${asset['price']:.4f}")
        
        print(f"  {'─' * 76}")
        print(f"  {'TOTAL':8}                  | ${trading_assets_total:>12,.2f}")
    
    # Summary
    print()
    print("=" * 80)
    print(f"💰 TOTAL PORTFOLIO VALUE: ${total_value:,.2f}")
    print(f"   Stablecoins (tradable): ${usdc_total:,.2f} ({usdc_total/total_value*100:.1f}%)")
    print(f"   Trading Positions:      ${trading_assets_total:,.2f} ({trading_assets_total/total_value*100:.1f}%)")
    print("=" * 80)
    
    # Recommendations
    print()
    print("💡 ANALYSIS:")
    if usdc_total < 100:
        print("   ⚠️  Low USDC balance - may need to sell some positions")
    elif usdc_total < 1000:
        print("   ⚡ Moderate USDC - can make a few trades")
    else:
        print("   ✅ Good USDC balance - ready to trade!")
    
    if usdc_total > 0:
        # Check which chains have USDC
        chains_with_usdc = set(c["chain"] for c in stablecoins if c["value"] >= 1.0)
        print(f"   🌐 USDC available on: {', '.join(sorted(chains_with_usdc))}")
        
        # Cheapest chain recommendation
        if "polygon" in chains_with_usdc:
            print("   💡 Recommend: Use Polygon for lowest gas fees!")
        elif "arbitrum" in chains_with_usdc:
            print("   💡 Recommend: Use Arbitrum for low gas fees!")
        elif "base" in chains_with_usdc:
            print("   💡 Recommend: Use Base for low gas fees!")
    
    print()


if __name__ == "__main__":
    asyncio.run(diagnose_portfolio())