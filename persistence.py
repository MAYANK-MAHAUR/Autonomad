"""
Persistence Manager - Memory-Only State Management
Perfect for Railway deployment - no file I/O
"""
import asyncio
from datetime import datetime, timezone, date
from typing import Dict, Optional
from collections import deque

from logging_manager import get_logger

logger = get_logger("Persistence")


class PersistenceManager:
    """
    Memory-only state management for Railway
    State is lost on restart, but agent rebuilds from portfolio
    """
    
    def __init__(self, state_file: str = None, backup_count: int = 5):
        # Parameters kept for API compatibility, but not used
        self._memory_state: Dict = {}
        self._lock = asyncio.Lock()
        self._state_history: deque = deque(maxlen=10)  # Keep last 10 snapshots
        
        logger.info("💾 Memory-only persistence initialized (Railway compatible)")
    
    async def load_state(self) -> Dict:
        """
        Load state from memory
        On first run, returns empty dict (agent will rebuild from portfolio)
        """
        async with self._lock:
            if self._memory_state:
                logger.info("✅ Loaded state from memory")
                return self._deserialize_state(self._memory_state)
            else:
                logger.info("💾 No cached state. Starting fresh (will rebuild from portfolio)")
                return {}
    
    async def save_state(self, state: Dict):
        """Save state to memory"""
        async with self._lock:
            serialized = self._serialize_state(state)
            serialized['saved_at'] = datetime.now(timezone.utc).isoformat()
            
            # Store in memory
            self._memory_state = serialized
            
            # Keep history for rollback
            self._state_history.append(serialized.copy())
            
            logger.debug(f"💾 State saved to memory")
    
    def get_state_snapshot(self) -> Dict:
        """Get current state snapshot (for monitoring)"""
        return self._memory_state.copy() if self._memory_state else {}
    
    def get_state_history(self) -> list:
        """Get state history (for analysis)"""
        return list(self._state_history)
    
    def clear_state(self):
        """Clear all state (useful for testing)"""
        self._memory_state = {}
        self._state_history.clear()
        logger.info("🗑️ Memory state cleared")
    
    def _serialize_state(self, state: Dict) -> Dict:
        """Serialize state for storage"""
        serialized = {}
        
        # Handle each state component
        for key, value in state.items():
            if key == "metrics" and hasattr(value, 'to_dict'):
                serialized[key] = value.to_dict()
            elif key == "last_trade_date":
                serialized[key] = value.isoformat() if isinstance(value, date) else value
            else:
                serialized[key] = value
        
        return serialized
    
    def _deserialize_state(self, data: Dict) -> Dict:
        """Deserialize state from storage"""
        from models import TradingMetrics
        
        state = {}
        
        # Metrics
        if 'metrics' in data:
            state['metrics'] = TradingMetrics.from_dict(data['metrics'])
        
        # Date handling
        if 'last_trade_date' in data and data['last_trade_date']:
            try:
                state['last_trade_date'] = date.fromisoformat(data['last_trade_date'])
            except (ValueError, TypeError):
                state['last_trade_date'] = None
        
        # Copy other fields
        for key in ['trades_today', 'daily_start_value', 'portfolio_state']:
            if key in data:
                state[key] = data[key]
        
        return state