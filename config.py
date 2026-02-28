"""
ADMDNN Configuration Manager
Handles environment variables, Firebase initialization, and global constants.
"""
import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, firestore

# Load environment variables
load_dotenv()

@dataclass
class ExchangeConfig:
    """Configuration for cryptocurrency exchange connections"""
    name: str
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    sandbox: bool = True  # Start in sandbox mode by default

@dataclass
class ModelConfig:
    """Configuration for neural network models"""
    sequence_length: int = 100
    prediction_horizon: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    dropout_rate: float = 0.2

class ADMDNNConfig:
    """Main configuration manager for the ADMDNN system"""
    
    def __init__(self):
        self._validate_env_vars()
        self._init_firebase()
        self._setup_logging()
        
        # Exchange configurations
        self.exchanges: Dict[str, ExchangeConfig] = {
            'binance': ExchangeConfig(
                name='binance',
                api_key=os.getenv('BINANCE_API_KEY'),
                api_secret=os.getenv('BINANCE_API_SECRET')
            ),
            'coinbase': ExchangeConfig(
                name='coinbase',
                api_key=os.getenv('COINBASE_API_KEY'),
                api_secret=os.getenv('COINBASE_API_SECRET')
            )
        }
        
        # Model configuration
        self.model = ModelConfig()
        
        # Database collections
        self.collections = {
            'market_data': 'market_data',
            'training_jobs': 'training_jobs',
            'strategies': 'trading_strategies',
            'performance': 'strategy_performance',
            'system_logs': 'system_logs'
        }
        
        # Data collection intervals (seconds)
        self.collection_intervals = {
            'tick': 1,
            'minute': 60,
            'hour': 3600,
            'day': 86400
        }
        
        # Risk management
        self.max_position_size = float(os.getenv('MAX_POSITION_SIZE', '0.1'))  # 10% of portfolio
        self.max_daily_loss = float(os.getenv('MAX_DAILY_LOSS', '0.02'))  # 2% max daily loss
        
    def _validate_env_vars(self) -> None:
        """Validate required environment variables"""
        required_vars = [
            'GOOGLE_APPLICATION_CREDENTIALS',
            'FIREBASE_PROJECT_ID'
        ]
        
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")
            
    def _init_firebase(self) -> None:
        """Initialize Firebase Admin SDK"""
        try:
            if not firebase_admin._apps:
                cred_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
                if not os.path.exists(cred_path):
                    raise FileNotFoundError(f"Firebase credentials file not found: {cred_path}")
                    
                cred = credentials.Certificate(cred_path)
                firebase_admin.initialize_app(cred, {
                    'projectId': os.getenv('FIREBASE_PROJECT_ID')
                })
                
            self.db = firestore.client()
            logging.info("Firebase initialized successfully")
            
        except Exception as e:
            logging.error(f"Failed to initialize Firebase: {str(e)}")
            raise
            
    def _setup_logging(self) -> None:
        """Configure application logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('admdnn.log')
            ]
        )
        
    def get_exchange_config(self, exchange_name: str) -> ExchangeConfig:
        """Get configuration for a specific exchange"""
        config = self.exchanges.get(exchange_name)
        if not config:
            raise ValueError(f"Unknown exchange: {exchange_name}")
        return config