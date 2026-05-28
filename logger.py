import logging
import sys
from datetime import datetime

# Create logs directory if needed
import os
os.makedirs("logs", exist_ok=True)

# Configure logging
log_file = f"logs/detector_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)