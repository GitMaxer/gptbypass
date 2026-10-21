# conftest.py
# Канонические импорты листингов 5-6: `from src.spiker.core... import ...`
# Обеспечиваем корень репозитория в sys.path для pytest.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
