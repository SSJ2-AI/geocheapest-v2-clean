import pytest
from httpx import AsyncClient
from unittest.mock import MagicMock, AsyncMock
import sys
import os

# Add app directory to path so we can import app.main
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app, get_db

@pytest.fixture
def mock_firestore():
    mock_client = AsyncMock()
    mock_collection = MagicMock()
    mock_doc = AsyncMock()
    
    # Setup chain: client.collection().document()
    mock_client.collection.return_value = mock_collection
    mock_collection.document.return_value = mock_doc
    
    # Setup doc.get() to return a mock snapshot
    mock_snapshot = MagicMock()
    mock_snapshot.exists = True
    mock_snapshot.to_dict.return_value = {}
    mock_doc.get.return_value = mock_snapshot
    
    return mock_client

@pytest.fixture
async def client(mock_firestore):
    # Override the dependency
    app.dependency_overrides[get_db] = lambda: mock_firestore
    
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac
    
    # Clean up
    app.dependency_overrides = {}
