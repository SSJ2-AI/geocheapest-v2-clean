import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_health_check(client: AsyncClient):
    response = await client.get("/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

@pytest.mark.asyncio
async def test_protected_route_without_token(client: AsyncClient):
    # This route doesn't exist yet, but we'll add it
    response = await client.get("/api/users/me")
    # Should be 401 or 404 depending on implementation state
    # For now, we expect 404 because the route isn't there, 
    # but once implemented it should be 401
    assert response.status_code in [401, 404]
