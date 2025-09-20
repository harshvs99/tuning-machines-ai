"""
Simple test script to verify the FastAPI endpoint functionality.
"""
import asyncio
import investing_agent
from main import app
from fastapi.testclient import TestClient

def test_investing_agent():
    """Test the investing_agent module directly."""
    async def run_test():
        urls = ["https://example.com/pitch1.pdf", "https://example.com/pitch2.pdf"]
        result = await investing_agent.run_investment_analysis(urls)
        print("Direct investing_agent test:", result)
        assert result["status"] == "success"
        assert result["analysis_count"] == 2
        assert result["analyzed_urls"] == urls
    
    asyncio.run(run_test())
    print("✓ investing_agent module test passed")

def test_fastapi_endpoints():
    """Test the FastAPI endpoints."""
    client = TestClient(app)
    
    # Test root endpoint
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "Investment Analysis API is running"
    print("✓ Root endpoint test passed")
    
    # Test analysis endpoint with 'all'
    response = client.post(
        "/analyze/all",
        json={"pitchdeck_urls": ["https://example.com/pitch1.pdf"]}
    )
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "success"
    assert result["agent_type"] == "all"
    print("✓ Analysis endpoint with 'all' test passed")
    
    # Test analysis endpoint with 'agent5'
    response = client.post(
        "/analyze/agent5",
        json={"pitchdeck_urls": ["https://example.com/pitch1.pdf", "https://example.com/pitch2.pdf"]}
    )
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "success"
    assert result["agent_type"] == "agent5"
    assert result["analysis_count"] == 2
    print("✓ Analysis endpoint with 'agent5' test passed")
    
    # Test invalid agent type
    response = client.post(
        "/analyze/invalid",
        json={"pitchdeck_urls": ["https://example.com/pitch1.pdf"]}
    )
    assert response.status_code == 400
    assert "Invalid agent_type" in response.json()["detail"]
    print("✓ Invalid agent type test passed")

if __name__ == "__main__":
    print("Running tests...")
    test_investing_agent()
    test_fastapi_endpoints()
    print("All tests passed! ✅")