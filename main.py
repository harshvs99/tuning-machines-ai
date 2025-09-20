"""
FastAPI application for investment analysis.
"""
import asyncio

from fastapi import FastAPI, HTTPException
from typing import List
from pydantic import BaseModel
import investing_agent_helper as investing_agent

app = FastAPI(title="Investment Analysis API", version="1.0.0")


class AnalysisRequest(BaseModel):
    """Request model for investment analysis."""
    documents_url: str
    company_name: str
    api_key: str


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Investment Analysis API is running"}


@app.post("/analyze/{agent_type}")
async def analyze_investment(agent_type: str, request: AnalysisRequest):
    """
    Analyze investment pitch decks using the specified agent type.
    
    Args:
        agent_type: Either "all" or "agent1" through "agent7"
        request: Request containing list of pitch deck URLs
        
    Returns:
        Analysis results from the investing agent
    """
    # Validate agent_type
    valid_agents = ["all"] + [f"agent{i}" for i in range(1, 8)]
    if agent_type not in valid_agents:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid agent_type. Must be one of: {', '.join(valid_agents)}"
        )
    
    try:
        # Call the investing agent function
        # result = await investing_agent.run_investment_analysis(request.pitchdeck_urls).model_dump_json()
        result = asyncio.run(investing_agent.run_investment_analysis(pitch_deck_urls=investing_agent.fetch_all_required_files(request.documents_url,
                                                                                                                              request.company_name),
                                            company_name=request.company_name)).model_dump_json()
        
        # result["agent_type"] = agent_type
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)