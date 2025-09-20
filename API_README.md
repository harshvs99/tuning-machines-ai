# Investment Analysis API

A FastAPI-based service for analyzing investment pitch decks using AI agents.

## Features

- RESTful API for investment analysis
- Support for multiple agent types (`all`, `agent1`-`agent7`)
- Async processing of pitch deck URLs
- Automatic API documentation with Swagger UI

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the server:
```bash
python main.py
```

The API will be available at `http://localhost:8000`

## API Endpoints

### GET /
Returns API status.

**Response:**
```json
{
  "message": "Investment Analysis API is running"
}
```

### POST /analyze/{agent_type}
Analyzes investment pitch decks using the specified agent.

**Parameters:**
- `agent_type` (path): Agent type to use. Options: `all`, `agent1`, `agent2`, `agent3`, `agent4`, `agent5`, `agent6`, `agent7`

**Request Body:**
```json
{
  "pitchdeck_urls": ["https://example.com/pitch1.pdf", "https://example.com/pitch2.pdf"]
}
```

**Response:**
```json
{
  "status": "success",
  "analyzed_urls": ["https://example.com/pitch1.pdf", "https://example.com/pitch2.pdf"],
  "analysis_count": 2,
  "message": "Investment analysis completed successfully",
  "agent_type": "all"
}
```

## API Documentation

Interactive API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Testing

Run the included tests:
```bash
python test_api.py
```

## Development

The `investing_agent.py` module contains the core analysis functionality and can be extended with additional features as needed.