#!/bin/bash
if [ "$SERVICE_TYPE" = "streamlit" ]; then
    echo "Starting Streamlit Dashboard..."
    exec streamlit run app.py --server.port=8501 --server.address=0.0.0.0
else
    echo "Starting FastAPI Service..."
    exec uvicorn api_app:app --host 0.0.0.0 --port 8000
fi
