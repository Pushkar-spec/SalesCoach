import json
import os
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

# Load environment variables from .env if present
load_dotenv()

app = FastAPI(title="Real-Time AI Sales Copilot API")


@app.get('/')
def read_root():
    return {"message": "Hello World"}


@app.get('/health')
def health_check():
    return {
        "status": "ok",
        "service": "sales-copilot-api",
        "environment": os.getenv("ENVIRONMENT", "development"),
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Bidirectional WebSocket endpoint for real-time sales copilot communication.

    Lifecycle:
    1. Accept the incoming client handshake.
    2. Enter an event loop while the client is connected.
    3. Receive incoming text frames and parse as JSON safely.
    4. Validate structure (object, type, message).
    5. Reply with corresponding response ('pong' echoing 'message' or 'error').
    6. Catch WebSocketDisconnect gracefully on connection termination.
    """
    await websocket.accept()
    try:
        while True:
            raw_data = await websocket.receive_text() # raw_data stores data in json format

            # 1. Parse JSON safely
            try:
                payload = json.loads(raw_data) # json.loads() converts JSON text into the corresponding Python object
            except json.JSONDecodeError:       # catches if the JSON is invalid
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON format",
                })
                continue # Skip the rest of this loop iteration and start the next one.

            # 2. Validate that payload is a JSON object (dict)
            if not isinstance(payload, dict):
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid message: expected a JSON object",
                })
                continue

            # 3. Validate message type
            message_type = payload.get("type")  # .get() is a dictionary method used to retrieve a value without raising an error if the key is missing.
            if message_type == "ping":
                await websocket.send_json({
                    "type": "pong",
                    "message": payload.get("message", ""),
                })
            else:
                await websocket.send_json({
                    "type": "error",
                    "message": "Unsupported message type",
                })

    except WebSocketDisconnect:
        # Gracefully handle normal client disconnection
        pass

