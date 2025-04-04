﻿from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import redis
import time
import numpy as np
import torch
import torch.nn as nn
import logging
import asyncio
from typing import List, Dict
import json
import os
from pydantic import BaseModel
from cryptography.fernet import Fernet
import base64
import requests

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración de Redis (usar variables de entorno para producción)
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
REDIS_USERNAME = os.getenv("REDIS_USERNAME", "default")  # Añadir usuario predeterminado

# Conectar a Redis
try:
    r = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        username=REDIS_USERNAME,  # Especificar el usuario
        password=REDIS_PASSWORD
    )
    r.ping()  # Probar la conexión
    logger.info("Conectado a Redis correctamente.")
except redis.ConnectionError as e:
    logger.error(f"Error al conectar a Redis: {e}")
    raise Exception("No se pudo conectar a Redis. Asegúrate de que el servicio esté corriendo.")

# Configuración Anti-DDoS
RATE_LIMIT = 50
WINDOW = 60

# Almacenamiento de datos para IA y reportes
traffic_data = []
reports = []

# Modelo de IA con PyTorch
class ThreatModel(nn.Module):
    def __init__(self):
        super(ThreatModel, self).__init__()
        self.layer1 = nn.Linear(4, 16)  # 4 métricas: request_count, mouse_movements, time_delta, timestamp
        self.layer2 = nn.Linear(16, 8)
        self.layer3 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.layer3(x))
        return x

model = ThreatModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()
trained = False
model.eval()  # Modo evaluación inicial

def train_model(data: List[List[float]], labels: List[int]):
    global trained
    if len(data) < 100:
        return
    X = torch.tensor(np.array(data), dtype=torch.float32)
    y = torch.tensor(np.array(labels), dtype=torch.float32).reshape(-1, 1)
    model.train()
    for _ in range(5):  # 5 épocas
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
    trained = True
    model.eval()
    logger.info("Modelo PyTorch entrenado.")

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host
    key = f"rate_limit:{client_ip}"
    
    current = r.get(key)
    if current is None:
        r.setex(key, WINDOW, 1)
        request_count = 1
    else:
        request_count = int(current) + 1
        r.incr(key)

    timestamp = time.time()
    traffic_data.append([request_count, 0, 0, timestamp % 60])  # Placeholder para métricas adicionales
    if len(traffic_data) > 100:
        labels = [1 if x[0] > RATE_LIMIT else 0 for x in traffic_data[-100:]]
        train_model(traffic_data[-100:], labels)

    if request_count > RATE_LIMIT:
        logger.warning(f"IP {client_ip} excedió el límite: {request_count} solicitudes")
        raise HTTPException(status_code=429, detail="Demasiadas solicitudes")

    response = await call_next(request)
    return response

@app.get("/status")
async def status():
    return {"message": "Servidor Anti-DDoS activo"}

@app.post("/report")
async def report_activity(request: Request):
    data = await request.json()
    request_count = data.get("request_count", 0)
    mouse_movements = data.get("mouse_movements", 0)
    time_delta = data.get("time_delta", 0)
    action_type = data.get("action_type", "unknown")
    timestamp = data.get("timestamp", time.time())

    # Almacenar para IA
    metrics = [request_count, mouse_movements, time_delta, timestamp % 60]
    traffic_data.append(metrics)
    
    # Predicción con PyTorch
    prediction = None
    if trained and len(traffic_data) > 50:
        X = torch.tensor(np.array([metrics]), dtype=torch.float32)
        with torch.no_grad():
            prediction = model(X).item()
        is_threat = prediction > 0.5
        if is_threat:
            logger.warning(f"Anomalía avanzada detectada: {data}, Predicción: {prediction}")

    # Guardar reporte
    report = {
        "timestamp": timestamp,
        "request_count": request_count,
        "mouse_movements": mouse_movements,
        "time_delta": time_delta,
        "action_type": action_type,
        "prediction": float(prediction) if prediction is not None else None,
        "ip": data.get("ip", "simulated_ip")
    }
    reports.append(report)
    if len(reports) > 1000:  # Limitar a 1000 reportes
        reports.pop(0)

    logger.info(f"Reporte recibido: {data}")
    return {"status": "Reporte recibido", "threat_detected": prediction > 0.5 if prediction else False}

@app.get("/reports", response_model=List[Dict])
async def get_reports():
    return reports

# Modelo para validar los datos de entrada del endpoint /encrypt-and-upload
class EncryptUploadRequest(BaseModel):
    data: str
    userAddress: str
    timestamp: int

# Configuración de la clave de encriptación
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY")
if not ENCRYPTION_KEY:
    raise Exception("La variable de entorno ENCRYPTION_KEY no está definida.")
# Convertir la clave a un formato compatible con Fernet (base64)
try:
    ENCRYPTION_KEY = base64.urlsafe_b64encode(ENCRYPTION_KEY.encode()[:32].ljust(32, b'\0'))
    fernet = Fernet(ENCRYPTION_KEY)
except Exception as e:
    logger.error(f"Error al configurar la clave de encriptación: {e}")
    raise Exception("Error al configurar la clave de encriptación.")

# Configuración de la API Key de Fleek
FLEEK_API_KEY = os.getenv("FLEEK_API_KEY")
if not FLEEK_API_KEY:
    raise Exception("La variable de entorno FLEEK_API_KEY no está definida.")

# Endpoint para encriptar y subir a IPFS
@app.post("/encrypt-and-upload")
async def encrypt_and_upload(request: EncryptUploadRequest):
    try:
        # Extraer los datos del cuerpo de la solicitud
        data = request.data
        user_address = request.userAddress
        timestamp = request.timestamp

        # Validar los datos
        if not data or not user_address or not timestamp:
            raise HTTPException(status_code=400, detail="Faltan datos: data, userAddress y timestamp son requeridos")

        # Encriptar los datos usando Fernet
        encrypted_data = fernet.encrypt(data.encode()).decode()

        # Preparar los datos para subir a IPFS
        files = {
            'file': ('encrypted_data.txt', encrypted_data.encode())
        }
        headers = {
            'Authorization': f'Bearer {FLEEK_API_KEY}'
        }

        # Subir el archivo a IPFS usando la API de Fleek
        response = requests.post(
            'https://ipfs.fleek.co/api/v0/add',
            files=files,
            headers=headers
        )

        # Verificar la respuesta de Fleek
        if response.status_code != 200:
            logger.error(f"Error al subir a IPFS: {response.text}")
            raise HTTPException(status_code=500, detail=f"Error al subir a IPFS: {response.text}")

        # Extraer el CID de la respuesta
        result = response.json()
        cid = result.get('Hash')
        if not cid:
            raise HTTPException(status_code=500, detail="No se pudo obtener el CID de IPFS")

        logger.info(f"Archivo subido a IPFS con CID: {cid}")
        return {"cid": cid}

    except Exception as e:
        logger.error(f"Error en /encrypt-and-upload: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error al subir a IPFS: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))  # Usar el puerto proporcionado por el entorno (necesario para Render)
    uvicorn.run(app, host="0.0.0.0", port=port)