#!/bin/bash
# deploy.sh — run from repo root
# Usage: bash deploy.sh
#
# Prerequisites:
#   gcloud auth login
#   gcloud config set project arize-gemini-api

set -e

PROJECT_ID="arize-gemini-api"
REGION="us-central1"
SERVICE_NAME="prometheus"

echo "▶ Loading env vars from .env..."
source .env

echo "▶ Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
  --source . \
  --region $REGION \
  --project $PROJECT_ID \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --set-env-vars \
"GOOGLE_API_KEY=${GOOGLE_API_KEY},\
PHOENIX_API_KEY=${PHOENIX_API_KEY},\
PHOENIX_COLLECTOR_ENDPOINT=${PHOENIX_COLLECTOR_ENDPOINT},\
PHOENIX_PROJECT_NAME=${PHOENIX_PROJECT_NAME},\
GEMINI_MODEL=${GEMINI_MODEL},\
TAVILY_API_KEY=${TAVILY_API_KEY}"

echo ""
echo "Deployed. Your URL is above — paste it into Devpost."