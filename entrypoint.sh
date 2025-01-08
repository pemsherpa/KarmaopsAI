#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Run database migrations and start the Streamlit app
alembic upgrade head && streamlit run main.py