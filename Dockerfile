FROM python:3.11-slim
WORKDIR /opt/nba-props
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY streamlit_app.py nba_props_data.py nba_props_demo.py nba_props_model.py ./
RUN useradd --create-home streamlit
USER streamlit
EXPOSE 8501
CMD ["python", "-m", "streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.headless=true"]
