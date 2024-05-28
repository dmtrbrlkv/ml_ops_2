FROM python:3.10
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY /app /app
EXPOSE 3001
CMD streamlit run Предсказание_оттока.py --server.port 3001