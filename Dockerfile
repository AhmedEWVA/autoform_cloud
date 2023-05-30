FROM python:3-slim
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
EXPOSE 8001
CMD python ./app.py