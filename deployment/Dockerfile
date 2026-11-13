FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
 && python -m nltk.downloader -d /usr/local/share/nltk_data punkt punkt_tab stopwords wordnet omw-1.4

COPY . .
EXPOSE 5000
CMD ["python", "app.py"]
