FROM python:3.9-buster
WORKDIR /app

RUN pip install --upgrade pip
RUN pip install --upgrade cython

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000
CMD ["gunicorn"  , "-b", "0.0.0.0:8000", "server:app"]


#CMD [ "gunicorn", "server:server" ]

#["gunicorn", "-w", "20", "-b", "127.0.0.1:8083", "main_file:server"]
