FROM python:3.9-buster

RUN apt-get update -y \
  && apt-get install cron -y

COPY update_ods.cron /etc/cron.d/update_ods.cron

RUN chmod 0644 /etc/cron.d/update_ods.cron \
  && crontab /etc/cron.d/update_ods.cron

CMD ["cron", "-f"]


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
