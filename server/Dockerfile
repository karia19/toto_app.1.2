FROM python:3.9-buster

RUN apt-get update -y \
  && apt-get install cron -y  && apt-get install nano 

#COPY update_ods.cron /etc/cron.d/update_ods.cron
#RUN chmod 0644 /etc/cron.d/update_ods.cron
#RUN /usr/bin/crontab /etc/cron.d/update_ods.cron
#RUN chmod 0644 /etc/cron.d/update_ods.cron \
#  && crontab /etc/cron.d/update_ods.cron
COPY crontab /etc/cron.d/crontab

RUN chmod 0644 /etc/cron.d/crontab
#RUN chmod u+x test_cron.py
#RUN chmod u+x update_redis.py

RUN /usr/bin/crontab /etc/cron.d/crontab

#CMD ["cron", "-f"]
CMD ["service", "cron", "start"]

WORKDIR /app

#RUN chmod u+x test_cron.py

RUN pip install --upgrade pip
RUN pip install --upgrade cython

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

RUN chmod u+x /app/test_cron.py
RUN chmod u+x /app/update_redis.py


CMD ["gunicorn"  , "-b", "0.0.0.0:8000", "server:app"]


#CMD [ "gunicorn", "server:server" ]

#["gunicorn", "-w", "20", "-b", "127.0.0.1:8083", "main_file:server"]
