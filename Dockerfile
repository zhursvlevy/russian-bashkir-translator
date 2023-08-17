FROM ubuntu

RUN apt-get update

RUN apt-get install -y python3
RUN apt-get -y install python3-pip
RUN pip install --no-cache --upgrade pip setuptools

WORKDIR /app

COPY . .

RUN pip3 install -r requirements.txt


