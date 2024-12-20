# python base image in the container from Docker Hub
FROM python:3.11.5-slim

# Installer libgomp1 pour résoudre l'erreur de bibliothèque manquante
RUN apt-get update && apt-get install -y libgomp1

# copy files to the /app folder in the container
COPY ./api.py /app/api.py
COPY ./requirements_api.txt /app/requirements.txt
ADD ./models /app/models
COPY ./test_api.py /app/test_api.py

# set the working directory in the container to be /app
WORKDIR /app

# install the packages from the Pipfile in the container
RUN pip install -r requirements.txt
RUN pip list

# expose the port that uvicorn will run the app on
EXPOSE 8000

# execute the command python api.py (in the WORKDIR) to start the app
#CMD ["uvicorn", "api:app", "--reload"]
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]