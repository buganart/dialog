FROM python:3.6
COPY ./dialog .
COPY ./requirements.txt ./requirements.txt
RUN pip install --upgrade pip && \
    pip install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install -r requirements.txt
EXPOSE 8080
CMD ["python", "./api.py"]
