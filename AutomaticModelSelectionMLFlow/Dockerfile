FROM python:3.6
WORKDIR /workdir
COPY ./requirements.txt ./
RUN pip install -r requirements.txt
RUN mkdir -p app
COPY ./app app
RUN pip install "git+https://github.com/maciejkula/spotlight.git#egg=spotlight"
CMD ["python","app/SystematicModelOptimization.py"]