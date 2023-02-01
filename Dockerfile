FROM python:3.7.6
WORKDIR /app
ADD requirement_build.sh /app
ADD requirements_arm32.txt /app
ADD requirements_arm64.txt /app
ADD requirements_amd.txt /app
RUN chmod +x /app/requirement_build.sh
RUN /app/requirement_build.sh
COPY . .
CMD ["python", "-u", "client.py"]



