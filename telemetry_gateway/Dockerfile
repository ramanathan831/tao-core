
################# BASE IMAGE #################

FROM python:3.12-slim
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential

# install telemetry gateway dependencies
COPY telemetry_gateway/requirements.txt requirements.txt
RUN python -m pip install --disable-pip-version-check -r requirements.txt

# install tao-core
COPY . tao-core
RUN cd tao-core && ls -la && bash release/python/build_wheel.sh && \
    find dist/ -name "nvidia_tao_core*.whl" -type f | xargs -n 1 pip install && \
    cd .. && rm -rf tao-core

# Install API
WORKDIR /opt/api
COPY telemetry_gateway .
RUN ls -la
EXPOSE 9877

# Default command
CMD ["/bin/bash", "start.sh"]
