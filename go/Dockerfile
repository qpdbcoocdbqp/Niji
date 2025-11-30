FROM golang:1.25 AS builder
WORKDIR /app
COPY ./app /app
RUN go mod tidy
RUN go build -o benchmark .

FROM debian:bookworm-slim AS application
WORKDIR /app
# Install dependencies including curl and tar for downloading ONNX Runtime
RUN apt-get update && apt-get install -y ca-certificates libgomp1 curl tar && rm -rf /var/lib/apt/lists/*

# Download and install ONNX Runtime
ENV ONNXRUNTIME_VERSION=1.17.1
RUN curl -L https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz -o onnxruntime.tgz && \
    tar -xzf onnxruntime.tgz && \
    mv onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}/lib/libonnxruntime.so* /usr/lib/ && \
    rm -rf onnxruntime.tgz onnxruntime-linux-x64-${ONNXRUNTIME_VERSION} && \
    ln -s /usr/lib/libonnxruntime.so /usr/lib/onnxruntime.so

COPY --from=builder /app/benchmark .
ENTRYPOINT []