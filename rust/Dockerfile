FROM rust:1.91.1-slim AS builder
RUN apt-get update && apt-get install -y libssl-dev pkg-config build-essential

WORKDIR /app
COPY ./app .
RUN cargo build --release

FROM debian:stable-slim AS application
RUN apt-get update && apt-get install -y ca-certificates libssl-dev && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder /app/target/release/rust_benchmark .
