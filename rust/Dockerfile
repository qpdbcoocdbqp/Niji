FROM rust:1.91.1-slim AS builder
RUN apt-get update && apt-get install -y libssl-dev pkg-config build-essential

WORKDIR /app
COPY ./app .
RUN cargo build --release

FROM debian:stable-slim AS application
WORKDIR /app
COPY --from=builder /app/target/release/rust_benchmark .
