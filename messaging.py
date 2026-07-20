import json
import os
import uuid
from datetime import datetime, timezone

import pika
from dotenv import load_dotenv

load_dotenv()

RABBITMQ_HOST = os.environ.get("RABBITMQ_HOST", "localhost")
RABBITMQ_PORT = int(os.environ.get("RABBITMQ_PORT", "5672"))
RABBITMQ_USERNAME = os.environ.get("RABBITMQ_USERNAME", "guest")
RABBITMQ_PASSWORD = os.environ.get("RABBITMQ_PASSWORD", "guest")

SERVICE_ID = "spectra"

METRICS_EXCHANGE = "metrics"
COMMANDS_EXCHANGE = "commands"
COMMANDS_ROUTING_KEY_IN = "spectra"  # messages addressed to us
BATCH_QUEUE = "spectra-commands"


def _connection_params() -> pika.ConnectionParameters:
    return pika.ConnectionParameters(
        host=RABBITMQ_HOST,
        port=RABBITMQ_PORT,
        credentials=pika.PlainCredentials(RABBITMQ_USERNAME, RABBITMQ_PASSWORD),
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def publish_metric(channel, event_type: str, process_type: str, code: str, payload: dict, trace_id: str | None = None) -> None:
    """Publish a MetricEventDTO-shaped message to the shared `metrics` fanout exchange
    (already consumed by metriq — see MetricConsumer.java on the Java side)."""
    body = {
        "serviceId": SERVICE_ID,
        "brandName": None,
        "type": event_type,
        "processType": process_type,
        "timestamp": _now_iso(),
        "traceId": trace_id or str(uuid.uuid4()),
        "parentTraces": [],
        "code": code,
        "payload": payload,
    }
    channel.basic_publish(
        exchange=METRICS_EXCHANGE,
        routing_key="metric.event",
        body=json.dumps(body).encode(),
    )


def open_channel():
    connection = pika.BlockingConnection(_connection_params())
    channel = connection.channel()
    return connection, channel


def ensure_batch_queue(channel) -> None:
    """Declare (idempotent) our own inbound queue and bind it to the shared
    `commands` topic exchange under our routing key. Does not attempt to
    declare the `commands` exchange itself — that's owned by the Java side."""
    channel.queue_declare(queue=BATCH_QUEUE, durable=True)
    channel.queue_bind(
        queue=BATCH_QUEUE,
        exchange=COMMANDS_EXCHANGE,
        routing_key=COMMANDS_ROUTING_KEY_IN,
    )
