import json
import logging
import os
import sys

from analysis import analyze, get_model_bundle
from messaging import ensure_batch_queue, open_channel, publish_metric, BATCH_QUEUE
from models_setup import ensure_models
from storage import download_to_temp

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("spectra.batch_worker")


def process_item(channel, batch_id: str, item: dict) -> None:
    song_id = item["songId"]
    file_key = item["fileKey"]
    local_path = None
    try:
        local_path = download_to_temp(file_key)
        result = analyze(local_path)
        publish_metric(
            channel,
            event_type="INFORMATION",
            process_type="FLOW",
            code="track_analyzed",
            payload={"batchId": batch_id, "songId": song_id, "fileKey": file_key, "metadata": result},
        )
        logger.info("Analyzed song=%s fileKey=%s", song_id, file_key)
    except Exception as e:
        logger.exception("Failed to analyze song=%s fileKey=%s", song_id, file_key)
        publish_metric(
            channel,
            event_type="ERROR",
            process_type="FLOW",
            code="analysis_failed",
            payload={"batchId": batch_id, "songId": song_id, "fileKey": file_key, "error": str(e)},
        )
    finally:
        if local_path and os.path.exists(local_path):
            os.remove(local_path)


def on_message(channel, method, properties, body):
    try:
        message = json.loads(body)
    except json.JSONDecodeError:
        logger.error("Discarding malformed batch message: %r", body[:200])
        channel.basic_ack(method.delivery_tag)
        return

    payload = message.get("payload", message)
    batch_id = payload.get("batchId", "unknown")
    items = payload.get("items", [])
    logger.info("Processing batch=%s with %d item(s)", batch_id, len(items))

    for item in items:
        process_item(channel, batch_id, item)

    channel.basic_ack(method.delivery_tag)


def main():
    ensure_models(quiet=True)
    get_model_bundle()  # load once, kept resident for the life of this process

    connection, channel = open_channel()
    ensure_batch_queue(channel)
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue=BATCH_QUEUE, on_message_callback=on_message)

    logger.info("Listening on queue '%s' for batch analysis requests...", BATCH_QUEUE)
    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        channel.stop_consuming()
    finally:
        connection.close()


if __name__ == "__main__":
    main()
