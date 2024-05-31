# Prometheus monitoring

hawk_home has a metrics endpoint on http://localhost:6178/metrics, it will not
be publically accessible.

If you download a prometheus binary and run it in this directory it will start
scraping the hawk_home endpoint every 5 seconds.  Prometheus will be accessible
as http://localhost:9090 (prometheus by default listens on all interfaces and
is therefore publically accessible). You can also point a Grafana instance at
this prometheus source for enhanced graphing.

Some of the metrics that can be queried/graphed

hawk_unlabeled_received_sum: On the wire message size of all samples received from each scout

hawk_unlabeled_received_count: Number of samples received from each scout

hawk_unlabeled_queue_length: Length of the priority queue with samples waiting to be written to unlabeled.jsonl

hawk_labeler_queued_length: Number of samples written to unlabeled.jsonl but not yet labeled by the labeler

hawk_labeler_queued_time: Time elapsed from being written to unlabeled.jsonl until a label is sent back to the scout (seconds)

hawk_labeled_positive: Number of samples that were labeled as True Positive

hawk_labeled_negative: Number of samples that were labeled as False Positive

hawk_labeled_queue_length: Number labels queued to be sent back to each scout
