# Prometheus monitoring and Grafana graphing

- `hawk_home` exposes a metrics endpoint on http://0.0.0.0:6178/metrics
- `hawk_scout` exposes its metrics endpoint on http://0.0.0.0:6103/metrics
  (actually it uses the configured --a2s-port + 3)

The metrics endpoints are publically accessible and can be monitored remotely,
which is needed to access both the remote scouts and the home endpoint from the
Prometheus docker container because the container is using a private network
and as such doesn't see localhost services.

If you run `docker-compose up -d` from the current directory it will
orchestrate two containers. One container runs Prometheus which will scrape
the configured hawk_home and hawk_scout endpoints every 5 seconds.

The other container runs a Grafana instance listening on port 3000, you can log
in with admin/admin and it will prompt you to change the password.

State will be persisted in docker volumes, to clear all state completely run
`docker-compose down -v` to stop the containers and remove the persisted state
in the docker volumes.

Some of the metrics that can be queried/graphed

## Hawk Scout/Inferencing related metrics

```
hawk_retriever_total_images: Gauge
    Total number of images in mission

hawk_retriever_total_objects: Gauge
    Total number of samples in mission

hawk_retriever_retrieved_images: Counter
    Number of images retrieved

hawk_retriever_retrieved_objects: Counter
    Number of samples retrieved

hawk_retriever_dropped_objects: Counter
    Number of samples dropped by retriever queue

hawk_retriever_queue_length: Gauge
    Number of tiles queued for inferencing
```

```
hawk_inferenced_objects{gt="..."}: Histogram (_count, _sum, _buckets)
    Histogram buckets for confidence scores of inferenced samples broken down
    by mission and groundtruth
```
This metric is quite powerful and can be used to derive other metrics

Number of objects inferenced:
    `sum(hawk_inferenced_objects_count)`

Number of true positives (Oracle):
    `sum(hawk_inferenced_objects_count{gt!="0"})`

Average confidence score for all positives:
    `sum(hawk_inferenced_objects_sum{gt!="0"}) / sum(hawk_inferenced_objects_count{gt!="0"})`

Percentage of samples that had a score less than 0.6
    `hawk_inferenced_objects_bucket{le="0.6"} / ignoring(le) hawk_inferenced_objects_count`

Number of samples had a score greater than 0.6
    `hawk_inferenced_objects_count - ignoring(le) hawk_inferenced_objects_bucket{le="0.6"}`

To display a heatmap of confidence scores for positive samples in Grafana:
    `sum(rate(hawk_inferenced_objects_buckets{gt="1"}[$__rate_interval])) by (le)`

```
hawk_selector_skipped_objects: Counter
    Number of samples that skipped selector queue because there was no model

hawk_selector_priority_queue_length: Gauge
    Number of samples queued in the selector priority queue(s)

hawk_selector_revisited_objects: Counter
    Number of reexamined samples

hawk_selector_result_queue_length: Gauge
    Number of samples queued for sending to hawk_home
```

```
hawk_survivability_true_positive: Counter
    True positive based on the survivability countermeasure threshold

hawk_survivability_false_positive: Counter
    False positive based on the survivability countermeasure threshold

hawk_survivability_false_negative: Counter
    False negative based on the survivability countermeasure threshold

hawk_survivability_true_negative: Counter
    True negative based on the survivability countermeasure threshold

hawk_survivability_threats_not_countered: Counter
    Survivability threats not countered (FN + TP after depleting CMs)
```

## Hawk Home/Labeling related metrics

```
hawk_unlabeled_received{scout="..."}: Summary (_count, _sum)
    Number of samples received from each scout and cumulative message size.

hawk_unlabeled_queue_length: Gauge
    Length of the priority queue with samples waiting to be written to unlabeled.jsonl

hawk_labeler_queued_length: Gauge
    Number of samples written to unlabeled.jsonl but not yet labeled by the labeler

hawk_labeler_queued_time: Histogram
    Time elapsed from being written to unlabeled.jsonl until a label is sent back to the scout (seconds)

hawk_labeled_objects{label="..."}: Counter
    Number of samples that were labeled as the class specified by 'label'

hawk_labeled_queue_length{scout="..."}: Gauge
    Number of labels queued to be sent back to each scout
```
