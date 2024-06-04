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

The other container runs a Grafana instance on port 3000, you can log in with
admin/admin and it will prompt you to change the password.

State will be persisted in docker volumes, to clear all state completely run
`docker-compose down -v` to stop the containers and remove the persisted state
in the docker volumes.

Some of the metrics that can be queried/graphed

## Hawk Home/Labeling related metrics

```
hawk_unlabeled_received_sum:
    On the wire message size of all samples received from each scout

hawk_unlabeled_received_count:
    Number of samples received from each scout

hawk_unlabeled_queue_length:
    Length of the priority queue with samples waiting to be written to unlabeled.jsonl

hawk_labeler_queued_length:
    Number of samples written to unlabeled.jsonl but not yet labeled by the labeler

hawk_labeler_queued_time:
    Time elapsed from being written to unlabeled.jsonl until a label is sent back to the scout (seconds)

hawk_labeled_positive:
    Number of samples that were labeled as True Positive

hawk_labeled_negative:
    Number of samples that were labeled as False Positive

hawk_labeled_queue_length:
    Number of labels queued to be sent back to each scout
```
