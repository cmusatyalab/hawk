# SPDX-FileCopyrightText: 2022 Carnegie Mellon University <satya-group@lists.andrew.cmu.edu>
#
# SPDX-License-Identifier: GPL-2.0-only

from collections import defaultdict
from logzero import logger
from google.protobuf.json_format import MessageToDict
import json
import os

def get_stats(condition, *args):
    if condition == "time":
        return TimerStats(*args)
    elif condition == "model":
        return VersionStats(*args)
    else:
        raise NotImplementedError()


class HawkStats(object):

    def __init__(self, stubs, mission_id, log_dir):
        self.stubs = stubs
        self.mission_id = mission_id
        self.log_files = [open(os.path.join(log_dir, 'get-stats-{}.txt'.format(i)), "a") for i, stub in enumerate(self.stubs)]
        
    def accumulate_mission_stats(self):
        stats = defaultdict(lambda: 0)
        str_ignore = ['server_time', 'ctime', 
                      'train_positives', 'server_positives', 
                      'msg']
        single = ['server_time', 'train_positives', 'version']
        for i, stub in enumerate(self.stubs):
            try:
                mission_stat = stub.GetMissionStats(self.mission_id)
                mission_stat = MessageToDict(mission_stat)
                self.log_files[i].write("{}\n".format(json.dumps(mission_stat)))
                for k, v in mission_stat.items():
                    if isinstance(v, dict):
                        others = v
                        for key, value in others.items():
                            if key in mission_stat:
                                continue
                            if key in str_ignore:
                                if i == 0:
                                    stats[key] = value
                            elif key in single:
                                if i == 0:
                                    stats[key] = float(value)
                            else:
                                stats[key] += float(value)
                    else:
                        stats[k] += float(v)
            except Exception as e:
                logger.error(e)
                pass

        return stats

    def get_latest_model_version(self):
        model_version = self.stubs[0].a2s_get_test_results(self.mission_id).version

        return model_version

    def stop(self):
        [f.close() for f in self.log_files]


class TimerStats(HawkStats):
    """
    Returns Mission statistics at regular intervals
    """
    def __init__(self, stubs, mission_id, stop_event, stats_queue, interval, log_dir):
        super().__init__(stubs, mission_id, log_dir)
        self.stop_event = stop_event
        self.stats_queue = stats_queue
        self.interval = interval

    def start(self):

        while not self.stop_event.wait(self.interval):
            stats = self.accumulate_mission_stats()
            if stats:
                self.stats_queue.put(stats)

        self.stats_queue.put(None)


class VersionStats(HawkStats):
    """
    Returns Mission statistics at regular intervals
    """
    def __init__(self, stubs, mission_id, stop_event, stats_queue, interval, log_dir):
        super().__init__(stubs, mission_id, log_dir)
        self.stop_event = stop_event
        self.stats_queue = stats_queue
        self.interval = interval
        self.version = 0

    def start(self):

        while not self.stop_event.wait(timeout=self.interval):
            current_version = self.get_latest_model_version()
            if self.version != current_version:
                logger.info("Version not equal {} -> {}".format(self.version, current_version))
                self.version = current_version
                stats = self.accumulate_mission_stats()
                if stats:
                    stats['version'] = self.version
                    self.stats_queue.put(stats)

        self.stats_queue.put(None)
