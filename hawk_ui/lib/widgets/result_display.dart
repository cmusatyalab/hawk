import 'package:flutter/material.dart';
import 'package:hawk_ui/providers/websocket.dart';
import 'package:hawk_ui/models/results.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'dart:async';
import 'dart:io';
import 'dart:core';
import 'package:web_socket_channel/web_socket_channel.dart';
import 'package:web_socket_channel/io.dart';
import 'dart:developer' as developer;
import 'package:flutter/foundation.dart';

//print("Welcome to GeeksforGeeks!");
class ResultDisplay extends StatefulWidget {
  const ResultDisplay({
    Key? key,
  }) : super(key: key);

  @override
  State<ResultDisplay> createState() => _ResultDisplayState();
}

class _ResultDisplayState extends State<ResultDisplay> {
  //sending a post request to the url
  //final WebSocket channel = WebSocket('ws://cloudlet038.elijah.cs.cmu.edu:5000');
  // Change back to 127.0.0.1 as needed above
  late WebSocketChannel _channel;
  int total_samples_received = 0;

  Map<String, String> labels = {"1": "T", "0": "F"};
  String? label;
  String? img_num;
  var _isConnected = false;
  List<ResultItem> results = [];
  List<String> selected_samples = [];
  Map<String, Map<String, bool>> sample_state = {};
  //Map<String, String> staged_for_label = {};
  int total_positive_labels = 0;
  int total_negative_labels = 0;
  int current_model_version = 0;
  int total_samples_inferenced = 0;
  double positive_label_ratio = 0;
  double received_ratio = 0;
  List<bool> image_selected = [];
  dynamic lastDataItem;
  Color neg_color = Color.fromARGB(255, 168, 15, 4);
  Color pos_color = Color.fromARGB(255, 8, 112, 11);
  int old_length = 0;
  ResultItem? unique_result;
  //print('After vars');
  int elapsed_time = 0;
  String time_string = '00:00';
  int counter = 0;
  StreamController<ValueNotifier<int>> timerStreamController =
      StreamController<ValueNotifier<int>>();
  ValueNotifier<int> counterValueNotifier = ValueNotifier<int>(0);

  late StreamSubscription _subscription;
  @override
  void initState() {
    super.initState();
    _connectToWebSocket();

    timerStreamController.stream.listen((valueNotifier) {
      counterValueNotifier.value = valueNotifier.value;
    });
    _startTimer();
  }

  void _startTimer() {
    Timer.periodic(Duration(seconds: 1), (timer) {
      counterValueNotifier.value = timer.tick;
      timerStreamController.add(counterValueNotifier);
    });
  }

  void handleMsg(dynamic message) {
    total_samples_received++;
  }

  void _connectToWebSocket() async {
    _channel = await WebSocketChannel.connect(
        Uri.parse('ws://cloudlet038.elijah.cs.cmu.edu:5000'));
    setState(() {
      _isConnected = true;
    });
    listener();
  }

  void listener() {
    _channel.stream.listen((var newdata) {
      var data = json.decode(newdata.toString());
      //var newData = snapshot.data;
      ResultItem result = ResultItem.fromJson(data);
      results.add(result);
      total_samples_received += 1;
      setState(() {});
    });
  }

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
  }

  String time_elapsed(int elapsed_time) {
    var duration = Duration(seconds: elapsed_time);
    var minutes = duration.inMinutes;
    var seconds = elapsed_time % 60;
    var minutesString = '$minutes'.padLeft(1, '0');
    var secondsString = '$seconds'.padLeft(2, '0');
    return '$minutesString:$secondsString';
  }

  void pos_select_sample(ResultItem res) {
    if (res.pos_sel == true) {
      setState(() {
        res.pos_sel = false;
        res.unselected = true;
      });
    } else if (res.pos_labeled == false && res.neg_labeled == false) {
      setState(() {
        res.pos_sel = true;
        res.neg_sel = false;
        res.unselected = false;
      });
    }
  }

  void neg_select_sample(ResultItem res) {
    if (res.neg_sel == true) {
      setState(() {
        res.neg_sel = false;
        res.unselected = true;
      });
    } else if (res.pos_labeled == false && res.neg_labeled == false) {
      setState(() {
        res.neg_sel = true;
        res.pos_sel = false;
        res.unselected = false;
      });
    }
  }

  void clear_all_selected_samples(List<ResultItem> result_objects) async {
    setState(
      () {
        result_objects.forEach(
          (res) {
            res.pos_sel = false;
            res.neg_sel = false;
            res.unselected = true;
          },
        );
      },
    );
  }

  IconData getPosIconType(ResultItem res) {
    bool pos_flag = res.pos_sel;
    bool pos_labeled_flag = res.pos_labeled;
    if (pos_flag == true) {
      return Icons.check_box_outlined;
    } else if (pos_labeled_flag == true) {
      return Icons.check_box;
    } else {
      return Icons.check_box_outline_blank;
    }
  }

  IconData getNegIconType(ResultItem res) {
    bool neg_flag = res.neg_sel;
    bool neg_labeled_flag = res.neg_labeled;
    if (neg_flag == true) {
      return Icons.do_disturb_on_outlined;
    } else if (neg_labeled_flag == true) {
      return Icons.do_disturb_on;
    } else {
      return Icons.circle_outlined;
    }
  }

  Map<String, String> label_list(List<ResultItem> result_objects) {
    Map<String, String> staged_for_label = {};
    setState(() {
      result_objects.forEach((res) {
        if (res.pos_sel == true) {
          staged_for_label[res.name] = 'positive';
          res.pos_sel = false;
          res.pos_labeled = true;
          total_positive_labels += 1;
        } else if (res.neg_sel == true) {
          staged_for_label[res.name] = 'negative';
          res.neg_sel = false;
          res.neg_labeled = true;
          total_negative_labels += 1;
        }
      });
      positive_label_ratio = total_positive_labels /
          (total_positive_labels + total_negative_labels);
    });
    return staged_for_label;
  }

  void count_unique_images() {
    setState(() {
      total_samples_received++;
    });
  }

  @override
  Widget build(BuildContext context) {
    //channel.connect();
    return Scaffold(
      appBar: AppBar(
        title: const Text("Hawk Results"),
      ),
      body: Padding(
        padding: const EdgeInsets.all(20.0),
        child: Center(
          child: Row(
            children: [
              Column(
                children: [
                  _isConnected
                      ? SizedBox(
                          width: MediaQuery.of(context).size.width * 0.6,
                          height: MediaQuery.of(context).size.height - 100,
                          child: results.length > 0
                              ? Row(
                                  children: <Widget>[
                                    Expanded(
                                      child: GridView.builder(
                                        gridDelegate:
                                            SliverGridDelegateWithFixedCrossAxisCount(
                                          crossAxisSpacing: 4,
                                          mainAxisSpacing: 20,
                                          crossAxisCount: 5,
                                        ),
                                        itemCount: results.length,
                                        itemBuilder: (context, index) {
                                          ResultItem this_result =
                                              results[index];
                                          label =
                                              this_result.name.split(' ')[1];
                                          if (label == '1') {
                                            label = "T";
                                          } else {
                                            label = "F";
                                          }
                                          img_num = results[index]
                                              .name
                                              .split(' ')[0]
                                              .split('.')[0]
                                              .split('/')[6];
                                          //int img_num_int =
                                          //num.parse(img_num!).toInt();

                                          return Container(
                                            key: ValueKey(index),
                                            child: Column(
                                              children: [
                                                Expanded(
                                                  child: Image(
                                                      gaplessPlayback: true,
                                                      excludeFromSemantics:
                                                          true,
                                                      fit: BoxFit.cover,
                                                      width: double.infinity,
                                                      image: MemoryImage(
                                                          this_result
                                                              .getContent())),
                                                ),
                                                Container(
                                                  padding: EdgeInsets.all(2.0),
                                                  color: Colors.black
                                                      .withOpacity(0.1),
                                                  child: Expanded(
                                                    child: Row(
                                                      mainAxisAlignment:
                                                          MainAxisAlignment
                                                              .spaceBetween,
                                                      children: [
                                                        IconButton(
                                                          icon: Icon(
                                                              getPosIconType(
                                                                  this_result),
                                                              color: Color
                                                                  .fromARGB(
                                                                      255,
                                                                      8,
                                                                      112,
                                                                      11)),
                                                          onPressed: () {
                                                            pos_select_sample(
                                                                this_result);
                                                          },
                                                        ),
                                                        Text(
                                                          '${index + 1}/${results.length}, '
                                                          '$label', //${img_num_int}',
                                                          style: TextStyle(
                                                            color: Colors.black,
                                                            fontSize: 16,
                                                          ),
                                                        ),
                                                        IconButton(
                                                            icon: Icon(
                                                                getNegIconType(
                                                                    this_result),
                                                                color: Color
                                                                    .fromARGB(
                                                                        255,
                                                                        168,
                                                                        15,
                                                                        4)),
                                                            onPressed: () {
                                                              neg_select_sample(
                                                                  this_result);
                                                            }),
                                                      ],
                                                    ),
                                                  ),
                                                ),
                                              ],
                                            ),
                                          );
                                        },
                                      ),
                                    ),
                                  ],
                                )
                              : CircularProgressIndicator(),
                        )
                      : const Text("Initiate Connection")
                ],
              ),
              SizedBox(
                  width: 5, height: MediaQuery.of(context).size.height - 100),
              Expanded(
                child: Container(
                  decoration: BoxDecoration(
                    border: Border.all(
                      color: Colors.black,
                      width: 2,
                    ),
                  ),
                  child: Center(
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.start,
                      children: [
                        Text("HAWK MISSION STATS",
                            style: TextStyle(fontSize: 24)),
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                          children: [
                            Flexible(
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  Text("Total Positives Labeled:",
                                      style: TextStyle(fontSize: 16)),
                                  Text("Total Negatives Labeled:",
                                      style: TextStyle(fontSize: 16)),
                                  Text("Total Samples Labeled:",
                                      style: TextStyle(fontSize: 16)),
                                  Text("Total Samples Received:",
                                      style: TextStyle(fontSize: 16)),
                                  Text("Total Samples Inferenced:",
                                      style: TextStyle(fontSize: 16)),
                                  Text("Received Sample Ratio:",
                                      style: TextStyle(fontSize: 16)),
                                  Text("Positive Label Sample Ratio:",
                                      style: TextStyle(fontSize: 16)),
                                  Text("Current Model Version:",
                                      style: TextStyle(fontSize: 16)),
                                  //Text("Elapsed Mission Time:", style: TextStyle(fontSize:16)),
                                  Text("Elapsed Mission Time: ",
                                      style: TextStyle(fontSize: 16)),
                                ],
                              ),
                            ),
                            Column(
                              crossAxisAlignment: CrossAxisAlignment.center,
                              children: [
                                Text("${total_positive_labels}",
                                    style: TextStyle(fontSize: 16)),
                                Text("${total_negative_labels}",
                                    style: TextStyle(fontSize: 16)),
                                Text(
                                    "${total_positive_labels + total_negative_labels}",
                                    style: TextStyle(fontSize: 16)),
                                Text("${results.length}",
                                    style: TextStyle(fontSize: 16)),
                                Text("${total_samples_inferenced}",
                                    style: TextStyle(fontSize: 16)),
                                Text(
                                    "${(received_ratio * 100).toStringAsFixed(2)}%",
                                    style: TextStyle(fontSize: 16)),
                                Text(
                                    "${(positive_label_ratio * 100).toStringAsFixed(2)}%",
                                    style: TextStyle(fontSize: 16)),
                                Text("${current_model_version}",
                                    style: TextStyle(fontSize: 16)),
                                //Text("$time_string", style: TextStyle(fontSize:16)),
                                ValueListenableBuilder(
                                  valueListenable: counterValueNotifier,
                                  builder: (context, value, child) {
                                    return Text("${time_elapsed(value)}",
                                        style: TextStyle(fontSize: 16));
                                  },
                                )
                              ],
                            ),
                          ],
                        ),
                        SizedBox(height: 10, width: double.infinity),
                        ElevatedButton(
                          style: ElevatedButton.styleFrom(
                            padding: const EdgeInsets.symmetric(
                                horizontal: 20.0, vertical: 20.0),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(10.0),
                            ),
                            backgroundColor: Colors.blue[100],
                            textStyle: const TextStyle(fontSize: 16),
                          ),
                          onPressed: () async {
                            var stats_url =
                                'http://cloudlet038.elijah.cs.cmu.edu:8000/get_stats';
                            var stats_response =
                                await http.get(Uri.parse(stats_url));
                            debugPrint("${stats_response}");
                            String stats_body = stats_response.body;
                            debugPrint("${stats_body}");
                            Map<String, dynamic> json_data =
                                jsonDecode(stats_body);
                            debugPrint("${json_data}");
                            setState(() {
                              current_model_version =
                                  json_data['version'].toInt();
                              total_samples_inferenced =
                                  json_data['processedObjects'];
                              elapsed_time = json_data['home_time'].toInt();
                              time_string = time_elapsed(elapsed_time);
                              received_ratio = total_samples_received /
                                  total_samples_inferenced;
                              positive_label_ratio = total_positive_labels /
                                  (total_positive_labels +
                                      total_negative_labels);
                            });
                          },
                          child: const Text('Refresh Stats'),
                        ),
                        SizedBox(height: 10, width: double.infinity),
                        Text("Labeling Control",
                            style: TextStyle(fontSize: 24)),
                        SizedBox(height: 10, width: double.infinity),
                        ElevatedButton(
                          style: ElevatedButton.styleFrom(
                            padding: const EdgeInsets.symmetric(
                                horizontal: 20.0, vertical: 20.0),
                            shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(10.0)),
                            backgroundColor: Colors.tealAccent[400],
                            textStyle: const TextStyle(fontSize: 16),
                          ),
                          onPressed: () async {
                            Map<String, String> label_dict =
                                label_list(results);
                            final url =
                                'http://cloudlet038.elijah.cs.cmu.edu:8000/hawk_push_labels';
                            //sending a post request to the url (default was 127.0.0.1:8000/start)
                            final response = await http.post(Uri.parse(url),
                                body: json.encode(label_dict));
                          },
                          child: const Text('Send New Labels'),
                        ),
                        SizedBox(height: 20, width: double.infinity),
                        ElevatedButton(
                          style: ElevatedButton.styleFrom(
                            padding: const EdgeInsets.symmetric(
                                horizontal: 20.0, vertical: 20.0),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(10.0),
                            ),
                            backgroundColor: Colors.amber[400],
                            textStyle: const TextStyle(fontSize: 16),
                          ),
                          onPressed: () async {
                            clear_all_selected_samples(results);
                          },
                          child: const Text('Clear Unsent Labels'),
                        ),
                        SizedBox(height: 15, width: double.infinity),
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                          children: [
                            Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text("Select to stage for Positive Label",
                                    style: TextStyle(fontSize: 16)),
                                Text("Staged for Positive Label",
                                    style: TextStyle(fontSize: 16)),
                                Text("Positive Label Sent",
                                    style: TextStyle(fontSize: 16)),
                                Text("Select to stage for Negative Label",
                                    style: TextStyle(fontSize: 16)),
                                Text("Staged for Negative Label",
                                    style: TextStyle(fontSize: 16)),
                                Text("Negative Label Sent",
                                    style: TextStyle(fontSize: 16)),
                              ],
                            ),
                            Column(
                              children: [
                                Icon(Icons.check_box_outline_blank,
                                    color: pos_color),
                                Icon(Icons.check_box_outlined,
                                    color: pos_color),
                                Icon(Icons.check_box, color: pos_color),
                                Icon(Icons.circle_outlined, color: neg_color),
                                Icon(Icons.do_disturb_on_outlined,
                                    color: neg_color),
                                Icon(Icons.do_disturb_on, color: neg_color),
                              ],
                            ),
                          ],
                        ),
                      ],
                    ),
                  ),
                  width: 400,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
