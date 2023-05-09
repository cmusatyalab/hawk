import 'package:flutter/material.dart';
import 'package:hawk_ui/providers/websocket.dart';
import 'package:hawk_ui/models/results.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

class ResultDisplay extends StatefulWidget {

  const ResultDisplay({
    Key? key,
  }) : super(key: key);

  @override
  State<ResultDisplay> createState() => _ResultDisplayState();
}

class _ResultDisplayState extends State<ResultDisplay> {
  //sending a post request to the url
  final WebSocket channel = WebSocket('ws://127.0.0.1:5000');

  List<ResultItem> results = [];

  @override
  Widget build(BuildContext context) {
    var _isConnected = true; 

    channel.connect();

    return Scaffold(
      appBar: AppBar(
        title: const Text("Hawk Results"),
      ),
      body: Padding(
        padding: const EdgeInsets.all(20.0),
        child: Center(
          child: Column(
            children: [
              _isConnected
                  ? SizedBox(
                      width: MediaQuery.of(context).size.width,
                      height: MediaQuery.of(context).size.height - 100,
                      child: StreamBuilder(
                        stream: channel.stream,
                        builder: (context, snapshot) {
                          if (!snapshot.hasData) {
                            return const CircularProgressIndicator();
                          }

                          if (snapshot.connectionState ==
                              ConnectionState.done) {
                            return const Center(
                              child: Text("Connection Closed !"),
                            );
                          }
                          // json encoded data
                          var data = json.decode(snapshot.data.toString());
                          ResultItem result = ResultItem.fromJson(data);
                          results.add(result);
                          return Row(
                            children: <Widget>[
                              Expanded(
                                child: GridView.builder(
                                  gridDelegate:
                                      SliverGridDelegateWithFixedCrossAxisCount(
                                    crossAxisSpacing: 10,
                                    mainAxisSpacing: 10,
                                    crossAxisCount: 5,
                                  ),
                                  itemCount: results.length,
                                  itemBuilder: (context, index) {
                                    return Container(
                                        child: Image(
                                          gaplessPlayback: true,
                                          excludeFromSemantics: true,
                                          fit: BoxFit.cover,
                                          image: MemoryImage(
                                              results[index].getContent()),
                                        ),
                                      );
                                  },
                                ),
                              ),
                            ],
                          );
                        },
                      ),
                    )
                  : const Text("Initiate Connection")
            ],
          ),
        ),
      ),
    );
  }
}