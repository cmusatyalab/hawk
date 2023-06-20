
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:hawk_ui/widgets/dropdown.dart';
import 'package:hawk_ui/models/filters.dart';
import 'dart:convert';


class FilterSetup extends StatefulWidget {
  @override
  State<FilterSetup> createState() => _FilterSetupState();
}

class _FilterSetupState extends State<FilterSetup> {

  FilterConfig config = FilterConfig(name:'null', args:{'support':'null'});
  //HawkConfig hconfig = HawkConfig(name:'null', args:{'support':'null'});
  @override
  Widget build(BuildContext context) {
    final width = MediaQuery.of(context).size.width;
    final height = MediaQuery.of(context).size.height;

    return Scaffold(
      body: Container(
        alignment: Alignment.center,
        width: width,
        child: Column(
          children: <Widget>[
            const Padding(
              padding: EdgeInsets.symmetric(vertical: 18.0),//18.0
              child: Text(
                'Welcome to the Hawk Browser',
                style: TextStyle(
                  fontSize: 24,
                  fontWeight: FontWeight.w800,
                ),
              ),
            ),
            const SizedBox(height: 10),

            Expanded(
              child: Container(
                child: DropDownWidget(config: config),
              ),
            ),
            SizedBox(height:MediaQuery.of(context).size.height*.1, width:MediaQuery.of(context).size.width, child: Expanded(child:Container(decoration: BoxDecoration(border:Border.all(color:Colors.black, width:2.0,),),alignment: Alignment.center, child: Center(child: const Text('Hawk is a live learning system that leverages distributed machine learning, human domain expertise, and edge computing to detect the presence of rare objects and gather valuable training samples under austere and degraded conditions.',overflow:TextOverflow.ellipsis,maxLines:10,textAlign:TextAlign.center,style: TextStyle(
                  fontSize: 20,
                  fontWeight: FontWeight.w800,
                ),),),),),),
            const SizedBox(height: 20), //vertical spacer just above start/top buttons
            Center(
              child: Row( //Row of Start and stop buttons at the bottom
                mainAxisAlignment: MainAxisAlignment.center,crossAxisAlignment: CrossAxisAlignment.center,
                children: [
                  ElevatedButton(
                    style: ElevatedButton.styleFrom(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 40.0, vertical: 20.0),
                      shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(10.0)),
                      backgroundColor: Colors.tealAccent[400],
                    ),
                    onPressed: () async {
                      //url to send the post configure request 
                      final url = 'http://cloudlet038.elijah.cs.cmu.edu:8000/start';
                      //sending a post request to the url (default was 127.0.0.1:8000/start)
                      final response = await http.post(Uri.parse(url),
                        body: json.encode({'name': config.name,
                        'args': config.args}));


                    },
                    child: const Text('Start Mission'),
                  ),
                  const SizedBox(width: 50),
                  ElevatedButton(
                    style: ElevatedButton.styleFrom(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 40.0, vertical: 20.0),
                      shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(10.0)),
                      backgroundColor: Colors.tealAccent[400],
                    ),
                    onPressed: () async {
                      final url = 'http://cloudlet038.elijah.cs.cmu.edu:8000/stop';
                      final response = await http.post(Uri.parse(url),
                        body: json.encode({'name': "Sending stop"}));
                    },
                    child: const Text('Stop Mission'),
                  ),
                ],
              ),
            ),
           const SizedBox(height: 20),
          ],
        ),
      ),
    );
  }
}
