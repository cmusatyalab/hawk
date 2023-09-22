import 'package:flutter/material.dart';
import 'package:hawk_ui/widgets/color_filter.dart';
import 'package:hawk_ui/widgets/fsl_filter.dart';
import 'package:hawk_ui/models/filters.dart';
import 'package:hawk_ui/widgets/hawk_mode.dart';

const List<String> list = <String>['Texture', 'Few-Shot', 'Hawk'];

class DropDownWidget extends StatefulWidget {

  FilterConfig config;
  //HawkConfig hconfig;

DropDownWidget({
    Key? key,
    required this.config,
  }) : super(key: key);


  @override
  State<DropDownWidget> createState() => _DropDownWidgetState();
}

class _DropDownWidgetState extends State<DropDownWidget> {
  String? _chosenValue;

  List<DropdownMenuItem<String>> _createList() {
  return list
      .map<DropdownMenuItem<String>>(
        (e) => DropdownMenuItem(
          value: e,
          child: Text(e),
        ),
      )
      .toList();
}

  @override
  Widget build(BuildContext context) {
    Widget page;

    switch (_chosenValue) {
      case 'Texture':
        page = Placeholder();
        break;
      case 'Few-Shot':
        page = FSLSetup(config: widget.config);
        break;
        case 'Hawk':
        page = HawkSetup(config: widget.config);
        break;
      case null:
        page = Center(child: Text("NO FILTER CHOSEN"));
        break;
      default:
        //page = FSLSetup(config: widget.config);
        throw UnimplementedError('no widget for $_chosenValue');
    }

    return LayoutBuilder(builder: (context, constraints) {
      return Container(
          padding: const EdgeInsets.all(0.0),
          child: Column(
            children: [
              DropdownButton<String>(
                value: _chosenValue,
                //elevation: 5,
                style: TextStyle(color: Colors.black),

                items:_createList(),
                hint: Text(
                  "Please select a Mission Mode",
                  style: TextStyle(
                      color: Colors.black,
                      fontSize: 24,
                      fontWeight: FontWeight.w600),
                ),
                onChanged: (String? value) {
                  setState(() {
                    _chosenValue = value ?? "";
                  });
                },
              ),
              Expanded(
              child: Container(
                color: Theme.of(context).colorScheme.primaryContainer,
                child: page,
              ),
            ),
            ],
          ),
        );
    });
  }
}
