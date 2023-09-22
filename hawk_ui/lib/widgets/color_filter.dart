import 'package:flutter/material.dart';
import 'package:flutter_colorpicker/flutter_colorpicker.dart';

class ColorSetup extends StatefulWidget {
  const ColorSetup({super.key});

  @override
  State<ColorSetup> createState() => _ColorSetupState();
}

class _ColorSetupState extends State<ColorSetup> {
  Color currentColor = Color(0xff443a49);
  final textController =
      TextEditingController(text: '#2F19DB');
  void changeColor(Color color) => setState(() => currentColor = color);

  void _showcontent() {
    showDialog(
        context: context,
        builder: (BuildContext context) {
        return AlertDialog(
    title: const Text('Pick a color!'),
    content: SingleChildScrollView(
      child: ColorPicker(
        pickerColor: currentColor,
        onColorChanged: changeColor,
        colorPickerWidth: 300,
        pickerAreaHeightPercent: 0.7,
        enableAlpha: false,
        displayThumbColor: true,
        paletteType: PaletteType.hsvWithHue,
        labelTypes: const [],
        pickerAreaBorderRadius: const BorderRadius.only(
          topLeft: Radius.circular(2),
          topRight: Radius.circular(2),
        ),
        hexInputController: textController, // <- here
        portraitOnly: true,


      ),
      ),
    actions: <Widget>[
      ElevatedButton(
        child: const Text('Done'),
        onPressed: () {
          print(currentColor);
          Navigator.of(context).pop();
        },
      ),
    ],
  );
      },
    );
  }


  @override
  Widget build(BuildContext context) {

    return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text(
              'Choose Color:',
            ),
            ElevatedButton(
              onPressed: _showcontent,
              style: ButtonStyle(
                backgroundColor: MaterialStateProperty.all(Colors.blueAccent),
              ),
              child: Text('Click here', style: TextStyle(color: Colors.white)),
            ),
          ],
        ),
      );
  }
}
