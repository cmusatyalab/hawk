import 'dart:convert';
import 'dart:typed_data';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:hawk_ui/widgets/filter_select.dart';
import 'package:hawk_ui/widgets/result_display.dart';


void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return ChangeNotifierProvider(
      create: (context) => MyAppState(),
      child: MaterialApp(
          debugShowCheckedModeBanner: false,
          title: 'Hawk App v2',
          theme: ThemeData(
            useMaterial3: true,
            colorScheme: ColorScheme.fromSeed(seedColor: Colors.teal[600]!),
          ),
          home: MyHomePage(),
          ),
    );
  }
}

class MyAppState extends ChangeNotifier {
  Uint8List? imageFile;

  uploadImage() async {
  FilePickerResult? result = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['png', 'jpg', 'svg', 'jpeg', 'ppm', 'bmp']);

  if (result != null) {
    PlatformFile file = result.files.first;
    imageFile = file.bytes!;
    notifyListeners();
  } else {
    // User canceled the picker
  }
}
}

class MyHomePage extends StatefulWidget {
  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  var selectedIndex = 0;

  @override
  Widget build(BuildContext context) {
    var appState = context.watch<MyAppState>();
    Widget page;
    var theme = Theme.of(context);

    switch (selectedIndex) {
      case 0:
        page = FilterSetup();
        break;
      case 1:
        page = ResultDisplay();
        break;
      default:
        throw UnimplementedError('no widget for $selectedIndex');
    }

    return LayoutBuilder(builder: (context, constraints) {
      return Scaffold(
        body: Row(
          children: [
            SafeArea(
              child: NavigationRail(
                backgroundColor: theme.colorScheme.primary,
                extended: constraints.maxWidth >= 250,
                destinations: [
                  NavigationRailDestination(
                    icon: Icon(
                      Icons.filter_alt,
                      color: Colors.white,
                    ),
                    label: Text(
                      'Mission Config',
                      style: TextStyle(
                        color: Colors.white,
                      ),
                    ),
                  ),
                  NavigationRailDestination(
                    icon: Icon(
                      Icons.add_photo_alternate,
                      color: Colors.white,
                    ),
                    label: Text(
                      'Results',
                      style: TextStyle(
                        color: Colors.white,
                      ),
                    ),
                  ),
                  NavigationRailDestination(
                    icon: Icon(
                      Icons.favorite,
                      color: Colors.white,
                    ),
                    label: Text(
                      'Other Options',
                      style: TextStyle(
                        color: Colors.white,
                      ),
                    ),
                  ),
                ],
                selectedIndex: selectedIndex,
                onDestinationSelected: (value) {
                  setState(() {
                    selectedIndex = value;
                  });
                },
              ),
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