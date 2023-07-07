import 'dart:convert';
import 'dart:developer';
import 'dart:typed_data';
import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:hawk_ui/styles/buttons.dart';
import 'package:hawk_ui/models/filters.dart';

class FSLSetup extends StatefulWidget {

  FilterConfig config; 
  FSLSetup({
    Key? key,
    required this.config,
  }) : super(key: key);  

  @override
  State<FSLSetup> createState() => _FSLSetupState();
}

class _FSLSetupState extends State<FSLSetup> {
  Uint8List? imageFile;
  String? filePath;
  
  uploadImage() async {
    FilePickerResult? result = await FilePicker.platform.pickFiles(
        type: FileType.custom,
        allowedExtensions: ['png', 'jpg', 'svg', 'jpeg', 'ppm', 'bmp']);

    if (result != null) {
      PlatformFile file = result.files.first!;
      Uint8List newFile = file.bytes!;

      setState(() {
        imageFile = newFile;
      });
      
      
      } else {
      // User canceled the picker
    }
  

    String image_string = base64.encode(imageFile!);
    widget.config.name = 'fsl';
    widget.config.args = {'support': image_string};
}

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(builder: (context, constraints) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            ElevatedButton(
                  style: ButtonStyles.buttonWithIcon,
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: <Widget>[
                      const Icon(Icons.file_upload),
                      const SizedBox(width: 10),
                      const Text('Upload Image',style:TextStyle(fontSize:20))
                    ],
                  ),
                  onPressed: () => uploadImage(),
                ),
                const SizedBox(width: 30),
                Expanded(
                child: imageFile == null
                  ? Center(child: Text('No Image Uploaded'))
                  : Center(child: Row(mainAxisAlignment: MainAxisAlignment.center,children: [SizedBox(width:500,height:200, child:Text("You have chosen the support image to the right.  Hawk will use Few Shot Learning mode to use a feature extractor and distance to find the most similar images from the selected data source.", overflow:TextOverflow.ellipsis,maxLines:10),),Image.memory(imageFile!),],),),
                ),
                const SizedBox(width: 30),
          ],
        ),
      ); 
    });
  }
}
