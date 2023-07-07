import 'dart:convert';
import 'dart:typed_data';
import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:hawk_ui/styles/buttons.dart';
import 'package:hawk_ui/models/filters.dart';

class HawkSetup extends StatefulWidget {

  FilterConfig config; 
  
  HawkSetup({
    Key? key,
    required this.config,
  }) : super(key: key);

  @override
  State<HawkSetup> createState() => _HawkSetupState();
}

class _HawkSetupState extends State<HawkSetup> {
  Uint8List? imageFile;
  

  uploadImage() async {
    FilePickerResult? result = await FilePicker.platform.pickFiles(
        type: FileType.custom,
        allowedExtensions: ['png', 'jpg', 'svg', 'jpeg', 'ppm', 'bmp']);

    if (result != null) {
      PlatformFile file = result.files.first;
      Uint8List newFile = file.bytes!;
      setState(() => imageFile = newFile);
    } else {
      // User canceled the picker
    }

    String image_string = base64.encode(imageFile!);
    
  }
  int image_dim = 100;
  
  @override
  Widget build(BuildContext context) {
  widget.config.name = 'hawk';
  widget.config.args = {'hawk_args': ''};
    return LayoutBuilder(builder: (context, constraints) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
          const SizedBox(width: 30, height:20),
          const Text("This is the Hawk mission configuration wizard.  Please select an option to upload the necessary file for each field."),
          const SizedBox(width: 30, height:30),
          Row(mainAxisAlignment: MainAxisAlignment.spaceEvenly,crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            Text("DOTA Dataset - Class Roundabout: see sample positive "
                "images -->", softWrap: true,style: TextStyle(fontSize: 20,
                  fontWeight: FontWeight.w800), overflow:TextOverflow
                .ellipsis,maxLines:10,),
            SizedBox(width: 20,),
            Image.asset('images/dota_roundabout/P2529_00960_00000.png', fit:BoxFit.cover,
                height:100,
          width:100),Image.asset
              ('images/dota_roundabout/P2545_00384_01536.png', fit:BoxFit
                .cover, height:100,
          width:100),Image.asset
              ('images/dota_roundabout/P2467_02232_00768.png',fit:BoxFit
                .cover, height:100,
          width:100),Image.asset
              ('images/dota_roundabout/P2464_00768_00768.png',fit:BoxFit
                .cover, height:100,
          width:100), Image.asset
              ('images/dota_roundabout/P2430_00576_01152.png', fit:BoxFit
                .cover, height:100,
          width:100),
          ],),
            const SizedBox(width: 30, height:20),

          Row(mainAxisAlignment: MainAxisAlignment.spaceEvenly,crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            Text("School Bus Dataset: see sample positive "
                "images -->", softWrap: true,style: TextStyle(fontSize: 20,
                  fontWeight: FontWeight.w800), overflow:TextOverflow
                .ellipsis,maxLines:10,),
            SizedBox(width: 20,),
            Image.asset('images/school_bus/6.jpeg', fit:BoxFit.cover,
                height:100,
          width:100),Image.asset
              ('images/school_bus/4.jpg', fit:BoxFit.cover, height:100,
          width:100),Image.asset
              ('images/school_bus/16.jpeg',fit:BoxFit.cover, height:100,
          width:100),Image.asset
              ('images/school_bus/24.jpeg',fit:BoxFit.cover, height:100,
          width:100), Image.asset
              ('images/school_bus/19.jpeg', fit:BoxFit.cover, height:100,
          width:100),


                /*
                ElevatedButton(
                  style: ButtonStyles.buttonWithIcon,
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: <Widget>[
                      const Icon(Icons.file_upload),
                      const SizedBox(width: 10),
                      const Text('Bootstrap Dataset') // change to upload dataset and write new function
                    ],
                  ),
                  onPressed: () => uploadImage(),
                ),*/
          ]),
          const SizedBox(width: 30, height:20),
          Row(mainAxisAlignment: MainAxisAlignment.center,children: [
            const Text("Select a DNN Model or a Bootstrap data set -- >"),
            /*const Text("Select a DNN Model or a Bootstrap data set /*from "
          "which"
                " to generate an initial model on all scouts.  Uploading a "
                "Pre-Trained DNN model assumes that you have previously "
                "trained such a model on a small dataset consisting of a "
                "generally balanced set of true positives and negatives of a "
                "target class that you want to detect from the ingress data*/ "
                "source.",softWrap: true,overflow:TextOverflow.ellipsis,
              maxLines:3,),*/
                      const SizedBox(width: 100),

ElevatedButton(
                  style: ButtonStyles.buttonWithIcon,
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: <Widget>[
                      const Icon(Icons.file_upload),
                      const SizedBox(width: 10),
                      const Text('DNN Model')
                    ],
                  ),
                  onPressed: () => uploadImage(), // change to upload model and write new function
                ),],
          ),
          //const SizedBox(width:400, height:200, child:

                const SizedBox(width: 30, height:30),
                /*
                Expanded(
                child: imageFile == null
                  ? const Center(child: Text('No Image Uploaded'))
                  : Image.memory(imageFile!),
                ),*/
                const SizedBox(width: 60),
                

                                const SizedBox(width: 60),
          ],
        ),
      );
    });
  }
}
