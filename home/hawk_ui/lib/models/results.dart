import 'dart:convert';
import 'dart:typed_data';

class ResultItem {
  ResultItem({
    required this.name,
    required this.image,
  });

  String name;
  String image;

  factory ResultItem.fromJson(Map<String, dynamic> json) => ResultItem(
        name: json['name'],
        image: json['image'],
      );

  Uint8List getContent() {
    return Uint8List.fromList(base64Decode(image));
  }
}