import 'dart:convert';
import 'dart:typed_data';

class FilterConfig {
  FilterConfig({
    required this.name,
    required this.args,
  });

  String name;
  Map<String, String> args;

  factory FilterConfig.fromJson(Map<String, dynamic> json) => FilterConfig(
        name: json['name'],
        args: json['args'],
      );

}