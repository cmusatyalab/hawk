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

class HawkConfig {
  HawkConfig({
    required this.name,
    required this.args,
  });

  String name;
  Map<String, String> args;

  factory HawkConfig.fromJson(Map<String, dynamic> json) => HawkConfig(
        name: json['name'],
        args: json['args'],
      );

}
