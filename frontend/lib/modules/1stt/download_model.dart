/*
 * Copyright (c) 田梓萱[小草林] 2021-2024.
 * All Rights Reserved.
 * All codes are protected by China's regulations on the protection of computer software, and infringement must be investigated.
 * 版权所有 (c) 田梓萱[小草林] 2021-2024.
 * 所有代码均受中国《计算机软件保护条例》保护，侵权必究.
 */

import "dart:io";

import "package:flutter/foundation.dart";

/// Available whisper models
enum WhisperModel {
  // no model
  none(""),

  /// tiny model for all languages
  tiny("tiny"),
  tinyQ5_1("tiny-q5_1"),
  tinyQ8_0("tiny-q8_0"),
  tinyEn("tiny.en"),
  tinyEnQ5_1("tiny.en-q5_1"),
  tinyEnQ8_0("tiny.en-q8_0"),

  /// base model for all languages
  base("base"),
  baseQ5_1("base-q5_1"),
  baseQ8_0("base-q8_0"),
  baseEn("base.en"),
  baseEnQ5_1("base.en-q5_1"),
  baseEnQ8_0("base.en-q8_0"),

  /// small model for all languages
  small("small"),
  smallQ5_1("small-q5_1"),
  smallQ8_0("small-q8_0"),
  smallEn("small.en"),
  smallEnQ5_1("small.en-q5_1"),
  smallEnQ8_0("small.en-q8_0"),
  smallEnTdrz("small.en-tdrz"),

  /// medium model for all languages
  medium("medium"),
  mediumQ5_0("medium-q5_0"),
  mediumQ8_0("medium-q8_0"),
  mediumEn("medium.en"),
  mediumEnQ5_0("medium.en-q5_0"),
  mediumEnQ8_0("medium.en-q8_0"),

  /// large model for all languages
  largeV1("large-v1"),
  largeV2("large-v2"),
  largeV2Q5_0("large-v2-q5_0"),
  largeV2Q8_0("large-v2-q8_0"),
  largeV3("large-v3"),
  largeV3Q5_0("large-v3-q5_0"),
  largeV3Turbo("large-v3-turbo"),
  largeV3TurboQ5_0("large-v3-turbo-q5_0"),
  largeV3TurboQ8_0("large-v3-turbo-q8_0");

  const WhisperModel(this.modelName);

  /// Public name of model
  final String modelName;

  /// Get local path of model file
  String getPath(String dir) {
    return "$dir/ggml-$modelName.bin";
  }
}

/// Download [model] to [destinationPath]
Future<String> downloadModel({
  required WhisperModel model,
  required String destinationPath,
  String? downloadHost,
}) async {
  if (kDebugMode) {
    debugPrint("Download model ${model.modelName}");
  }
  final httpClient = HttpClient();

  Uri modelUri;

  if (downloadHost == null || downloadHost.isEmpty) {
    /// Huggingface url to download model
    modelUri = Uri.parse(
      "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-${model.modelName}.bin",
    );
  } else {
    modelUri = Uri.parse("$downloadHost/ggml-${model.modelName}.bin");
  }

  final request = await httpClient.getUrl(modelUri);

  final response = await request.close();

  final file = File("$destinationPath/ggml-${model.modelName}.bin");
  final raf = file.openSync(mode: FileMode.write);

  await for (var chunk in response) {
    raf.writeFromSync(chunk);
  }

  await raf.close();

  if (kDebugMode) {
    debugPrint("Download Down . Path = ${file.path}");
  }
  return file.path;
}
