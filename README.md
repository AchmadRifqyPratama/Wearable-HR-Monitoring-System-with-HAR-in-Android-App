# Wearable HR Monitoring System with HAR in Android App

Android app ("Wearable AI") that receives PPG and accelerometer data from a wearable device over Bluetooth, estimates heart rate (HR) on the phone with a Python signal-processing pipeline, and classifies the user's activity (Human Activity Recognition, HAR) with an on-device TensorFlow Lite LSTM model.

## Features

- **Bluetooth connection** – lists paired Bluetooth devices (`SelectDeviceActivity`) and opens an RFCOMM socket to the selected one (`MainActivity`). The *Receive / Stop Receive* button sends `1` / `0` to the device to start/stop streaming.
- **Live sensor readout** – shows the incoming Red PPG, Infrared PPG and X/Y/Z accelerometer values. Each value from the device is terminated by a tag character (`r`, `i`, `x`, `y`, `z`).
- **Activity recognition (HAR)** – every 100 samples, a window of 3-axis accelerometer data (`[1, 100, 3]`) is fed to `model_HAR_RNNLSTM_ACC.tflite` (6 output classes). The UI shows two aggregated scores: **R** (sum of classes 0, 1, 4) and **N** (sum of classes 2, 3, 5).
- **Heart-rate estimation** – every 8 s window (fs = 100 Hz, 1 s step) the PPG and accelerometer buffers are passed via [Chaquopy](https://chaquo.com/chaquopy/) to `HREstimation_v5.py`, which:
  - builds Hankel matrices and applies SVD to separate PPG components from motion-artifact components correlated with the accelerometer,
  - reconstructs the cleaned PPG signals and finds the dominant spectral peak (FFT, 60–180 BPM band) for both PPG channels,
  - returns the averaged BPM, shown as **BPM** in the UI.
- **Simple abnormality flag** – while the HAR "R" score is ≥ 0.5, the app keeps a running average of the BPM; after 100 such estimates it shows `Normal: Y` if the average is within 60–100 BPM, otherwise `Normal: N`.

## Project structure

```
app/src/main/
├── AndroidManifest.xml
├── java/com/wearableapp2/wearableai/
│   ├── MainActivity.java          # Bluetooth threads, UI, HAR + HR processing
│   ├── SelectDeviceActivity.java  # paired-device list
│   ├── DeviceListAdapter.java     # RecyclerView adapter for devices
│   ├── DeviceInfoModel.java       # device name/address model
│   └── HARClassifier.java         # TFLite interpreter wrapper
├── ml/model_HAR_RNNLSTM_ACC.tflite  # HAR LSTM model (accelerometer only)
└── python/HREstimation_v5.py      # HR estimation (Chaquopy)
```

## Tech stack

- Java, Android SDK (compileSdk/targetSdk 30, minSdk 24), AndroidX, Material Components
- TensorFlow Lite (`tensorflow-lite`, `tensorflow-lite-support`, select TF ops)
- Chaquopy 10.0.1 (Python in Android) with `numpy`, `scipy`, `pandas`, `tflite-runtime`, `serial`
- Android Gradle Plugin 7.0.0, Gradle 7.0.2 (wrapper)

## Build & run

1. Open the project in Android Studio (Arctic Fox / AGP 7.0 era) with JDK 11.
2. Chaquopy needs a local Python to build. In `app/build.gradle`, update
   `python { buildPython "..." }` to point to your own Python executable
   (it is currently set to a path on the author's machine).
3. The app is built for `armeabi-v7a` only (`ndk.abiFilters`), so run it on a physical ARM device (or add `"x86"` / other ABIs for an emulator).
4. Pair the wearable device with the phone in Android Bluetooth settings first, then in the app tap **Connect**, choose the device, and tap **Receive**.

```bash
./gradlew assembleDebug
```

> Note: the project uses `tensorflow-lite:0.0.0-nightly-SNAPSHOT` and the `jcenter()` repository, both of which may no longer resolve; you may need to pin a released TensorFlow Lite version to build today.
