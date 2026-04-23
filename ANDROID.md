# Android FFI Roadmap

> On-device Vietnamese speech-to-text for Android apps, powered by phostt's Rust inference engine.

---

## Overview

Phostt's inference engine (`phostt::inference::Engine`) is pure Rust and has no dependency on the server stack (`axum`, `tokio` runtime, etc.). This makes it an ideal candidate for on-device STT inside Android applications — no network latency, no cloud API keys, and full privacy.

This document describes the architecture, build process, and integration steps needed to ship phostt as a native library (`libphostt.so`) inside an Android app.

---

## Architecture

```
┌─────────────────────────────────────────┐
│  Kotlin / Android App                   │
│  (UI, microphone, file picker)          │
├─────────────────────────────────────────┤
│  PhosttBridge.kt  ──►  JNI             │
│  (object with external fun declarations)│
├─────────────────────────────────────────┤
│  libphostt.so  ──►  C-ABI FFI          │
│  (src/ffi.rs)                           │
├─────────────────────────────────────────┤
│  phostt::inference::Engine              │
│  (ONNX Runtime + Zipformer-vi RNN-T)    │
└─────────────────────────────────────────┘
```

1. **Kotlin layer** — thin bridge that loads `libphostt.so` and calls the four exported functions.
2. **JNI glue** — generated headers from `javac -h` (or `cargo-jni`). The Kotlin signatures map directly to the C symbols in `src/ffi.rs`.
3. **Rust FFI layer** — `src/ffi.rs` exposes:
   - `phostt_engine_new(model_dir)` → opaque engine handle
   - `phostt_transcribe_file(engine, wav_path)` → allocated C string
   - `phostt_string_free(s)` — frees the C string safely
   - `phostt_engine_free(engine)` — tears down the engine
4. **Inference engine** — `Engine::load` reads ONNX models from disk, `transcribe_file` runs the full encoder + decoder + joiner pipeline.

---

## Prerequisites

- **Android NDK** (r26c or newer recommended). Download via Android Studio SDK Manager or from [developer.android.com/ndk](https://developer.android.com/ndk).
- **Rust toolchain** with Android targets:
  ```sh
  rustup target add aarch64-linux-android
  rustup target add armv7-linux-androideabi
  ```
- **cargo-ndk** — Cargo helper that sets up the NDK compiler and linker:
  ```sh
  cargo install cargo-ndk
  ```
- Ensure `$NDK_HOME` (or `$ANDROID_NDK_HOME`) points to your NDK root.

---

## Building the Rust Library for Android

### 1. Configure the linker (already done in `.cargo/config.toml`)

```toml
[target.aarch64-linux-android]
linker = "aarch64-linux-android35-clang"

[target.armv7-linux-androideabi]
linker = "armv7a-linux-androideabi35-clang"
```

> **Adjust the API level** (`35`) to match your installed NDK version and your app's `minSdkVersion`. Common choices:
> - API 28 — Android 9 (Pie)
> - API 30 — Android 11
> - API 33 — Android 13
> - API 35 — Android 15 (current default)

### 2. Build with cargo-ndk

```sh
cargo ndk -t arm64-v8a -o ./android/app/src/main/jniLibs build --release --features ffi
```

For multiple architectures:

```sh
cargo ndk \
  -t arm64-v8a \
  -t armeabi-v7a \
  -t x86_64 \
  -o ./android/app/src/main/jniLibs \
  build --release --features ffi
```

The resulting `.so` files land in:

```
android/app/src/main/jniLibs/
├── arm64-v8a/libphostt.so
├── armeabi-v7a/libphostt.so
└── x86_64/libphostt.so
```

Gradle packages these automatically into the APK/AAB.

---

## Model Bundling

The Zipformer-vi INT8 model set is ~75 MB on disk:

| File | Size (approx) |
|------|---------------|
| `encoder.int8.onnx` | ~30 MB |
| `decoder.onnx` | ~10 MB |
| `joiner.int8.onnx` | ~20 MB |
| `bpe.model` | ~1 MB |
| `tokens.txt` | ~50 KB |

You have two strategies for shipping models:

### A. Bundle in `assets/` (simplest)

1. Copy the contents of `~/.phostt/models/` into:
   ```
   android/app/src/main/assets/phostt_models/
   ```
2. On first app launch, copy the files from assets to the app's private storage:
   ```kotlin
   val assetManager = context.assets
   val modelDir = File(context.filesDir, "phostt_models")
   copyAssets(assetManager, "phostt_models", modelDir)
   ```
3. Pass `modelDir.absolutePath` to `PhosttBridge.engineNew(...)`.

### B. Download on first run (smaller APK)

1. Ship a tiny stub APK.
2. On first launch, download the model tarball from your own CDN or the [sherpa-onnx release](https://github.com/k2-fsa/sherpa-onnx/releases).
3. Extract to the app's private files directory.

> ⚠️ The total APK size with bundled assets is ~80–100 MB (models + compressed `.so`). Google Play supports up to 200 MB for APKs and larger sizes via App Bundles, but consider using [Play Feature Delivery](https://developer.android.com/guide/playcore/feature-delivery) or download-on-demand for the model files.

---

## ONNX Runtime on Android

The `ort` crate supports Android through multiple execution providers:

- **NNAPI** (Neural Networks API) — uses the device's NPU / DSP when available. Enabled automatically when you build with `--features ffi` because the `ffi` feature pulls in `ort/nnapi`.
- **CPU** — pure CPU fallback, works on every device. This is the default when NNAPI is unavailable or fails to initialize.
- **GPU (Vulkan / OpenCL)** — `ort` may add future GPU EPs for Android; check the [`ort` documentation](https://docs.rs/ort) for updates.

No code changes are required to switch between EPs; `ort` selects the best available provider at session creation time.

---

## Kotlin / JNI Bridge Skeleton

The skeleton lives at [`ffi/android/PhosttBridge.kt`](ffi/android/PhosttBridge.kt).

### File transcription

```kotlin
class MainActivity : AppCompatActivity() {

    private var enginePtr: Long = 0L

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val modelDir = File(filesDir, "phostt_models")
        enginePtr = PhosttBridge.engineNew(modelDir.absolutePath)
        if (enginePtr == 0L) {
            Toast.makeText(this, "Failed to load STT engine", Toast.LENGTH_LONG).show()
            return
        }
    }

    fun transcribe(wavPath: String): String {
        if (enginePtr == 0L) return ""
        return PhosttBridge.transcribeFile(enginePtr, wavPath)
    }

    override fun onDestroy() {
        if (enginePtr != 0L) {
            PhosttBridge.engineFree(enginePtr)
            enginePtr = 0L
        }
        super.onDestroy()
    }
}
```

### Real-time streaming

```kotlin
class StreamingActivity : AppCompatActivity() {

    private var enginePtr: Long = 0L
    private var streamPtr: Long = 0L
    private val audioBuffer = ByteArray(16000 * 2) // 1 second of 16 kHz PCM16

    fun startStreaming() {
        val modelDir = File(filesDir, "phostt_models")
        enginePtr = PhosttBridge.engineNew(modelDir.absolutePath)
        if (enginePtr == 0L) { /* handle error */ return }

        streamPtr = PhosttBridge.streamNew(enginePtr)
        if (streamPtr == 0L) { /* handle error */ return }

        // Start AudioRecord in a background thread...
    }

    fun onAudioChunk(pcm16: ByteArray, sampleRate: Int) {
        if (streamPtr == 0L) return
        val json = PhosttBridge.streamProcessChunk(enginePtr, streamPtr, pcm16, sampleRate)
        val segments = JSONArray(json)
        for (i in 0 until segments.length()) {
            val seg = segments.getJSONObject(i)
            val text = seg.getString("text")
            val isFinal = seg.getString("type") == "final"
            runOnUiThread { updateTranscript(text, isFinal) }
        }
    }

    fun stopStreaming() {
        if (streamPtr != 0L) {
            val finalJson = PhosttBridge.streamFlush(enginePtr, streamPtr)
            // ...process final segments...
            PhosttBridge.streamFree(streamPtr)
            streamPtr = 0L
        }
        if (enginePtr != 0L) {
            PhosttBridge.engineFree(enginePtr)
            enginePtr = 0L
        }
    }
}
```

### Generating JNI headers (optional)

If you prefer explicit JNI glue instead of relying on `javac`'s automatic stub generation:

```sh
cd ffi/android
javac -h . PhosttBridge.kt
```

This produces `com_phostt_PhosttBridge.h` with the C function signatures. You can then write a thin C wrapper that forwards to the Rust symbols, or skip this step entirely if the automatic JNI binding works for your toolchain.

---

## Size Considerations

| Component | Approximate Size |
|-----------|------------------|
| `libphostt.so` (arm64, stripped, release, LTO) | ~15–25 MB |
| ONNX models | ~75 MB |
| **Total on-device** | ~90–100 MB |

Tips to reduce binary size:

- `strip = true` and `lto = true` are already enabled in `Cargo.toml` for release builds.
- Build only for `arm64-v8a` if you do not need 32-bit ARM support.
- Use `cargo-ndk` with `--release` (debug builds are much larger).
- Consider downloading models at runtime instead of bundling.

---

## Current Limitations

1. **Server code is excluded** — The FFI build only exposes `phostt::inference::Engine`. The WebSocket server, REST handlers, rate limiting, and `tokio` runtime with `rt-multi-thread` are compiled out when used as a library. They still exist in the server binary (`cargo build --bin phostt`).

2. **Synchronous transcription** — `phostt_transcribe_file` blocks the calling thread while inference runs. Call it from a Kotlin coroutine (`withContext(Dispatchers.IO)`) or `AsyncTask` so the UI thread stays responsive.

3. **No Java exception translation** — Rust errors are logged and returned as `NULL` / empty string. The Kotlin side should treat `engineNew == 0L`, `streamNew == 0L`, or empty string results as failure and surface a generic error message.

5. **Model directory layout is fixed** — `Engine::load` expects exactly the filenames from the sherpa-onnx release (`encoder.int8.onnx`, `decoder.onnx`, `joiner.int8.onnx`, `bpe.model`, `tokens.txt`). Do not rename them.

---

## Next Steps

1. [ ] Write a complete Android sample app (`android/` module) with:
   - `MainActivity` + `ViewModel` + coroutines
   - Audio recording via `AudioRecord` (16 kHz, mono, 16-bit PCM)
   - Real-time streaming FFI bindings
2. [ ] Add GitHub Actions workflow that builds `libphostt.so` for all four ABIs on every release.
3. [ ] Publish the Kotlin bridge as a small Maven artifact (`com.phostt:phostt-android:0.1.0`) so downstream apps only need a Gradle dependency.
4. [ ] Investigate [Oboe](https://github.com/google/oboe) for low-latency audio capture on the native side.

---

## Troubleshooting

### `cargo-ndk` cannot find the linker

Make sure `$NDK_HOME` is set and the NDK version in `.cargo/config.toml` matches your installed NDK. Example:

```sh
export NDK_HOME=$HOME/Android/Sdk/ndk/26.1.10909125
```

Then update `.cargo/config.toml`:

```toml
[target.aarch64-linux-android]
linker = "aarch64-linux-android26-clang"
```

### `ort` fails to compile for Android

Ensure you added the Android Rust targets:

```sh
rustup target add aarch64-linux-android armv7-linux-androideabi
```

If you see linker errors about missing `libclang_rt.builtins`, your NDK installation may be incomplete. Re-install the NDK via Android Studio.

### NNAPI is not used at runtime

Check `adb logcat` for `ort` messages. NNAPI requires:
- Android API 27+ (preferably 28+)
- A device with a compatible NPU / DSP driver
- The model ops must be supported by NNAPI (some ops fall back to CPU automatically)

This is safe — inference still works, just on CPU.

---

*Last updated: 2026-04-21*
