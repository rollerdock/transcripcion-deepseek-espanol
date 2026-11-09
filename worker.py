#!/usr/bin/env python3
"""Worker de transcripción faster-whisper.

Protocolo de línea JSON sobre stdin/stdout:
  - Al estar listo imprime en stdout: {"ready": true, "model": "..."}
  - Lee de stdin una línea JSON por petición: {"wav": "<ruta>", "language": "es"}
  - Responde en stdout: {"text": "...", "audioSeconds": 12.3, "inferenceSeconds": 1.5}
  - Los errores responden: {"error": "..."}
  - Todo el registro (progreso de descarga/carga) va a stderr.
"""
import sys
import os
import glob
import json
import argparse
import time
import wave
import site

import numpy as np


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def _prepare_cuda_dlls():
    """Registrar las DLL de CUDA instaladas vía pip (paquetes nvidia-*) en Windows.

    ctranslate2 necesita encontrar cublas/cudnn; los paquetes pip de NVIDIA los
    dejan en site-packages/nvidia/*/bin, que no está en el PATH por defecto.
    """
    if sys.platform != "win32":
        return
    try:
        packages = []
        try:
            packages.extend(site.getsitepackages())
        except Exception:
            pass
        try:
            packages.append(site.getusersitepackages())
        except Exception:
            pass
        for packages_dir in packages:
            if not packages_dir:
                continue
            for dll_dir in glob.glob(os.path.join(packages_dir, "nvidia", "*", "bin")):
                if not os.path.isdir(dll_dir):
                    continue
                try:
                    os.add_dll_directory(dll_dir)
                except OSError:
                    pass
                os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")
    except Exception as exc:
        log("AVISO: no se pudieron registrar las DLL de CUDA: %s" % exc)


def _ensure_capable_python():
    """Re-ejecutar con un Python que tenga faster-whisper si el actual no lo tiene.

    El plugin puede arrancar el `python` del PATH (que en este equipo es 3.15 y
    no admite las ruedas de ctranslate2/av/onnxruntime). En ese caso nos
    reemplazamos por un intérprete compatible conservando stdin/stdout, de modo
    que el protocolo con el plugin no cambia.
    """
    try:
        import faster_whisper  # noqa: F401
        return
    except ImportError:
        pass

    candidates = []
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        for version in ("312", "313", "311", "314"):
            candidates.append(os.path.join(local_app_data, "Programs", "Python", "Python" + version, "python.exe"))
    candidates.extend([
        r"C:\Python312\python.exe",
        r"C:\Python313\python.exe",
    ])

    current = os.path.abspath(sys.executable)
    for candidate in candidates:
        if not os.path.isfile(candidate):
            continue
        if os.path.abspath(candidate) == current:
            continue
        try:
            os.execv(candidate, [candidate, "-u", os.path.abspath(__file__)] + sys.argv[1:])
        except OSError:
            continue
    # Si ninguno funciona, seguimos y main() reportará FASTER_WHISPER_MISSING.


def read_wav(path):
    """Leer un WAV mono PCM 16 bits a un array float32 en su frecuencia nativa."""
    with wave.open(path, "rb") as wf:
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    if channels != 1 or width != 2:
        raise ValueError("WAV no soportado: se esperaba mono PCM 16 bits")
    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    return samples, rate


def create_model(args, WhisperModel):
    """Crear WhisperModel con fallback de dispositivo y caché local.

    - Prueba primero el dispositivo pedido y, si falla por falta de CUDA/cuBLAS
      (p. ej. 'cublas64_12.dll is not found'), reintenta en CPU.
    - Intenta cargar de la caché local antes de tocar la red para evitar
      bloqueos de HuggingFace tras la descarga.
    """
    devices = [args.device] if args.device == "cpu" else [args.device, "cpu"]
    last_error = None
    for device in devices:
        compute_type = "int8" if device == "cpu" else args.compute_type
        try:
            model = WhisperModel(
                args.model,
                device=device,
                compute_type=compute_type,
                download_root=args.model_dir,
                local_files_only=True,
            )
            log("Modelo cargado desde caché local (device=%s)." % device)
            return model
        except Exception as local_error:
            last_error = local_error
            message = str(local_error).lower()
            if any(token in message for token in ("cublas", "cudnn", "cuda", ".dll", "gpu")):
                log("Dispositivo '%s' no disponible (%s); probando el siguiente." % (device, local_error))
                continue
            log("Caché local no disponible (%s); se intentará descargar." % local_error)
        try:
            model = WhisperModel(
                args.model,
                device=device,
                compute_type=compute_type,
                download_root=args.model_dir,
                local_files_only=False,
            )
            log("Modelo descargado y cargado (device=%s)." % device)
            return model
        except Exception as exc:
            last_error = exc
            log("Fallo al cargar con device=%s: %s" % (device, exc))
            continue
    raise last_error


def main():
    _prepare_cuda_dlls()
    _ensure_capable_python()

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--compute-type", default="int8")
    ap.add_argument("--beam-size", type=int, default=5)
    args = ap.parse_args()

    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        log("FASTER_WHISPER_MISSING: " + str(exc))
        log("Instala faster-whisper con: pip install faster-whisper")
        sys.exit(2)

    try:
        import onnxruntime  # noqa: F401
        has_vad = True
    except Exception:
        has_vad = False
        log("AVISO: onnxruntime no disponible; se desactiva el filtro VAD "
            "(instala 'onnxruntime' para mejor calidad).")

    log("Cargando modelo Whisper %s (dispositivo=%s, compute=%s)..."
        % (args.model, args.device, args.compute_type))
    t0 = time.time()
    try:
        model = create_model(args, WhisperModel)
    except Exception as exc:
        log("MODEL_LOAD_FAILED: " + str(exc))
        sys.exit(3)
    log("Modelo listo en %.1fs" % (time.time() - t0))
    print(json.dumps({"ready": True, "model": args.model}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as exc:
            print(json.dumps({"error": "petición inválida: %s" % exc}), flush=True)
            continue
        wav = req.get("wav")
        lang = req.get("language") or None
        if not wav:
            print(json.dumps({"error": "falta el campo 'wav'"}), flush=True)
            continue
        lang = None if lang in (None, "", "auto") else lang
        try:
            t = time.time()
            audio, rate = read_wav(wav)
            segments, info = model.transcribe(
                audio,
                language=lang,
                beam_size=args.beam_size,
                vad_filter=has_vad,
            )
            text = "".join(seg.text for seg in segments).strip()
            print(json.dumps({
                "text": text,
                "audioSeconds": float(getattr(info, "duration", 0.0) or 0.0),
                "inferenceSeconds": round(time.time() - t, 3),
            }), flush=True)
        except Exception as exc:
            print(json.dumps({"error": str(exc)}), flush=True)


if __name__ == "__main__":
    main()
