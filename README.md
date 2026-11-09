<p align="center">
  <img src="icon.svg" width="96" height="96" alt="Transcripción DeepSeek en Español">
</p>

<h1 align="center">Transcripción DeepSeek en Español</h1>

<p align="center">
  Complemento (bundle) para <b>DeepSeek Harness</b> que añade <b>entrada por voz en español</b>:
  grabas con el micrófono y el texto se inserta en el editor. Transcripción
  <b>local, privada y rápida</b> con <code>faster-whisper</code> (Whisper) y
  <b>aceleración por GPU CUDA</b>.
</p>

---

## ✨ Características

- 🎙️ **Botón de micrófono** integrado en el editor de la conversación.
- 🇪🇸 Transcripción en **español** (idioma `es` y detección automática opcional).
- ⚡ **Muy rápido**: modelo `large-v3-turbo`, decodificación greedy (`beamSize: 1`) y GPU CUDA.
- 🔒 **Privado**: el audio se procesa en tu equipo, no se envía a la nube.
- 🧠 El modelo se mantiene cargado 30 min para que los dictados seguidos sean casi instantáneos.
- 🛟 Si la GPU falla, el worker **cae automáticamente a CPU** sin romperse.
- 🌐 Interfaz localizada (español, inglés y más).

## 📋 Requisitos

| Requisito | Notas |
|---|---|
| [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) | Con el gestor de plugins (`@deepseek-ai/dsh-plugin-manager`). |
| Python 3.9+ (64 bits) | Con `pip`. |
| `faster-whisper` | `pip install faster-whisper onnxruntime` |
| GPU NVIDIA + CUDA (opcional) | Para máxima velocidad. Sin GPU funciona en CPU. |
| Runtime CUDA (si usas GPU) | `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cuda-runtime-cu12` |

## 🚀 Instalación

### 1) Clona el repositorio

```bash
git clone https://github.com/rollerdock/transcripcion-deepseek-espanol.git
```

### 2) Instala las dependencias de Python

```bash
pip install faster-whisper onnxruntime

# Opcional (recomendado si tienes GPU NVIDIA) para acelerar por CUDA:
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cuda-runtime-cu12
```

> Usa el Python que quieras que emplee el complemento. Si `python` del PATH no
> es ese intérprete, configura su ruta absoluta en `python` (ver Configuración).

### 3) Instala el bundle en DeepSeek Harness

Desde el agente de DSH, indica la **ruta absoluta** de la carpeta clonada:

```
plugin_manager install_bundle <ruta-absoluta-a-transcripcion-deepseek-espanol>
```

O hazlo desde la interfaz de **Plugins** de DeepSeek Harness.

### 4) Prepara el modelo

Abre la página del complemento y pulsa **«Descargar y preparar»**.
La primera vez descarga el modelo (~1-2 GB) y lo carga en la GPU/CPU.

### 5) Úsalo

Recarga la página de DSH y pulsa el **micrófono** del editor de la conversación.
Dicta en español y pulsa de nuevo para transcribir: el texto se inserta en el borrador.

## ⚙️ Configuración

La fila `whisper-es` del patch admite estas claves:

| Clave | Valor por defecto | Descripción |
|---|---|---|
| `python` | `python` (PATH) | Ruta absoluta al intérprete con faster-whisper. |
| `dataRoot` | `dshHomePath('speech-to-text','whisper')` | Carpeta del modelo y temporales. |
| `device` | `cuda` | `cuda`, `cpu` o `auto`. |
| `computeType` | `float16` | `float16`, `int8_float16`, `int8`, `float32`… |
| `model` | `large-v3-turbo` | `large-v3-turbo`, `medium`, `small`, `large-v3`… |
| `beamSize` | `1` | 1 = greedy (rápido); 5 = algo más preciso. |
| `idleTimeoutMs` | `1800000` | Tiempo que el modelo queda caliente (ms). |

Ejemplo de configuración para CPU o para un modelo más ligero:

```yaml
- id: whisper-es
  name: '@local/dsh-voice-input-es'
  config:
    dataRoot: !!js dshHomePath('speech-to-text', 'whisper')
    device: cpu
    computeType: int8
    model: medium
```

## 🛠️ Solución de problemas

- **`cublas64_12.dll is not found`** → falta el runtime CUDA. Instala los
  paquetes `nvidia-*` de pip o pon `device: cpu`.
- **`faster-whisper no está instalado`** → ejecuta `pip install faster-whisper`
  con el intérprete configurado en `python`.
- **Va lento** → comprueba con `nvidia-smi` que la GPU está en uso y que
  `device: cuda`. Si no tienes GPU, usa `model: small` o `medium`.
- **No aparece el micrófono** → recarga la página de DSH (el cliente se
  actualiza al recargar).

## 📄 Licencia

MIT — ver [LICENSE](LICENSE).
