// Host half of the Spanish Whisper voice-input bundle.
// Registers a local faster-whisper (large-v3) transcription provider into the
// Harness speech registry (ctx.speechToText) so the shipped voice-input UI can
// transcribe Spanish with the highest-capability local Whisper model.
import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { isAbsolute, join } from "node:path";
import { mkdir, writeFile, rm } from "node:fs/promises";
import { randomUUID } from "node:crypto";
import { fileURLToPath } from "node:url";

const name = "whisper-es";

const MODEL_BYTES = {
  tiny: 7.5e7,
  base: 1.45e8,
  small: 4.84e8,
  medium: 1.53e9,
  large: 3.1e9,
  "large-v2": 3.1e9,
  "large-v3": 3.1e9,
  "large-v3-turbo": 1.62e9,
  "distil-large-v3": 1.54e9,
};

const STEP_ORDER = ["check", "model", "load"];

// Prefer a Python installation that ships faster-whisper. The bare `python` on
// PATH may be a newer release without compatible ML wheels, so check the
// per-user python.org installs first and only then fall back to PATH.
function resolvePython() {
  const candidates = [];
  const localAppData = process.env.LOCALAPPDATA;
  if (localAppData) {
    for (const version of ["312", "313", "311", "314"]) {
      candidates.push(`${localAppData}\\Programs\\Python\\Python${version}\\python.exe`);
    }
  }
  for (const candidate of candidates) {
    if (existsSync(candidate)) return candidate;
  }
  return "python";
}

export const inject = ["speechToText"];

function withTimeout(signal, ms, label, operation) {
  if (signal?.aborted) return Promise.reject(signal.reason ?? new Error("Operación cancelada"));
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(new Error(`${label}: tiempo de espera agotado (${ms}ms)`)), ms);
  const onAbort = () => controller.abort(signal.reason ?? new Error("Operación cancelada"));
  signal?.addEventListener("abort", onAbort, { once: true });
  return Promise.resolve()
    .then(() => operation(controller.signal))
    .finally(() => {
      clearTimeout(timer);
      signal?.removeEventListener("abort", onAbort);
    });
}

export function apply(ctx, config) {
  const cfg = {
    providerId: config.providerId ?? "whisper-es",
    dataRoot: config.dataRoot,
    model: config.model ?? "large-v3-turbo",
    modelDirectory: config.modelDirectory ?? undefined,
    python: config.python ?? resolvePython(),
    device: config.device ?? "cuda",
    computeType: config.computeType ?? "float16",
    beamSize: config.beamSize ?? 1,
    languages: Array.isArray(config.languages) ? config.languages : ["es", "auto"],
    prepareTimeoutMs: config.prepareTimeoutMs ?? 30 * 60 * 1000,
    inferenceTimeoutMs: config.inferenceTimeoutMs ?? 10 * 60 * 1000,
    idleTimeoutMs: config.idleTimeoutMs ?? 30 * 60 * 1000,
    maxLogBytes: config.maxLogBytes ?? 64 * 1024,
    maxPending: config.maxPending ?? 4,
  };

  if (typeof cfg.dataRoot !== "string" || !isAbsolute(cfg.dataRoot)) {
    throw new Error("whisper-es: 'dataRoot' debe ser una ruta absoluta");
  }

  const workerPath = fileURLToPath(new URL("./worker.py", import.meta.url));
  const worker = new WhisperWorker(ctx, cfg, workerPath);

  ctx.effect(() => {
    const unregister = ctx.speechToText.register({
      info: {
        id: cfg.providerId,
        name: `Whisper ${cfg.model} (español)`,
        location: "host-local",
        languages: cfg.languages,
        setupEstimate: {
          recommendedDiskBytes: MODEL_BYTES[cfg.model] ?? 3.1e9,
          expectedMemoryBytes: (MODEL_BYTES[cfg.model] ?? 3.1e9) * 2,
          minimumMinutes: 5,
          maximumMinutes: 60,
        },
      },
      preparation: worker,
      transcribe: (input, signal) => worker.transcribe(input, signal),
    });
    return async () => {
      const removing = unregister();
      await worker.dispose();
      await removing;
    };
  });
}

class WhisperWorker {
  constructor(ctx, cfg, workerPath) {
    this.ctx = ctx;
    this.cfg = cfg;
    this.workerPath = workerPath;
    this.lifetime = new AbortController();
    this.listeners = new Set();
    this.worker = undefined;
    this.tail = Promise.resolve();
    this.pending = 0;
    this.idle = undefined;
    this.preparing = undefined;
    this.stderrChunks = [];
    this.readyRemainder = "";
    this.state = {
      phase: "unprepared",
      steps: STEP_ORDER.map((kind) => ({ kind, status: "pending" })),
    };
  }

  snapshot() {
    return this.state;
  }

  subscribe(listener) {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  publish(state) {
    if (this.preparing && ["ready", "standby", "unprepared", "failed", "cancelled"].includes(state.phase)) {
      this.preparing.completed = true;
    }
    const previous = this.state;
    const steps = state.steps ?? previous.steps.map((item) => {
      if (state.phase === "ready") return { ...item, status: "complete" };
      if (state.phase === "failed" || state.phase === "cancelled") {
        return item.status === "running" ? { ...item, status: state.phase } : item;
      }
      if (state.step === item.kind) {
        return item.status === "running" ? item : { ...item, status: "running", startedAt: Date.now() };
      }
      if (state.step === undefined) return item;
      if (STEP_ORDER.indexOf(item.kind) < STEP_ORDER.indexOf(state.step)) {
        return item.status === "running" ? item : { ...item, status: "complete" };
      }
      return item;
    });
    this.state = { ...state, steps };
    for (const listener of this.listeners) listener();
  }

  prepare(options = {}) {
    if (this.preparing || this.worker) return;
    this.lifetime.signal.throwIfAborted();
    const abort = new AbortController();
    const task = { abort, settled: Promise.resolve(), completed: false };
    this.preparing = task;
    task.settled = this.enqueue((combined) => this.start(combined), abort.signal)
      .catch((error) => {
        if (this.lifetime.signal.aborted) return;
        this.publish(abort.signal.aborted ? { phase: "cancelled" } : { phase: "failed", message: error instanceof Error ? error.message : String(error) });
      })
      .finally(() => {
        this.preparing = undefined;
      });
  }

  async cancel() {
    const task = this.preparing;
    if (!task) {
      if (["checking", "loading", "waking", "cancelling"].includes(this.state.phase)) {
        this.publish({ phase: "cancelled" });
      }
      return;
    }
    if (!task.completed) {
      this.publish({ phase: "cancelling", startedAt: Date.now() });
      task.abort.abort(new Error("Preparación cancelada"));
    }
    await task.settled;
  }

  transcribe(input, signal) {
    return this.enqueue((combined) => this.execute(input, combined), signal);
  }

  enqueue(run, signal) {
    const combined = AbortSignal.any([signal, this.lifetime.signal]);
    if (combined.aborted) return Promise.reject(combined.reason ?? new Error("Operación cancelada"));
    if (this.pending >= this.cfg.maxPending) return Promise.reject(new Error("La cola de transcripción está llena"));
    clearTimeout(this.idle);
    this.pending++;
    const job = this.tail.then(async () => {
      combined.throwIfAborted();
      return await run(combined);
    });
    this.tail = job.then(() => {}, () => {}).finally(() => {
      this.pending--;
      if (this.pending === 0 && !this.lifetime.signal.aborted && this.cfg.idleTimeoutMs > 0) {
        this.idle = setTimeout(() => {
          this.tail = this.tail.then(async () => this.stop());
        }, this.cfg.idleTimeoutMs);
      }
    });
    return job;
  }

  async start(signal) {
    if (this.worker?.closed) await this.stop();
    if (this.worker) return this.worker;

    this.publish({ phase: "checking", step: "check", startedAt: Date.now() });

    // El directorio de trabajo del worker debe existir: `spawn` falla con ENOENT
    // (que se confundiría con "Python no encontrado") si `cwd` no existe.
    await mkdir(this.cfg.dataRoot, { recursive: true });

    const argv = [
      this.cfg.python, "-u", this.workerPath,
      "--model", this.cfg.model,
      "--device", this.cfg.device,
      "--compute-type", this.cfg.computeType,
      "--beam-size", String(this.cfg.beamSize),
      "--model-dir", this.cfg.modelDirectory ?? join(this.cfg.dataRoot, "models"),
    ];

    const child = spawn(argv[0], argv.slice(1), {
      cwd: this.cfg.dataRoot,
      env: { ...process.env, PYTHONUNBUFFERED: "1", PYTHONIOENCODING: "utf-8" },
      stdio: ["pipe", "pipe", "pipe"],
      windowsHide: true,
    });

    this.stderrChunks = [];
    child.stderr.on("data", (chunk) => {
      this.stderrChunks.push(chunk);
      let bytes = 0;
      for (const c of this.stderrChunks) bytes += c.length;
      while (bytes > this.cfg.maxLogBytes && this.stderrChunks.length) {
        bytes -= this.stderrChunks[0].length;
        this.stderrChunks.shift();
      }
    });

    this.publish({ phase: "loading", step: "model", startedAt: Date.now() });

    try {
      const ready = await withTimeout(signal, this.cfg.prepareTimeoutMs, "Preparación de Whisper", (sig) => this.readReadyLine(child, sig));
      this.readyRemainder = this.readyRemainder;
      this.worker = {
        child,
        stdin: child.stdin,
        closed: false,
        buffer: this.readyRemainder,
        pendingResolve: undefined,
      };
      this.readyRemainder = "";

      child.stdout.on("data", (chunk) => this.onStdout(chunk));
      child.on("exit", () => {
        if (this.worker && !this.worker.closed) {
          this.worker.closed = true;
          if (this.worker.pendingResolve) {
            const resolve = this.worker.pendingResolve;
            this.worker.pendingResolve = undefined;
            resolve(JSON.stringify({ error: "El worker terminó durante la transcripción" }));
          }
          this.publish({ phase: "standby" });
        }
      });

      this.publish({ phase: "ready" });
      return this.worker;
    } catch (error) {
      try { child.kill(); } catch {}
      this.publish({ phase: "failed", message: error instanceof Error ? error.message : String(error) });
      throw error;
    }
  }

  readReadyLine(child, signal) {
    if (signal.aborted) return Promise.reject(signal.reason ?? new Error("Operación cancelada"));
    return new Promise((resolve, reject) => {
      let buffer = "";
      const onData = (chunk) => {
        buffer += chunk.toString("utf8");
        const idx = buffer.indexOf("\n");
        if (idx < 0) return;
        const line = buffer.slice(0, idx).trim();
        const remainder = buffer.slice(idx + 1);
        cleanup();
        this.readyRemainder = remainder;
        try {
          const msg = JSON.parse(line);
          if (msg && msg.ready) resolve(msg);
          else reject(new Error("Respuesta inesperada del worker: " + line));
        } catch (error) {
          reject(new Error("Respuesta inválida del worker: " + line));
        }
      };
      const onError = (error) => {
        cleanup();
        if (error && error.code === "ENOENT") {
          reject(new Error(`No se encontró el intérprete Python "${this.cfg.python}". Instálalo y asegúrate de que esté en el PATH, o apunta la configuración 'python' a un intérprete válido.`));
        } else {
          reject(new Error(`No se pudo iniciar el worker de Whisper: ${error?.message ?? error}`));
        }
      };
      const onExit = (code) => {
        cleanup();
        const tail = this.stderrTail();
        if (tail.includes("FASTER_WHISPER_MISSING")) {
          reject(new Error("faster-whisper no está instalado en el Python seleccionado. Instálalo con: pip install faster-whisper"));
        } else if (tail.includes("MODEL_LOAD_FAILED")) {
          reject(new Error("No se pudo cargar el modelo Whisper: " + tail.slice(-500)));
        } else {
          reject(new Error(`El worker de Whisper terminó antes de estar listo (código ${code}). ${tail.slice(-500)}`));
        }
      };
      const onAbort = () => {
        cleanup();
        reject(signal.reason ?? new Error("Operación cancelada"));
      };
      const cleanup = () => {
        child.stdout.off("data", onData);
        child.off("error", onError);
        child.off("exit", onExit);
        signal.removeEventListener("abort", onAbort);
      };
      child.stdout.on("data", onData);
      child.on("error", onError);
      child.on("exit", onExit);
      signal.addEventListener("abort", onAbort, { once: true });
    });
  }

  onStdout(chunk) {
    const worker = this.worker;
    if (!worker) return;
    worker.buffer += chunk.toString("utf8");
    let idx;
    while ((idx = worker.buffer.indexOf("\n")) >= 0) {
      const line = worker.buffer.slice(0, idx).trim();
      worker.buffer = worker.buffer.slice(idx + 1);
      if (worker.pendingResolve) {
        const resolve = worker.pendingResolve;
        worker.pendingResolve = undefined;
        resolve(line);
      }
    }
  }

  async execute(input, signal) {
    const worker = await this.start(signal);
    if (worker.stdin === undefined) throw new Error("El worker no tiene stdin disponible");

    const tmpDir = join(this.cfg.dataRoot, "tmp");
    await mkdir(tmpDir, { recursive: true });
    const wavPath = join(tmpDir, `rec-${randomUUID()}.wav`);
    await writeFile(wavPath, Buffer.from(input.audio));

    try {
      const line = await withTimeout(signal, this.cfg.inferenceTimeoutMs, "Transcripción de Whisper", (sig) => this.requestLine(worker, { wav: wavPath, language: input.language }, sig));
      let result;
      try {
        result = JSON.parse(line);
      } catch (error) {
        throw new Error("Respuesta inválida del worker: " + line);
      }
      if (result && result.error) throw new Error(result.error);
      if (!result || typeof result.text !== "string") throw new Error("Respuesta de transcripción incompleta");
      return {
        text: result.text,
        audioSeconds: Number(result.audioSeconds) || 0,
        inferenceSeconds: Number(result.inferenceSeconds) || 0,
      };
    } finally {
      await rm(wavPath, { force: true }).catch(() => {});
    }
  }

  requestLine(worker, payload, signal) {
    if (signal.aborted) return Promise.reject(signal.reason ?? new Error("Operación cancelada"));
    return new Promise((resolve, reject) => {
      worker.pendingResolve = resolve;
      worker.stdin.write(JSON.stringify(payload) + "\n", (error) => {
        if (error) {
          worker.pendingResolve = undefined;
          reject(error);
        }
      });
    });
  }

  stderrTail() {
    return Buffer.concat(this.stderrChunks.slice(-32)).toString("utf8");
  }

  async stop() {
    const worker = this.worker;
    if (!worker) return;
    worker.closed = true;
    try {
      worker.child.kill();
    } catch {}
    this.worker = undefined;
    this.publish({ phase: "standby" });
  }

  async dispose() {
    this.listeners.clear();
    this.lifetime.abort(new Error("Proveedor Whisper descargado"));
    clearTimeout(this.idle);
    try {
      await this.tail;
    } finally {
      await this.stop();
    }
    await this.preparing?.settled;
  }
}

export { name };
