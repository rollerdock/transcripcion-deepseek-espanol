// Client half of the Spanish Whisper voice-input bundle.
// Adds a minimal setup/preparation card to this bundle's page so the user can
// download and prepare the local Whisper model (the shipped voice-input UI is
// keyed to the official bundle name, so this bundle owns its own card).
window.__ModuleLoader__.load({
  id: "@local/dsh-voice-input-es",
  factory(require) {
    const React = require("react");
    const h = React.createElement;
    const { useState, useEffect } = React;

    const PHASE_TEXT = {
      unprepared: "Primer uso: hay que descargar el modelo Whisper large-v3 (~3 GB).",
      checking: "Comprobando recursos locales…",
      loading: "Descargando/cargando el modelo…",
      waking: "Activando el reconocimiento…",
      ready: "Reconocimiento local listo.",
      standby: "Modelo preparado; se activa al grabar.",
      cancelled: "Preparación cancelada.",
      cancelling: "Cancelando preparación…",
      failed: "Preparación fallida.",
    };

    const LANG = { es: "Español", auto: "Auto (detectar)", en: "Inglés", fr: "Francés", de: "Alemán", pt: "Portugués", it: "Italiano", ca: "Catalán" };

    function WhisperSetupCard({ catalog, prepare, cancelPreparation, configure }) {
      const [snapshot, setSnapshot] = useState(null);
      const [connError, setConnError] = useState("");
      const [busy, setBusy] = useState(false);
      const [actionError, setActionError] = useState("");

      useEffect(() => {
        let disposed = false;
        const refresh = async () => {
          try {
            const value = await catalog();
            if (!disposed) { setSnapshot(value); setConnError(""); }
          } catch (error) {
            if (!disposed) setConnError(error && error.message ? error.message : String(error));
          }
        };
        refresh();
        const timer = setInterval(refresh, 2000);
        return () => { disposed = true; clearInterval(timer); };
      }, [catalog]);

      const providers = snapshot && snapshot.providers ? snapshot.providers : [];
      const provider = providers.find((item) => item.id === (snapshot && snapshot.selection ? snapshot.selection.providerId : null)) ?? providers[0];

      if (connError && providers.length === 0) {
        return h("p", { role: "alert" }, "No se pudo conectar con el servicio de voz: " + connError);
      }
      if (!provider) return h("p", { role: "status" }, "Cargando servicio de voz…");

      const prep = provider.preparation || { phase: "ready" };
      const phase = prep.phase;
      const canPrepare = phase === "unprepared" || phase === "cancelled" || phase === "failed";
      const preparing = phase === "checking" || phase === "loading" || phase === "waking";
      const languages = Array.isArray(provider.languages) ? provider.languages : [];
      const selectedLanguage = snapshot && snapshot.selection ? snapshot.selection.language : "es";

      const run = async (action) => {
        setActionError("");
        setBusy(true);
        try { await action(); } catch (error) { setActionError(error && error.message ? error.message : String(error)); }
        finally { setBusy(false); }
      };

      const phaseText = (phase === "failed" && prep.message) ? "Preparación fallida: " + prep.message : (PHASE_TEXT[phase] || phase);

      return h("section", {
        style: {
          border: "1px solid var(--dsw-alias-border-l1, #d0d5dd)",
          borderRadius: 10,
          padding: 14,
          margin: "12px 0",
          display: "flex",
          flexDirection: "column",
          gap: 10,
          fontSize: 13,
        },
      },
        h("strong", null, provider.name),
        h("p", { role: phase === "failed" ? "alert" : "status", style: { margin: 0 } }, phaseText),
        languages.length > 1 && h("label", { style: { display: "flex", flexDirection: "column", gap: 6 } },
          "Idioma de reconocimiento",
          h("select", {
            value: selectedLanguage,
            disabled: busy,
            onChange: (event) => run(() => configure({ language: event.target.value })),
          }, languages.map((lang) => h("option", { value: lang }, LANG[lang] || lang)))
        ),
        h("div", { style: { display: "flex", gap: 8 } },
          canPrepare && h("button", {
            type: "button",
            disabled: busy,
            onClick: () => run(() => prepare(provider.id)),
            style: { cursor: "pointer" },
          }, phase === "unprepared" ? "Descargar y preparar" : "Reintentar preparación"),
          preparing && h("button", {
            type: "button",
            disabled: busy || phase === "cancelling",
            onClick: () => run(() => cancelPreparation(provider.id)),
            style: { cursor: "pointer" },
          }, "Cancelar preparación")
        ),
        actionError && h("p", { role: "alert", style: { margin: 0, color: "var(--dsw-alias-state-error-primary, #d92d20)" } }, actionError)
      );
    }

    return {
      inject: ["remote", "slots"],
      apply(ctx) {
        ctx.slots.inject("plugins.bundle.config", () => ctx.slots.register({
          name: "plugins.bundle.config",
          key: "@local/dsh-voice-input-es",
          inject: () => ({
            catalog: () => ctx.remote.speech.catalog().then((result) => {
              if (!result.ok) throw result.error;
              return result.value;
            }),
            prepare: (id) => ctx.remote.speech.prepare(id).then((result) => {
              if (!result.ok) throw result.error;
            }),
            cancelPreparation: (id) => ctx.remote.speech.cancelPreparation(id).then((result) => {
              if (!result.ok) throw result.error;
            }),
            configure: (patch) => ctx.remote.speech.configure(patch).then((result) => {
              if (!result.ok) throw result.error;
            }),
          }),
        }, WhisperSetupCard));
      },
    };
  },
});
