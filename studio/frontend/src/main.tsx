// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import "./index.css";
import { fetchDeviceType } from "./config/env";
import { App } from "./app/app";

const DYNAMIC_IMPORT_RELOAD_KEY = "__unsloth_dynamic_import_reload_at__";
const DYNAMIC_IMPORT_RELOAD_COOLDOWN_MS = 60_000;

const globalCrypto = globalThis.crypto as Crypto | undefined;

let dynamicImportReloadInFlight = false;

function triggerDynamicImportRecovery(reason: unknown): void {
  if (dynamicImportReloadInFlight || typeof window === "undefined") {
    return;
  }

  let shouldReload = true;
  try {
    const now = Date.now();
    const previous = Number.parseInt(
      window.sessionStorage.getItem(DYNAMIC_IMPORT_RELOAD_KEY) ?? "0",
      10,
    );
    shouldReload = !Number.isFinite(previous) || now - previous > DYNAMIC_IMPORT_RELOAD_COOLDOWN_MS;
    if (shouldReload) {
      window.sessionStorage.setItem(DYNAMIC_IMPORT_RELOAD_KEY, String(now));
    }
  } catch {
    // Some embedded browsers can block sessionStorage; fallback to one-time in-memory reload.
    shouldReload = true;
  }

  if (!shouldReload) {
    return;
  }

  dynamicImportReloadInFlight = true;
  console.warn("[app] Reloading after dynamic import failure", reason);
  window.location.reload();
}

function installDynamicImportRecovery(): void {
  if (typeof window === "undefined") {
    return;
  }

  const importFailure =
    /Failed to fetch dynamically imported modules?|Importing a module script failed|error loading dynamically imported modules?/i;

  window.addEventListener("vite:preloadError", (event) => {
    const preloadEvent = event as Event & { payload?: unknown; preventDefault?: () => void };
    preloadEvent.preventDefault?.();
    triggerDynamicImportRecovery(preloadEvent.payload ?? event);
  });

  window.addEventListener("unhandledrejection", (event) => {
    const message = String((event.reason as { message?: string })?.message ?? event.reason ?? "");
    if (!importFailure.test(message)) {
      return;
    }
    event.preventDefault();
    triggerDynamicImportRecovery(event.reason);
  });
}

installDynamicImportRecovery();

if (globalCrypto && typeof globalCrypto.randomUUID !== "function") {
  // Some envs ship `crypto` but no `randomUUID()` (or a non-function stub).
  // Provide a best-effort v4 UUID using `getRandomValues` when available.
  const cryptoRef = globalCrypto;

  function getRandomByte(): number {
    if (typeof cryptoRef.getRandomValues === "function") {
      return cryptoRef.getRandomValues(new Uint8Array(1))[0];
    }
    return Math.floor(Math.random() * 256);
  }

  cryptoRef.randomUUID = (() =>
    "10000000-1000-4000-8000-100000000000".replace(/[018]/g, (c) =>
      (+c ^ (getRandomByte() & (15 >> (+c / 4)))).toString(16),
    )) as Crypto["randomUUID"];
}

const rootElement = document.getElementById("root");
if (!rootElement) {
  throw new Error("Root element not found");
}

fetchDeviceType().then(() => {
  createRoot(rootElement).render(
    <StrictMode>
      <App />
    </StrictMode>,
  );
});
