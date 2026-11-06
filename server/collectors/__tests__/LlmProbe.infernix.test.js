/**
 * Unit tests for infernix (wallawalla47/infernix) detection and metrics.
 *
 * `fixtures/infernix-metrics.txt` is a real `GET /metrics` scrape from an infernix server
 * (Qwen3.8-Flash-Next, 6 lanes, DFlash2 drafter, INT8 KV) taken with one request decoding.
 * `INFERNIX_MODELS` is the same server's `GET /v1/models`.
 */
import { test, mock } from "node:test";
import { strict as assert } from "node:assert";
import { readFileSync } from "node:fs";
import { LlmProbe } from "../LlmProbe.js";
import { readServerGenerationTokens } from "../LlmStreaming.js";

const INFERNIX_METRICS = readFileSync(new URL("./fixtures/infernix-metrics.txt", import.meta.url), "utf8");

const INFERNIX_MODELS = {
  data: [
    {
      chat_template_caps: { supports_tools: true },
      created: 1794003571,
      id: "Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix",
      max_model_len: 131072,
      meta: { ftype: "nvfp4_mul", n_ctx: 131072, n_ctx_train: 262144, n_embd: 2560, n_params: 0, n_vocab: 248320, size: 72390251776 },
      object: "model",
      owned_by: "infernix",
    },
  ],
  object: "list",
};

/** Replace the value of one unlabelled series, or of the series whose full name-and-labels text is given. */
function advance(txt, changes) {
  let out = txt;
  for (const [series, value] of Object.entries(changes)) {
    const esc = series.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const re = new RegExp(`^${esc} .*$`, "m");
    assert.match(out, re, `fixture has no series ${series}`);
    out = out.replace(re, `${series} ${value}`);
  }
  return out;
}

/** The scrape with every line of the named series removed (an older build that does not export it). */
function without(txt, ...names) {
  return txt
    .split("\n")
    .filter((line) => !names.some((n) => line.startsWith(n) || line.startsWith(`# HELP ${n}`) || line.startsWith(`# TYPE ${n}`)))
    .join("\n");
}

const PREFILL = 'infernix_execution_seconds_total{kind="prefill"}';

test("fixture: the real scrape carries the series the probe maps", () => {
  for (const name of [
    "infernix_generation_tokens_total 85",
    "infernix_prefill_tokens_total 5100",
    "infernix_prompt_tokens_total 5100",
    "infernix_requests_running 1",
    "infernix_max_concurrency 6",
    "infernix_device_kv_capacity_pages 12294",
    'infernix_model_info{model_name="Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix"',
  ]) {
    assert.ok(INFERNIX_METRICS.includes(name), name);
  }
});

test("_metricsLookLikeInfernix: true for infernix exposition only", () => {
  assert.equal(LlmProbe._metricsLookLikeInfernix(INFERNIX_METRICS), true);
  assert.equal(LlmProbe._metricsLookLikeInfernix("infernix_engine_ready 0\n"), true);
  assert.equal(LlmProbe._metricsLookLikeInfernix("vllm:generation_tokens_total 10.0\n"), false);
  assert.equal(LlmProbe._metricsLookLikeInfernix("ninfer_generation_tokens_total 10\n"), false);
  // A series that merely contains the prefix does not count.
  assert.equal(LlmProbe._metricsLookLikeInfernix("my_infernix_generation_tokens_total 10\n"), false);
  assert.equal(LlmProbe._metricsLookLikeInfernix(""), false);
});

function serverFetch({ models = INFERNIX_MODELS, metrics, status = 200, health = true } = {}) {
  return async (url) => {
    const u = String(url);
    if (u.endsWith("/v1/models")) return { ok: true, status: 200, json: async () => models };
    if (u.endsWith("/metrics")) {
      return status === 200
        ? { ok: true, status, text: async () => metrics() }
        : { ok: false, status, text: async () => "", json: async () => ({}) };
    }
    if (u.endsWith("/health") && health) return { ok: true, status: 200, json: async () => ({ status: "ok" }) };
    return { ok: false, status: 404, json: async () => ({}), text: async () => "" };
  };
}

test("_detectServerType: owned_by infernix → infernix, not the vLLM default, without LM Studio or Ollama probes", async () => {
  const probe = new LlmProbe({ lanIp: "127.0.0.1" }, 8000);
  const requested = [];
  const inner = serverFetch({ metrics: () => INFERNIX_METRICS });
  probe._fetch = async (url, ...rest) => {
    requested.push(new URL(String(url)).pathname);
    return inner(url, ...rest);
  };
  await probe._detectServerType();
  assert.equal(probe.backendType, "infernix");
  assert.equal(probe.serverIsOpenAI, true);
  assert.ok(!requested.includes("/api/v0/models") && !requested.includes("/api/tags"), requested.join(","));
});

test("_detectServerType: a known infernix lane is not re-probed for llama.cpp /slots", async () => {
  const probe = new LlmProbe({ lanIp: "127.0.0.1" }, 8000);
  probe.backendType = "infernix";
  const requested = [];
  const inner = serverFetch({ metrics: () => INFERNIX_METRICS });
  probe._fetch = async (url, ...rest) => {
    requested.push(String(url));
    return inner(url, ...rest);
  };
  await probe._detectServerType();
  assert.equal(probe.backendType, "infernix");
  assert.equal(requested.some((u) => u.endsWith("/slots")), false);
});

test("_applyInfernixMetrics: the first sample only seeds; totals, lanes, KV, drafts and latency come from the real scrape", () => {
  const probe = new LlmProbe({ lanIp: "127.0.0.1" }, 8000);
  probe._applyInfernixMetrics(INFERNIX_METRICS, 1_700_000_000);
  assert.equal(probe.measurementValid, false);
  assert.match(probe.measurementError, /earlier sample/);
  assert.equal(probe.generationTps, 0, "lifetime totals are not one interval's work");
  assert.equal(probe.prefillTps, 0);

  assert.equal(probe.totalOutputTokens, 85);
  assert.equal(probe.totalPromptTokens, 5100);
  assert.equal(probe.totalCachedTokens, 0);
  assert.equal(probe.requestsRunning, 1);
  assert.equal(probe.requestsWaiting, 0);
  assert.equal(probe.slotsActive, 1);
  assert.equal(probe.slotsTotal, 6);
  assert.equal(probe.preemptionsTotal, 0);
  // 82 pages in use, 79 of them cached prefix blocks nobody holds: 3 of 12294.
  assert.equal(probe.kvCacheUsage, 0.0002);
  assert.equal(probe.mtpAcceptanceRate, 0.5625); // 45 / 80
  assert.equal(probe.ttftP95Seconds, 2.2); // 4 first tokens: 0.1 s x2, 0.25 s, 2.5 s bucket
  assert.equal(probe.e2eP95Seconds, 2.275); // 3 settled requests: two under 0.1 s, one in the 1 s to 2.5 s bucket
  assert.equal(probe.itlP95Seconds, null, "infernix publishes no inter-token latency");
  assert.equal(probe.gpuMemoryUtilization, 1); // infernix_engine_ready 1 → Active
});

test("_applyInfernixMetrics: decode is the generated-token delta; idle reads 0", () => {
  const probe = new LlmProbe({ lanIp: "127.0.0.1" }, 8000);
  probe._applyInfernixMetrics(INFERNIX_METRICS, 1_700_000_000);

  probe._applyInfernixMetrics(advance(INFERNIX_METRICS, { infernix_generation_tokens_total: 285 }), 2);
  assert.equal(probe.measurementValid, true);
  assert.equal(probe.measurementError, null);
  assert.equal(probe.generationTps, 100); // (285 - 85) / 2
  assert.equal(probe.totalOutputTokens, 285);

  probe._applyInfernixMetrics(advance(INFERNIX_METRICS, { infernix_generation_tokens_total: 285 }), 2);
  assert.equal(probe.generationTps, 0);
});

test("_applyInfernixMetrics: prefill is computed tokens over the engine's prefill time, held while requests are in flight", () => {
  const probe = new LlmProbe({ lanIp: "127.0.0.1" }, 8000);
  probe._applyInfernixMetrics(INFERNIX_METRICS, 1_700_000_000);

  // 4000 more prompt tokens computed in 1 s of prefill execution, inside a 2 s poll: 4000 tok/s, not 2000.
  const burst = advance(INFERNIX_METRICS, {
    infernix_prefill_tokens_total: 9100,
    infernix_prompt_tokens_total: 9100,
    [PREFILL]: 2.569138818,
  });
  probe._applyInfernixMetrics(burst, 2);
  assert.equal(probe.prefillTps, 4000);
  assert.equal(probe.uncachedPrefillTps, 4000);

  // The next poll is decode only (one request still running): the last prefill rate holds.
  probe._applyInfernixMetrics(burst, 2);
  assert.equal(probe.prefillTps, 4000);

  // Once nothing is running or waiting it returns to 0.
  const idle = advance(burst, { infernix_requests_running: 0, infernix_requests_decode_ready: 0 });
  probe._applyInfernixMetrics(idle, 2);
  assert.equal(probe.prefillTps, 0);
});

test("_applyInfernixMetrics: a build without prefill execution time spreads the computed tokens over the poll", () => {
  const probe = new LlmProbe({ lanIp: "127.0.0.1" }, 8000);
  const older = without(INFERNIX_METRICS, "infernix_execution_seconds_total");
  probe._applyInfernixMetrics(older, 1_700_000_000);
  probe._applyInfernixMetrics(advance(older, { infernix_prefill_tokens_total: 9100, infernix_prompt_tokens_total: 9100 }), 2);
  assert.equal(probe.prefillTps, 2000); // 4000 / 2
});

test("_applyInfernixMetrics: prefix-cache hits are the cached share, not computed prefill", () => {
  const probe = new LlmProbe({ lanIp: "127.0.0.1" }, 8000);
  const seeded = advance(INFERNIX_METRICS, { infernix_prompt_tokens_total: 5600, infernix_prompt_tokens_cached_total: 500 });
  probe._applyInfernixMetrics(seeded, 1_700_000_000);
  assert.equal(probe.totalPromptTokens, 5600);
  assert.equal(probe.totalCachedTokens, 500);
  assert.equal(probe.prefixCacheHitRate, 0.0893); // 500 / (5100 computed + 500 cached)

  // 3000 more tokens served from cache, none computed: the main prefill tile must not move.
  probe._applyInfernixMetrics(
    advance(seeded, { infernix_prompt_tokens_total: 8600, infernix_prompt_tokens_cached_total: 3500, infernix_requests_running: 0 }),
    2
  );
  assert.equal(probe.prefillTps, 0);
  assert.equal(probe.cachedPrefillTps, 1500); // 3000 / 2
  assert.equal(probe.uncachedPrefillTps, 0);
});

test("_applyInfernixMetrics: waiting counts paused requests; lanes and running come from their own series", () => {
  const probe = new LlmProbe({ lanIp: "127.0.0.1" }, 8000);
  probe._applyInfernixMetrics(
    advance(INFERNIX_METRICS, {
      infernix_requests_running: 6,
      infernix_requests_waiting: 3,
      infernix_requests_paused: 2,
      infernix_preemptions_total: 4,
    }),
    1_700_000_000
  );
  assert.equal(probe.requestsRunning, 6);
  assert.equal(probe.requestsWaiting, 5);
  assert.equal(probe.slotsActive, 6);
  assert.equal(probe.slotsTotal, 6);
  assert.equal(probe.preemptionsTotal, 4);
});

test("_applyInfernixMetrics: KV fill is used pages over capacity, less reclaimable cached blocks", () => {
  const probe = new LlmProbe({ lanIp: "127.0.0.1" }, 8000);
  probe._applyInfernixMetrics(
    advance(INFERNIX_METRICS, { infernix_device_kv_used_pages: 6200, infernix_prefix_cache_evictable_blocks: 200 }),
    1_700_000_000
  );
  assert.equal(probe.kvCacheUsage, 0.488); // 6000 / 12294

  // Without the prefix-cache gauge the plain occupancy is shown.
  probe._applyInfernixMetrics(
    advance(without(INFERNIX_METRICS, "infernix_prefix_cache_evictable_blocks"), { infernix_device_kv_used_pages: 6147 }),
    2
  );
  assert.equal(probe.kvCacheUsage, 0.5); // 6147 / 12294

  // More evictable blocks than used pages never goes negative.
  probe._applyInfernixMetrics(advance(INFERNIX_METRICS, { infernix_device_kv_used_pages: 10, infernix_prefix_cache_evictable_blocks: 79 }), 2);
  assert.equal(probe.kvCacheUsage, 0);

  // No capacity: no reading rather than a made-up one.
  probe._applyInfernixMetrics(without(INFERNIX_METRICS, "infernix_device_kv_capacity_pages"), 2);
  assert.equal(probe.kvCacheUsage, null);
  assert.equal(probe.kvCacheTokens, null);
});

test("_applyInfernixMetrics: an older build without drafts, histograms or lane count leaves those unset", () => {
  const probe = new LlmProbe({ lanIp: "127.0.0.1" }, 8000);
  const older = without(
    INFERNIX_METRICS,
    "infernix_spec_decode_",
    "infernix_time_to_first_token_seconds",
    "infernix_request_duration_seconds",
    "infernix_max_concurrency",
    "infernix_requests_paused",
    "infernix_preemptions_total"
  );
  probe._applyInfernixMetrics(older, 1_700_000_000);
  probe._applyInfernixMetrics(advance(older, { infernix_generation_tokens_total: 185 }), 2);
  assert.equal(probe.measurementValid, true);
  assert.equal(probe.generationTps, 50);
  assert.equal(probe.mtpAcceptanceRate, null);
  assert.equal(probe.ttftP95Seconds, null);
  assert.equal(probe.e2eP95Seconds, null);
  assert.equal(probe.slotsTotal, 0);
  assert.equal(probe.preemptionsTotal, null);
  assert.equal(probe.requestsWaiting, 0);
});

test("_applyInfernixMetrics: no drafted tokens is no acceptance rate, not 0 %", () => {
  const probe = new LlmProbe({ lanIp: "127.0.0.1" }, 8000);
  probe._applyInfernixMetrics(
    advance(INFERNIX_METRICS, { infernix_spec_decode_draft_tokens_total: 0, infernix_spec_decode_accepted_tokens_total: 0 }),
    1_700_000_000
  );
  assert.equal(probe.mtpAcceptanceRate, null);
});

test("_applyInfernixMetrics: an engine that is not ready is an unmeasured poll", () => {
  const probe = new LlmProbe({ lanIp: "127.0.0.1" }, 8000);
  probe._applyInfernixMetrics(INFERNIX_METRICS, 1_700_000_000);
  probe._applyInfernixMetrics(advance(INFERNIX_METRICS, { infernix_generation_tokens_total: 285 }), 2);
  assert.equal(probe.generationTps, 100);

  probe._applyInfernixMetrics(advance(INFERNIX_METRICS, { infernix_generation_tokens_total: 285, infernix_engine_ready: 0 }), 2);
  assert.equal(probe.measurementValid, false);
  assert.match(probe.measurementError, /not ready/);
  assert.equal(probe.generationTps, 0);
  assert.equal(probe.requestsRunning, null);
});

test("_applyInfernixMetrics: a scrape without the token counters is an unmeasured poll", () => {
  const probe = new LlmProbe({ lanIp: "127.0.0.1" }, 8000);
  probe._applyInfernixMetrics("infernix_engine_ready 1\n", 2);
  assert.equal(probe.measurementValid, false);
  assert.match(probe.measurementError, /counters/);
  assert.equal(probe.requestsRunning, null);
});

test("_applyInfernixMetrics: a sample with no usable interval has no rate and keeps no old one", () => {
  const probe = new LlmProbe({ lanIp: "127.0.0.1" }, 8000);
  probe._applyInfernixMetrics(INFERNIX_METRICS, 1_700_000_000);
  probe._applyInfernixMetrics(advance(INFERNIX_METRICS, { infernix_generation_tokens_total: 285 }), 2);
  assert.equal(probe.generationTps, 100);

  // 12 s since the last poll: 1200 tokens over 12 s is not a 2 s rate and the old 100 tok/s must not stay.
  probe._applyInfernixMetrics(advance(INFERNIX_METRICS, { infernix_generation_tokens_total: 1485 }), 12);
  assert.equal(probe.measurementValid, false);
  assert.match(probe.measurementError, /10 s/);
  assert.equal(probe.generationTps, 0);

  probe._applyInfernixMetrics(advance(INFERNIX_METRICS, { infernix_generation_tokens_total: 1685 }), 2);
  assert.equal(probe.measurementValid, true);
  assert.equal(probe.generationTps, 100);

  probe._applyInfernixMetrics(advance(INFERNIX_METRICS, { infernix_generation_tokens_total: 1885 }), 0);
  assert.equal(probe.measurementValid, false, "an interval of 0 is no interval");
});

test("_applyInfernixMetrics: counters that went backwards (an engine restart) re-seed instead of reading as an idle poll", () => {
  const probe = new LlmProbe({ lanIp: "127.0.0.1" }, 8000);
  probe._applyInfernixMetrics(INFERNIX_METRICS, 1_700_000_000);
  probe._applyInfernixMetrics(advance(INFERNIX_METRICS, { infernix_generation_tokens_total: 285, infernix_prefill_tokens_total: 5500 }), 2);
  assert.equal(probe.generationTps, 100);

  probe._applyInfernixMetrics(advance(INFERNIX_METRICS, { infernix_generation_tokens_total: 12, infernix_prefill_tokens_total: 30 }), 2);
  assert.equal(probe.measurementValid, false);
  assert.match(probe.measurementError, /restarted/);
  assert.equal(probe.generationTps, 0);

  probe._applyInfernixMetrics(advance(INFERNIX_METRICS, { infernix_generation_tokens_total: 212, infernix_prefill_tokens_total: 30 }), 2);
  assert.equal(probe.measurementValid, true);
  assert.equal(probe.generationTps, 100);
});

test("probe(): an infernix lane reports live decode and the served model instead of the vLLM zeros", async () => {
  mock.timers.enable({ apis: ["Date"], now: 1_700_000_000_000 });
  try {
    let metrics = INFERNIX_METRICS;
    const probe = new LlmProbe({ lanIp: "127.0.0.1" }, 8000);
    probe._fetch = serverFetch({ metrics: () => metrics });

    const first = await probe.probe();
    assert.equal(first.backend, "infernix");
    assert.equal(first.available, true);
    assert.equal(first.generationTps, 0); // seeds the baseline

    mock.timers.tick(2000);
    metrics = advance(INFERNIX_METRICS, { infernix_generation_tokens_total: 285 });
    const second = await probe.probe();
    assert.equal(second.backend, "infernix");
    assert.equal(second.measurementValid, true);
    assert.equal(second.generationTps, 100);
    assert.equal(second.requestsRunning, 1);
    assert.equal(second.slotsTotal, 6);
    assert.equal(second.modelId, "Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix");
    assert.equal(second.contextLength, 131072, "the served context, not the training context");
  } finally {
    mock.timers.reset();
  }
});

test("probe(): a lane first labelled vLLM is re-labelled once its /metrics is infernix's", async () => {
  mock.timers.enable({ apis: ["Date"], now: 1_700_000_000_000 });
  try {
    const probe = new LlmProbe({ lanIp: "127.0.0.1" }, 8000);
    // A card without owned_by gives nothing to go on, so detection falls through to the vLLM default.
    const models = { data: [{ id: "served-name", max_model_len: 131072 }] };
    probe._fetch = serverFetch({ models, metrics: () => INFERNIX_METRICS });
    const snap = await probe.probe();
    assert.equal(snap.backend, "infernix");
    assert.equal(snap.totalOutputTokens, 85);
  } finally {
    mock.timers.reset();
  }
});

test("probe(): the API key goes to /metrics, and a refused key is reported, not read as zero throughput", async () => {
  mock.timers.enable({ apis: ["Date"], now: 1_700_000_000_000 });
  try {
    const seen = [];
    let metricsStatus = 200;
    const probe = new LlmProbe({ lanIp: "127.0.0.1", llmApiKeys: { 8000: "sk-test" } }, 8000);
    const originalFetch = globalThis.fetch;
    globalThis.fetch = async (url, init = {}) => {
      const path = new URL(String(url)).pathname;
      seen.push([path, init.headers?.Authorization]);
      if (path === "/v1/models") return { ok: true, status: 200, json: async () => INFERNIX_MODELS };
      if (path === "/metrics") {
        return metricsStatus === 200
          ? { ok: true, status: 200, text: async () => INFERNIX_METRICS }
          : { ok: false, status: metricsStatus, text: async () => "", json: async () => ({}) };
      }
      return { ok: false, status: 404, json: async () => ({}), text: async () => "" };
    };
    try {
      const ok = await probe.probe();
      assert.equal(ok.backend, "infernix");
      assert.ok(seen.some(([path, auth]) => path === "/metrics" && auth === "Bearer sk-test"), JSON.stringify(seen));

      metricsStatus = 401;
      mock.timers.tick(2000);
      const refused = await probe.probe();
      assert.equal(refused.available, false, "a refused key reads as 'needs a key', like any other engine");
      assert.equal(refused.measurementValid, false);
      assert.match(refused.measurementError, /API key/);
      assert.equal(refused.generationTps, 0);
    } finally {
      globalThis.fetch = originalFetch;
    }
  } finally {
    mock.timers.reset();
  }
});

test("probe(): a failed /metrics scrape is an unmeasured poll and recovery does not spike", async () => {
  mock.timers.enable({ apis: ["Date"], now: 1_700_000_000_000 });
  try {
    let metrics = INFERNIX_METRICS;
    let status = 200;
    const probe = new LlmProbe({ lanIp: "127.0.0.1" }, 8000);
    const inner = serverFetch({ metrics: () => metrics });
    probe._fetch = async (url, ...rest) =>
      status !== 200 && String(url).endsWith("/metrics")
        ? { ok: false, status, text: async () => "", json: async () => ({}) }
        : inner(url, ...rest);

    await probe.probe();
    mock.timers.tick(2000);
    metrics = advance(INFERNIX_METRICS, { infernix_generation_tokens_total: 285 });
    const live = await probe.probe();
    assert.equal(live.measurementValid, true);
    assert.equal(live.generationTps, 100);

    status = 500;
    mock.timers.tick(2000);
    const failed = await probe.probe();
    assert.equal(failed.available, true); // the endpoint is reachable
    assert.equal(failed.measurementValid, false); // its stats are not
    assert.match(failed.measurementError, /500/);
    assert.equal(failed.generationTps, 0);
    assert.equal(failed.requestsRunning, null);

    // 5000 tokens were generated while the scrape failed: the first good poll only re-seeds.
    status = 200;
    metrics = advance(INFERNIX_METRICS, { infernix_generation_tokens_total: 5285 });
    mock.timers.tick(2000);
    const recovered = await probe.probe();
    assert.equal(recovered.measurementValid, false);
    assert.match(recovered.measurementError, /re-seeding/);
    assert.equal(recovered.generationTps, 0);

    metrics = advance(INFERNIX_METRICS, { infernix_generation_tokens_total: 5485 });
    mock.timers.tick(2000);
    const next = await probe.probe();
    assert.equal(next.measurementValid, true);
    assert.equal(next.generationTps, 100);
  } finally {
    mock.timers.reset();
  }
});

test("readServerGenerationTokens: infernix generated-token counter", async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (url) => {
    if (String(url).endsWith("/metrics")) {
      return { ok: true, status: 200, text: async () => INFERNIX_METRICS };
    }
    return { ok: false, status: 404 };
  };
  try {
    assert.equal(await readServerGenerationTokens("http://127.0.0.1:8000"), 85);
  } finally {
    globalThis.fetch = originalFetch;
  }
});
