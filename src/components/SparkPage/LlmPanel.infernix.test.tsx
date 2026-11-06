import { describe, expect, it, vi } from "vitest";
import { render } from "../../testing/render";
import { makeSpark } from "../../testing/fixtures";
import { LlmPanel } from "./LlmPanel";

vi.mock("./LlmTokenTotals", () => ({ LlmTokenTotals: () => null }));
vi.mock("./LlmDailyChart", () => ({ LlmDailyChart: () => null }));
vi.mock("./LlmTrendChart", () => ({ LlmTrendChart: () => null }));

describe("infernix live metrics", () => {
  it("names the engine and shows the engine tiles the probe fills", () => {
    const llm = {
      ...makeSpark().metrics.llm[0],
      backend: "infernix" as const,
      generationTps: 112.4,
      prefillTps: 4089,
      requestsRunning: 1,
      requestsWaiting: 0,
      slotsActive: 1,
      slotsTotal: 6,
      kvCacheUsage: 0.0002,
      mtpAcceptanceRate: 0.5625,
      ttftP95Seconds: 2.2,
      itlP95Seconds: null,
    };
    const { container } = render(<LlmPanel llm={llm} sparkId="fixture" llmPort={8000} />);
    expect(container.textContent).toContain("Infernix · :8000");
    expect(container.textContent).toContain("TTFT p95");
    expect(container.textContent).toContain("2200 ms");
    expect(container.textContent).not.toContain("Inter-token p95");
  });
});
