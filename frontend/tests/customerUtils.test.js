import assert from "node:assert/strict";
import test from "node:test";
import { buildCustomerQuery, riskBandForProbability } from "../src/customerUtils.js";

test("customer query preserves active filters and pagination", () => {
  const params = new URLSearchParams(
    buildCustomerQuery({
      q: "basic plan",
      subscription_type: "Basic",
      page: 2,
      page_size: 10,
      threshold: 0.45,
      region: "",
    }),
  );
  assert.equal(params.get("q"), "basic plan");
  assert.equal(params.get("subscription_type"), "Basic");
  assert.equal(params.get("page"), "2");
  assert.equal(params.get("threshold"), "0.45");
  assert.equal(params.has("region"), false);
});

test("risk bands use documented cutoffs and reject invalid scores", () => {
  assert.equal(riskBandForProbability(0.32), "low");
  assert.equal(riskBandForProbability(0.33), "medium");
  assert.equal(riskBandForProbability(0.66), "high");
  assert.equal(riskBandForProbability(Number.NaN), null);
  assert.equal(riskBandForProbability(1.1), null);
});
