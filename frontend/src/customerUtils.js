const CUSTOMER_QUERY_FIELDS = [
  "q",
  "segment",
  "churned",
  "risk_band",
  "subscription_type",
  "region",
  "device",
  "favorite_genre",
  "inactivity_band",
  "watch_band",
  "threshold",
  "page",
  "page_size",
  "sort_by",
  "order",
];

export function buildCustomerQuery(filters) {
  const params = new URLSearchParams();
  for (const field of CUSTOMER_QUERY_FIELDS) {
    const value = filters[field];
    if (value !== undefined && value !== null && value !== "") {
      params.set(field, String(value));
    }
  }
  return params.toString();
}

export function riskBandForProbability(probability) {
  if (!Number.isFinite(probability) || probability < 0 || probability > 1) {
    return null;
  }
  if (probability < 0.33) return "low";
  if (probability < 0.66) return "medium";
  return "high";
}
