import CustomerInsights from "./CustomerInsights";
import { useCallback, useEffect, useMemo, useState } from "react";
import axios from "axios";
import {
  Activity,
  AlertCircle,
  BarChart3,
  ChevronDown,
  ChevronUp,
  Clock3,
  Download,
  Film,
  LayoutDashboard,
  Lightbulb,
  RefreshCw,
  Search,
  ShieldAlert,
  Users,
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Legend,
  Line,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { buildCustomerQuery } from "./customerUtils";
import "./App.css";

const API = import.meta.env.VITE_API_URL || "http://127.0.0.1:5000/api";
const COLORS = ["#9b87f5", "#51c6a5", "#f4b95f", "#6fa8ff", "#ee829a", "#77d2dd", "#c39df1"];
const INITIAL_FILTERS = () => {
  const params = new URLSearchParams(window.location.search);
  return {
    q: params.get("q") || "",
    segment: params.get("segment") || "",
    churned: params.get("churned") || "",
    risk_band: params.get("risk_band") || "",
    subscription_type: params.get("subscription_type") || "",
    region: params.get("region") || "",
    device: params.get("device") || "",
    favorite_genre: params.get("favorite_genre") || "",
    inactivity_band: params.get("inactivity_band") || "",
    watch_band: params.get("watch_band") || "",
    threshold: Number(params.get("threshold") || 0.5),
    page: Number(params.get("page") || 1),
    page_size: Number(params.get("page_size") || 25),
    sort_by: params.get("sort_by") || "customer_id",
    order: params.get("order") || "asc",
  };
};

function App() {
  const [page, setPage] = useState(
    new URLSearchParams(window.location.search).get("view") || "Dashboard",
  );
  const [dashboard, setDashboard] = useState(null);
  const [segments, setSegments] = useState([]);
  const [recommendations, setRecommendations] = useState([]);
  const [filters, setFilters] = useState(INITIAL_FILTERS);
  const [customerPage, setCustomerPage] = useState(null);
  const [selectedCustomer, setSelectedCustomer] = useState(null);
  const [selectedSegments, setSelectedSegments] = useState([]);
  const [segmentEvaluation, setSegmentEvaluation] = useState(null);
  const [segmentEvaluationError, setSegmentEvaluationError] = useState("");
  const [loading, setLoading] = useState(true);
  const [customerLoading, setCustomerLoading] = useState(false);
  const [customerDetailLoading, setCustomerDetailLoading] = useState(false);
  const [segmentEvaluationLoading, setSegmentEvaluationLoading] = useState(false);
  const [error, setError] = useState("");
  const [customerError, setCustomerError] = useState("");
  const [customerDetailError, setCustomerDetailError] = useState("");
  const [searchCustomerId, setSearchCustomerId] = useState("");

  const queryString = useMemo(() => buildCustomerQuery(filters), [filters]);

  const requestDashboard = useCallback(() =>
    Promise.all([
      axios.get(`${API}/dashboard`),
      axios.get(`${API}/segments`),
      axios.get(`${API}/recommendations`),
    ]), []);

  const loadDashboard = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const [dashboardResponse, segmentsResponse, recommendationResponse] =
        await requestDashboard();
      setDashboard(dashboardResponse.data);
      setSegments(segmentsResponse.data);
      setRecommendations(recommendationResponse.data);
    } catch (requestError) {
      setError(
        requestError.response?.data?.error ||
          "Cannot connect to the analytics API. Start the Flask application and retry.",
      );
    } finally {
      setLoading(false);
    }
  }, [requestDashboard]);

  const reloadProjectData = async () => {
    setLoading(true);
    setError("");
    try {
      await axios.post(`${API}/reload`);
      await loadDashboard();
    } catch (requestError) {
      setError(
        requestError.response?.data?.error ||
          "Project data could not be reloaded. The current valid snapshot remains active.",
      );
      setLoading(false);
    }
  };

  useEffect(() => {
    let active = true;
    requestDashboard()
      .then(([dashboardResponse, segmentsResponse, recommendationResponse]) => {
        if (!active) return;
        setDashboard(dashboardResponse.data);
        setSegments(segmentsResponse.data);
        setRecommendations(recommendationResponse.data);
      })
      .catch((requestError) => {
        if (active) {
          setError(
            requestError.response?.data?.error ||
              "Cannot connect to the analytics API. Start the Flask application and retry.",
          );
        }
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [requestDashboard]);

  useEffect(() => {
    const params = new URLSearchParams(buildCustomerQuery(filters));
    if (page !== "Dashboard") params.set("view", page);
    const query = params.toString();
    window.history.replaceState(
      {},
      "",
      `${window.location.pathname}${query ? `?${query}` : ""}${window.location.hash}`,
    );
  }, [filters, page]);

  useEffect(() => {
    if (page !== "Customers") return undefined;
    let active = true;
    axios
      .get(`${API}/customers?${queryString}`)
      .then((response) => {
        if (active) setCustomerPage({ ...response.data, _query: queryString });
      })
      .catch((requestError) => {
        if (active) {
          setCustomerError(
            requestError.response?.data?.error || "Customer records could not be loaded.",
          );
        }
      })
      .finally(() => {
        if (active) setCustomerLoading(false);
      });
    return () => {
      active = false;
    };
  }, [page, queryString]);

  useEffect(() => {
    if (page !== "Segments") return undefined;
    let active = true;
    axios
      .get(`${API}/segments/evaluation`)
      .then((response) => {
        if (active) setSegmentEvaluation(response.data);
      })
      .catch((requestError) => {
        if (active) {
          setSegmentEvaluationError(
            requestError.response?.data?.error ||
              "Candidate segment scores could not be loaded.",
          );
        }
      })
      .finally(() => {
        if (active) setSegmentEvaluationLoading(false);
      });
    return () => {
      active = false;
    };
  }, [page]);

  const updateFilter = (name, value) => {
    setCustomerLoading(true);
    setCustomerError("");
    setFilters((current) => ({
      ...current,
      [name]: value,
      page: name === "page" ? value : 1,
    }));
  };

  const openCustomer = async (customerId) => {
    setSelectedCustomer(null);
    setCustomerDetailError("");
    setCustomerDetailLoading(true);
    try {
      const response = await axios.get(
        `${API}/customer/${encodeURIComponent(customerId)}`,
      );
      setSelectedCustomer(response.data);
    } catch (requestError) {
      setCustomerDetailError(
        requestError.response?.data?.error || "Customer details could not be loaded.",
      );
    } finally {
      setCustomerDetailLoading(false);
    }
  };

  const submitCustomerSearch = async (event) => {
    event.preventDefault();
    if (!searchCustomerId.trim()) return;
    setCustomerLoading(true);
    setCustomerError("");
    setPage("Customers");
    updateFilter("q", searchCustomerId.trim());
    await openCustomer(searchCustomerId.trim());
  };

  const drillDown = (nextFilters) => {
    setSelectedCustomer(null);
    setCustomerLoading(true);
    setCustomerError("");
    setFilters((current) => ({ ...current, ...nextFilters, page: 1 }));
    setPage("Customers");
  };

  const exportFiltered = () => {
    window.location.assign(`${API}/customers/export.csv?${queryString}`);
  };

  const sortCustomers = (sortBy) => {
    setCustomerLoading(true);
    setCustomerError("");
    setFilters((current) => ({
      ...current,
      sort_by: sortBy,
      order:
        current.sort_by === sortBy && current.order === "asc" ? "desc" : "asc",
      page: 1,
    }));
  };

  const navigate = (pageName) => {
    if (pageName === "Customers") {
      setCustomerLoading(true);
      setCustomerError("");
    }
    if (pageName === "Segments") {
      setSegmentEvaluationLoading(true);
      setSegmentEvaluationError("");
    }
    setPage(pageName);
  };

  const renderStatus = (message, isError = false) => (
    <div className={`status-panel${isError ? " error-panel" : ""}`} role={isError ? "alert" : "status"}>
      {isError ? <AlertCircle size={20} /> : <div className="loading-spinner" />}
      <p>{message}</p>
      {isError && (
        <button className="secondary-button" type="button" onClick={loadDashboard}>
          Retry
        </button>
      )}
    </div>
  );

  const renderDashboard = () => {
    if (loading) return renderStatus("Loading verified customer analytics...");
    if (error) return renderStatus(error, true);
    if (!dashboard) return renderStatus("No dashboard data is available.");
    const chartClick = (filterName, value) => {
      if (value !== undefined && value !== null) {
        drillDown({ [filterName]: String(value) });
      }
    };

    return (
      <>
        <div className="stats-grid">
          <KpiCard icon={<Users size={19} />} title="Customers" value={dashboard.total_customers.toLocaleString()} />
          <KpiCard icon={<ShieldAlert size={19} />} title="Recorded churn" value={`${dashboard.churn_rate}%`} />
          <KpiCard icon={<Clock3 size={19} />} title="Avg. viewing" value={`${dashboard.average_watch_hours} h`} />
          <KpiCard icon={<Activity size={19} />} title="Active customers" value={dashboard.active_customers.toLocaleString()} detail="Last login within 7 days" />
          <KpiCard icon={<BarChart3 size={19} />} title="Segments" value={dashboard.segment_count} />
        </div>
        <section className="charts-grid">
          <ChartPanel title="Churn by segment" description="Recorded churn label by existing cluster; select a bar to inspect its customers.">
            {dashboard.churn_by_segment.length ? (
              <ResponsiveContainer width="100%" height={270}>
                <BarChart data={dashboard.churn_by_segment} onClick={(event) => chartClick("segment", event?.activePayload?.[0]?.payload?.cluster)}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                  <YAxis unit="%" />
                  <Tooltip formatter={(value) => [`${value}%`, "Churn rate"]} />
                  <Bar dataKey="churn_rate" name="Churn rate" fill="#ee829a" radius={[5, 5, 0, 0]} cursor="pointer" />
                </BarChart>
              </ResponsiveContainer>
            ) : <EmptyChart message="Cluster assignments are unavailable." />}
          </ChartPanel>
          <ChartPanel title="Subscription mix" description="Customer count by current plan. Select a slice to filter customers.">
            {dashboard.subscription_mix.length ? (
              <ResponsiveContainer width="100%" height={270}>
                <PieChart>
                  <Pie data={dashboard.subscription_mix} dataKey="customers" nameKey="name" innerRadius={58} outerRadius={92} paddingAngle={3} onClick={(item) => chartClick("subscription_type", item.name)}>
                    {dashboard.subscription_mix.map((item, index) => <Cell key={item.name} fill={COLORS[index % COLORS.length]} cursor="pointer" />)}
                  </Pie>
                  <Tooltip formatter={(value) => [value, "Customers"]} />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            ) : <EmptyChart message="Subscription data is unavailable." />}
          </ChartPanel>
          <ChartPanel title="Favourite genre popularity" description="Customer count by recorded favourite genre; select a bar to filter.">
            <ResponsiveContainer width="100%" height={270}>
              <BarChart data={dashboard.genres} onClick={(event) => chartClick("favorite_genre", event?.activePayload?.[0]?.payload?.name)}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                <YAxis allowDecimals={false} />
                <Tooltip formatter={(value) => [value, "Customers"]} />
                <Bar dataKey="users" name="Customers" fill="#9b87f5" radius={[5, 5, 0, 0]} cursor="pointer" />
              </BarChart>
            </ResponsiveContainer>
          </ChartPanel>
          <ChartPanel title="Viewing-hours distribution" description="Recorded cumulative viewing hours per customer; select a range to filter.">
            <ResponsiveContainer width="100%" height={270}>
              <BarChart data={dashboard.watch_distribution} onClick={(event) => {
                const value = event?.activePayload?.[0]?.payload?.range;
                const bands = { "0–5": "0-5", ">5–10": ">5-10", ">10–20": ">10-20", ">20–40": ">20-40", ">40": ">40" };
                if (value) chartClick("watch_band", bands[value]);
              }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="range" tick={{ fontSize: 11 }} />
                <YAxis allowDecimals={false} />
                <Tooltip formatter={(value) => [value, "Customers"]} />
                <Bar dataKey="customers" name="Customers" fill="#51c6a5" radius={[5, 5, 0, 0]} cursor="pointer" />
              </BarChart>
            </ResponsiveContainer>
          </ChartPanel>
          <ChartPanel title="Churn and inactivity" description="Recorded churn rate by days since login; descriptive association, not causation. Select a bar to filter.">
            <ResponsiveContainer width="100%" height={270}>
              <BarChart data={dashboard.churn_by_inactivity} onClick={(event) => {
                const value = event?.activePayload?.[0]?.payload?.name;
                const bands = { "0–7 days": "0-7", "8–14 days": "8-14", "15–30 days": "15-30", ">30 days": ">30" };
                if (value) chartClick("inactivity_band", bands[value]);
              }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                <YAxis unit="%" />
                <Tooltip formatter={(value) => [`${value}%`, "Churn rate"]} />
                <Bar dataKey="churn_rate" name="Churn rate" fill="#f4b95f" radius={[5, 5, 0, 0]} cursor="pointer" />
              </BarChart>
            </ResponsiveContainer>
          </ChartPanel>
        </section>
        <DataQualityPanel quality={dashboard.data_quality} />
        <ModelInfoPanel model={dashboard.model} />
        <p className="notice">
          Segment and inactivity differences are descriptive. They do not prove
          that a customer attribute or behaviour causes churn. “Active” means a
          last login within 7 days; the source contains no event dates.
        </p>
      </>
    );
  };

  const renderCustomers = () => {
    if (error && !dashboard) return renderStatus(error, true);
    return (
      <div className="customer-workspace">
        <div className="workspace-heading">
          <div>
            <h2>Customer retention workspace</h2>
            <p>Search and filter the current customer snapshot. Risk is an estimate, not a guarantee.</p>
          </div>
          <button className="secondary-button" onClick={exportFiltered} disabled={!customerPage?.total}>
            <Download size={16} /> Export filtered CSV
          </button>
        </div>
        <div className="filter-grid">
          <FilterText label="Search customers" value={filters.q} onChange={(value) => updateFilter("q", value)} placeholder="ID, plan, region..." />
          <FilterSelect label="Segment" value={filters.segment} onChange={(value) => updateFilter("segment", value)} options={[["", "All segments"], ...segments.map((item) => [item.cluster, `${item.name} (ID ${item.cluster})`])]} />
          <FilterSelect label="Churn label" value={filters.churned} onChange={(value) => updateFilter("churned", value)} options={[["", "All labels"], ["0", "Not marked churned"], ["1", "Marked churned"]]} />
          <FilterSelect label="Predicted risk" value={filters.risk_band} onChange={(value) => updateFilter("risk_band", value)} options={[["", "All risk bands"], ["low", "Low (<33%)"], ["medium", "Medium (33–65.9%)"], ["high", "High (≥66%)"]]} />
          <FilterSelect label="Subscription plan" value={filters.subscription_type} onChange={(value) => updateFilter("subscription_type", value)} options={[["", "All plans"], ["Basic", "Basic"], ["Standard", "Standard"], ["Premium", "Premium"]]} />
          <FilterSelect label="Region" value={filters.region} onChange={(value) => updateFilter("region", value)} options={[["", "All regions"], ...["Africa", "Asia", "Europe", "North America", "Oceania", "South America"].map((item) => [item, item])]} />
          <FilterSelect label="Device" value={filters.device} onChange={(value) => updateFilter("device", value)} options={[["", "All devices"], ...["Desktop", "Laptop", "Mobile", "TV", "Tablet"].map((item) => [item, item])]} />
          <FilterSelect label="Favourite genre" value={filters.favorite_genre} onChange={(value) => updateFilter("favorite_genre", value)} options={[["", "All genres"], ...(dashboard?.genres || []).map((item) => [item.name, item.name])]} />
        </div>
        <div className="threshold-control">
          <label htmlFor="threshold">Churn classification threshold: <strong>{Number(filters.threshold).toFixed(2)}</strong></label>
          <input id="threshold" type="range" min="0" max="1" step="0.05" value={filters.threshold} onChange={(event) => updateFilter("threshold", Number(event.target.value))} />
          <p>A lower threshold flags more customers (typically more false positives, fewer missed churners); a higher threshold flags fewer (typically fewer false positives, more false negatives).</p>
          <span>Risk bands: low &lt;33%, medium 33–65.9%, high ≥66%.</span>
        </div>
        {customerLoading && <div className="inline-status" role="status">Loading filtered customers…</div>}
        {customerError && <div className="error-message" role="alert">{customerError}</div>}
        {!customerLoading && !customerError && customerPage && (
          <>
            <div className="table-summary">
              <span><strong>{customerPage.summary.customers.toLocaleString()}</strong> customers</span>
              <span><strong>{customerPage.summary.churn_rate}%</strong> recorded churn</span>
              <span><strong>{customerPage.summary.average_watch_hours} h</strong> average viewing</span>
            </div>
            {customerPage.customers.length ? (
              <div className="table-scroll">
                <table className="customer-table">
                  <thead>
                    <tr>
                      <SortHeader label="Customer ID" field="customer_id" active={filters} onSort={sortCustomers} />
                      <th scope="col">Segment</th>
                      <SortHeader label="Plan" field="subscription_type" active={filters} onSort={sortCustomers} />
                      <SortHeader label="Region" field="region" active={filters} onSort={sortCustomers} />
                      <SortHeader label="Device" field="device" active={filters} onSort={sortCustomers} />
                      <SortHeader label="Genre" field="favorite_genre" active={filters} onSort={sortCustomers} />
                      <SortHeader label="Viewing (h)" field="watch_hours" active={filters} onSort={sortCustomers} />
                      <SortHeader label="Last login (days)" field="last_login_days" active={filters} onSort={sortCustomers} />
                      <SortHeader label="Churn label" field="churned" active={filters} onSort={sortCustomers} />
                      <th scope="col"><button type="button" className="sort-header" onClick={() => sortCustomers("risk")} aria-label="Sort by predicted churn risk">Risk</button></th>
                    </tr>
                  </thead>
                  <tbody>
                    {customerPage.customers.map((item) => (
                      <tr key={item.customer_id}>
                        <td><button className="table-link" onClick={() => openCustomer(item.customer_id)}>{item.customer_id}</button></td>
                        <td>{item.segment_name || (item.cluster === null ? "Unassigned" : `Segment ${item.cluster}`)}</td>
                        <td>{item.subscription_type}</td>
                        <td>{item.region}</td>
                        <td>{item.device}</td>
                        <td>{item.favorite_genre}</td>
                        <td>{item.watch_hours}</td>
                        <td>{item.last_login_days}</td>
                        <td>{item.churned ? "Churned" : "Not churned"}</td>
                        <td>{item.churn_risk_percent === null ? "Unavailable" : <span className={`risk-pill ${item.risk_band}`}>{item.churn_risk_percent}% · {item.risk_band}</span>}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : <div className="empty-state"><Users size={22} /><p>No customers match these filters.</p></div>}
            <div className="pagination">
              <span>Page {customerPage.page} of {Math.max(customerPage.pages, 1)} · {customerPage.total.toLocaleString()} matching records</span>
              <div>
                <button className="secondary-button" disabled={customerPage.page <= 1} onClick={() => updateFilter("page", customerPage.page - 1)}>Previous</button>
                <button className="secondary-button" disabled={customerPage.page >= customerPage.pages} onClick={() => updateFilter("page", customerPage.page + 1)}>Next</button>
              </div>
            </div>
          </>
        )}
        {(customerDetailLoading || customerDetailError || selectedCustomer) && (
          <section className="customer-detail-panel">
            {customerDetailLoading && <div className="inline-status" role="status">Loading customer profile…</div>}
            {customerDetailError && <div className="error-message" role="alert">{customerDetailError}</div>}
            {selectedCustomer && <CustomerInsights key={selectedCustomer.customer_id} customer={selectedCustomer} threshold={filters.threshold} />}
          </section>
        )}
      </div>
    );
  };

  const renderSegments = () => {
    if (loading) return renderStatus("Loading segment profiles...");
    if (error) return renderStatus(error, true);
    const toggleSegment = (cluster) => {
      setSelectedSegments((current) =>
        current.includes(cluster)
          ? current.filter((value) => value !== cluster)
          : [...current, cluster],
      );
    };
    const compared = segments.filter((item) => selectedSegments.includes(item.cluster));
    return (
      <div className="segments-workspace">
        <div className="workspace-heading">
          <div>
            <h2>Customer segments</h2>
            <p>Names summarize measured viewing and recency relative to the overall medians.</p>
          </div>
        </div>
        <p className="notice">Segment profiles are descriptive summaries of this snapshot. Differences do not prove causes of churn.</p>
        <div className="segment-grid">
          {segments.map((segment) => (
            <article className="panel segment-card" key={segment.cluster}>
              <label className="compare-check">
                <input type="checkbox" checked={selectedSegments.includes(segment.cluster)} onChange={() => toggleSegment(segment.cluster)} aria-label={`Compare ${segment.name}`} />
                Compare
              </label>
              <div className="segment-heading">
                <div><p className="eyebrow">CLUSTER {segment.cluster}</p><h2>{segment.name}</h2></div>
                <span className="cluster-number">{segment.cluster}</span>
              </div>
              <div className="segment-users"><Users size={16} /><strong>{segment.users.toLocaleString()}</strong><span>customers</span></div>
              <div className="detail-list">
                <div><span>Average viewing</span><strong>{segment.avg_watch_hours} h</strong></div>
                <div><span>Average last login</span><strong>{segment.avg_last_login_days} days ago</strong></div>
                <div><span>Recorded churn rate</span><strong>{segment.churn_rate}%</strong></div>
                <div><span>Most common genre</span><strong>{segment.top_genre}</strong></div>
                <div><span>Plan mix</span><strong>{segment.plan_mix.map((plan) => `${plan.plan} ${plan.percent}%`).join(" · ")}</strong></div>
                <div><span>Genre mix</span><strong>{segment.genre_mix.slice(0, 3).map((genre) => `${genre.genre} ${genre.percent}%`).join(" · ")}</strong></div>
              </div>
              <button className="secondary-button segment-drill" onClick={() => drillDown({ segment: String(segment.cluster) })}>View these customers</button>
            </article>
          ))}
        </div>
        <ChartPanel title="Candidate cluster counts" description="Inertia and silhouette compare candidate k values using preprocessed, identifier-excluded features.">
          {segmentEvaluationLoading && <div className="inline-status">Evaluating candidate values of k…</div>}
          {segmentEvaluationError && <div className="error-message" role="alert">{segmentEvaluationError}</div>}
          {segmentEvaluation?.candidates?.length > 0 && (
            <>
              <ResponsiveContainer width="100%" height={300}>
                <ComposedChart data={segmentEvaluation.candidates}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="k" label={{ value: "Number of segments (k)", position: "insideBottom", offset: -5 }} />
                  <YAxis yAxisId="inertia" />
                  <YAxis yAxisId="silhouette" orientation="right" domain={[-1, 1]} />
                  <Tooltip />
                  <Legend />
                  <Bar yAxisId="inertia" dataKey="inertia" name="Inertia (lower)" fill="#9b87f5" />
                  <Line yAxisId="silhouette" type="monotone" dataKey="silhouette" name="Silhouette (higher)" stroke="#51c6a5" strokeWidth={2} />
                </ComposedChart>
              </ResponsiveContainer>
              <p className="panel-description">{segmentEvaluation.interpretation}</p>
            </>
          )}
        </ChartPanel>
        {compared.length >= 2 && (
          <ChartPanel title={`Comparing ${compared.length} segments`} description="Measured values for the selected existing segment IDs.">
            <div className="table-scroll">
              <table className="customer-table comparison-table">
                <thead><tr><th>Segment</th><th>Customers</th><th>Viewing (h)</th><th>Last login (days)</th><th>Churn rate</th><th>Top genre</th></tr></thead>
                <tbody>{compared.map((item) => <tr key={item.cluster}><td>{item.name} · ID {item.cluster}</td><td>{item.users}</td><td>{item.avg_watch_hours}</td><td>{item.avg_last_login_days}</td><td>{item.churn_rate}%</td><td>{item.top_genre}</td></tr>)}</tbody>
              </table>
            </div>
          </ChartPanel>
        )}
      </div>
    );
  };

  const renderRecommendations = () => {
    if (loading) return renderStatus("Loading genre recommendations...");
    if (error) return renderStatus(error, true);
    if (!recommendations.length) return <div className="empty-state"><Lightbulb size={22} /><p>No genre summary is available.</p></div>;
    return (
      <div className="recommendation-workspace">
        <div className="workspace-heading">
          <div><h2>Genre suggestions</h2><p>A transparent favourite-genre mapping baseline—not a learned recommender.</p></div>
        </div>
        <div className="notice">
          Genre suggestions below use a fixed transparent mapping. The movie
          catalogue is available in customer profiles from
          <code> data/movies_metadata.csv </code>, where titles match the
          customer’s favourite genre. There is no item-interaction history, and
          catalogue metadata does not verify current streaming availability.
        </div>
        <div className="recommendation-grid">
          {recommendations.map((item) => (
            <article className="panel recommendation-card" key={item.genre}>
              <div className="recommendation-title">
                <span className="genre-symbol"><Film size={20} /></span>
                <div><h2>{item.genre}</h2><p>{item.customers.toLocaleString()} customers list this as their favourite</p></div>
              </div>
              <h3>Related genres in fixed mapping</h3>
              <div className="tag-list">{item.recommended_genres.map((genre) => <span className="genre-tag" key={genre}>{genre}</span>)}</div>
              <p className="panel-description">{item.explanation}</p>
            </article>
          ))}
        </div>
      </div>
    );
  };

  const currentPage = {
    Dashboard: renderDashboard,
    Customers: renderCustomers,
    Segments: renderSegments,
    Recommendations: renderRecommendations,
  }[page] || renderDashboard;

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">OTT</div>
          <div><h2>OTT Insights</h2><p>Customer intelligence</p></div>
        </div>
        <nav className="nav-menu" aria-label="Main navigation">
          <NavButton active={page === "Dashboard"} onClick={() => navigate("Dashboard")} icon={<LayoutDashboard size={18} />}>Dashboard</NavButton>
          <NavButton active={page === "Customers"} onClick={() => navigate("Customers")} icon={<Users size={18} />}>Customers</NavButton>
          <NavButton active={page === "Segments"} onClick={() => navigate("Segments")} icon={<BarChart3 size={18} />}>Segments</NavButton>
          <NavButton active={page === "Recommendations"} onClick={() => navigate("Recommendations")} icon={<Lightbulb size={18} />}>Recommendations</NavButton>
        </nav>
        <div className="sidebar-card">
          <p>Data readiness</p>
          <div className={`status-pill ${dashboard?.data_quality?.status === "invalid" ? "offline" : "online"}`}>
            <span className="dot" />{dashboard?.data_quality?.status === "invalid" ? "Needs attention" : "Connected"}
          </div>
          <p className="sidebar-note">{dashboard?.data_quality?.record_count?.toLocaleString() || "—"} customer records</p>
        </div>
      </aside>
      <main className="main-panel">
        <header className="topbar">
          <div><p className="eyebrow">OTT CUSTOMER ANALYTICS</p><h1>{page}</h1></div>
          <div className="topbar-actions">
            <button className="ghost-button" onClick={reloadProjectData} disabled={loading}><RefreshCw size={16} className={loading ? "spin" : ""} />Reload data</button>
          </div>
        </header>
        <section className="search-box">
          <form onSubmit={submitCustomerSearch} className="search-form">
            <label htmlFor="customer-search" className="sr-only">Search customer ID</label>
            <div className="search-input-wrap"><Search size={16} /><input id="customer-search" value={searchCustomerId} onChange={(event) => setSearchCustomerId(event.target.value)} placeholder="Jump to a customer by ID" /></div>
            <button type="submit" className="primary-button" disabled={!searchCustomerId.trim()}>Open customer</button>
          </form>
        </section>
        <section className="content-panel" aria-live="polite">{currentPage()}</section>
        <footer className="app-footer">
          No automated messages or offers are sent. Customer risk is a model estimate; verify data and use human judgment.
        </footer>
      </main>
    </div>
  );
}

function KpiCard({ icon, title, value, detail }) {
  return <article className="stat-card"><div className="stat-top"><span>{title}</span><span className="stat-icon">{icon}</span></div><h2>{value}</h2><p>{detail || "Current dataset snapshot"}</p></article>;
}

function ChartPanel({ title, description, children }) {
  return <article className="panel chart-panel"><h2>{title}</h2><p className="panel-description">{description}</p><div className="chart-area">{children}</div></article>;
}

function EmptyChart({ message }) {
  return <div className="chart-empty">{message}</div>;
}

function DataQualityPanel({ quality }) {
  if (!quality) return null;
  return (
    <section className="panel quality-panel">
      <div><h2>Data quality</h2><p className="panel-description">Validation checks for the loaded customer CSV.</p></div>
      <div className="quality-metrics">
        <span><strong>{quality.record_count.toLocaleString()}</strong> records</span>
        <span><strong>{quality.missing_value_count.toLocaleString()}</strong> missing values</span>
        <span><strong>{quality.duplicate_count.toLocaleString()}</strong> duplicate IDs</span>
        <span className={`quality-status ${quality.status}`}><strong>{quality.status}</strong> status</span>
      </div>
      {quality.errors?.length > 0 && <p className="error-message">{quality.errors.join(" ")}</p>}
      {quality.warnings?.length > 0 && <details><summary>Validation warnings ({quality.warnings.length})</summary><ul>{quality.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul></details>}
    </section>
  );
}

function ModelInfoPanel({ model }) {
  if (!model) return null;
  const evaluations = model.evaluation || {};
  return (
    <section className="panel model-info-panel">
      <div>
        <h2>Churn model</h2>
        <p className="panel-description">
          {model.available
            ? `Version ${model.version || "unversioned"} · trained ${model.trained_at || "date not recorded"}`
            : model.error || "No trained model is available."}
        </p>
      </div>
      {Object.keys(evaluations).length > 0 && (
        <div className="table-scroll">
          <table className="customer-table">
            <caption className="sr-only">Holdout model evaluation comparison</caption>
            <thead><tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>ROC-AUC</th><th>PR-AUC</th></tr></thead>
            <tbody>
              {Object.entries(evaluations).map(([name, item]) => (
                <tr key={name}>
                  <td>{name.replaceAll("_", " ")}</td>
                  <td>{item.metrics.accuracy}</td>
                  <td>{item.metrics.precision}</td>
                  <td>{item.metrics.recall}</td>
                  <td>{item.metrics.f1}</td>
                  <td>{item.metrics.roc_auc}</td>
                  <td>{item.metrics.pr_auc}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <details>
        <summary>Prediction-time input schema ({model.features?.length || 0} features)</summary>
        <div className="tag-list model-features">
          {(model.features || []).map((feature) => <span className="genre-tag" key={feature}>{feature}</span>)}
        </div>
      </details>
      <p className="panel-description">
        Evaluation uses one stratified random holdout. It is not time-based and
        does not establish performance on future or production customers.
      </p>
    </section>
  );
}

function NavButton({ active, onClick, icon, children }) {
  return <button className={`nav-item${active ? " active" : ""}`} onClick={onClick} aria-current={active ? "page" : undefined}>{icon}{children}</button>;
}

function FilterSelect({ label, value, onChange, options }) {
  const id = `filter-${label.toLowerCase().replaceAll(" ", "-")}`;
  return <label className="filter-field" htmlFor={id}><span>{label}</span><select id={id} value={value} onChange={(event) => onChange(event.target.value)}>{options.map(([optionValue, optionLabel]) => <option key={optionValue} value={optionValue}>{optionLabel}</option>)}</select></label>;
}

function FilterText({ label, value, onChange, placeholder }) {
  const id = `filter-${label.toLowerCase().replaceAll(" ", "-")}`;
  return <label className="filter-field" htmlFor={id}><span>{label}</span><input id={id} value={value} onChange={(event) => onChange(event.target.value)} placeholder={placeholder} /></label>;
}

function SortHeader({ label, field, active, onSort }) {
  const selected = active.sort_by === field;
  return <th><button type="button" className="sort-header" onClick={() => onSort(field)}>{label}{selected ? active.order === "asc" ? <ChevronUp size={13} /> : <ChevronDown size={13} /> : null}</button></th>;
}

export default App;
