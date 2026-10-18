import { useEffect, useMemo, useState } from "react";
import axios from "axios";
import { Film, ShieldAlert, UserRound } from "lucide-react";

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:5000/api";
const PROFILE_FIELDS = [
  ["Age", "age"],
  ["Gender", "gender"],
  ["Subscription", "subscription_type"],
  ["Region", "region"],
  ["Device", "device"],
  ["Payment method", "payment_method"],
  ["Profiles", "number_of_profiles"],
  ["Favourite genre", "favorite_genre"],
];

function CustomerInsights({ customer, threshold }) {
  const [churn, setChurn] = useState(null);
  const [movies, setMovies] = useState(null);
  const [loading, setLoading] = useState(true);
  const [movieLoading, setMovieLoading] = useState(true);
  const [churnError, setChurnError] = useState("");
  const [movieError, setMovieError] = useState("");
  const [churnResultThreshold, setChurnResultThreshold] = useState(null);
  const [movieResultQuery, setMovieResultQuery] = useState(null);
  const [supportedFilters, setSupportedFilters] = useState([]);
  const [filterOptions, setFilterOptions] = useState({});
  const [catalogFilters, setCatalogFilters] = useState({});
  const [editableSuggestions, setEditableSuggestions] = useState(
    (customer.retention_suggestions || []).join("\n"),
  );

  const catalogQuery = useMemo(() => {
    const params = new URLSearchParams();
    for (const [key, value] of Object.entries(catalogFilters)) {
      if (value) params.set(key, value);
    }
    return params.toString();
  }, [catalogFilters]);

  useEffect(() => {
    let active = true;
    axios
      .get(
        `${API_URL}/churn/${encodeURIComponent(customer.customer_id)}?threshold=${threshold}`,
      )
      .then((response) => {
        if (active) {
          setChurn(response.data);
          setChurnError("");
          setChurnResultThreshold(threshold);
        }
      })
      .catch((requestError) => {
        if (active) {
          setChurnResultThreshold(threshold);
          setChurnError(
            requestError.response?.data?.error ||
              "Could not load the churn estimate.",
          );
        }
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [customer.customer_id, threshold]);

  useEffect(() => {
    let active = true;
    const query = catalogQuery ? `?${catalogQuery}` : "";
    axios
      .get(
        `${API_URL}/movie-recommendations/${encodeURIComponent(customer.customer_id)}${query}`,
      )
      .then((response) => {
        if (!active) return;
        setMovies(response.data);
        setMovieError("");
        setMovieResultQuery(catalogQuery);
        setSupportedFilters(response.data.supported_filters || []);
        setFilterOptions(response.data.filter_options || {});
      })
      .catch((requestError) => {
        if (active) {
          setMovieResultQuery(catalogQuery);
          setMovieError(
            requestError.response?.data?.error ||
              "Catalogue recommendations could not be loaded.",
          );
        }
      })
      .finally(() => {
        if (active) setMovieLoading(false);
      });
    return () => {
      active = false;
    };
  }, [customer.customer_id, catalogQuery]);

  return (
    <section className="customer-insights">
      <header className="customer-detail-heading">
        <div className="customer-avatar"><UserRound size={22} /></div>
        <div>
          <p className="eyebrow">CUSTOMER PROFILE</p>
          <h2>{customer.customer_id}</h2>
          <p className="customer-id">
            {customer.segment_name || "No segment assignment"}
            {customer.cluster !== null && customer.cluster !== undefined
              ? ` · Cluster ${customer.cluster}`
              : ""}
          </p>
        </div>
      </header>

      <div className="customer-detail-grid">
        <article className="insights-card">
          <h3><UserRound size={18} /> Profile</h3>
          <div className="customer-grid">
            {PROFILE_FIELDS.map(([label, field]) => (
              <div className="customer-field" key={field}>
                <span>{label}</span><strong>{customer[field] ?? "Not available"}</strong>
              </div>
            ))}
          </div>
        </article>
        <article className="insights-card">
          <h3><Film size={18} /> Activity indicators</h3>
          <div className="detail-list">
            <div><span>Cumulative viewing</span><strong>{customer.watch_hours} hours</strong></div>
            <div><span>Average daily viewing</span><strong>{customer.avg_watch_time_per_day} hours</strong></div>
            <div><span>Days since last login</span><strong>{customer.last_login_days} days</strong></div>
            <div><span>Recorded churn label</span><strong>{customer.churned ? "Churned" : "Not marked churned"}</strong></div>
          </div>
        </article>
      </div>

      <article className="insights-card">
        <h3><ShieldAlert size={18} /> Churn risk estimate</h3>
        {(loading || churnResultThreshold !== threshold) && <p role="status">Loading risk estimate…</p>}
        {churnError && churnResultThreshold === threshold && <p className="insights-error" role="alert">{churnError}</p>}
        {churn && churnResultThreshold === threshold && (
          <>
            <div className="risk-summary">
              <strong className={`risk-pill ${churn.risk_band}`}>{churn.churn_risk_percent}% · {churn.risk_band} risk</strong>
              <span>Threshold {churn.threshold.toFixed(2)} → {churn.status}</span>
              <span>Model {churn.model_version || "unversioned"}</span>
            </div>
            <p className="insights-note">{churn.caveat} Lowering the threshold generally increases false positives; raising it generally increases false negatives.</p>
            {churn.explanation?.features?.length > 0 && (
              <div className="importance-list">
                <strong>Largest global model predictors (not customer-specific explanations)</strong>
                {churn.explanation.features.slice(0, 5).map((item) => (
                  <div key={item.feature}><span>{item.feature.replaceAll("_", " ")}</span><span>{(item.importance * 100).toFixed(1)}%</span></div>
                ))}
              </div>
            )}
          </>
        )}
      </article>

      <article className="insights-card">
        <h3><ShieldAlert size={18} /> Retention ideas · informational only</h3>
        <p className="panel-description">Suggestions are triggered by observed viewing or login percentiles. Edit them for your own notes; nothing is sent or applied.</p>
        <label className="suggestions-label" htmlFor="retention-suggestions">Editable suggestions</label>
        <textarea id="retention-suggestions" rows="4" value={editableSuggestions} onChange={(event) => setEditableSuggestions(event.target.value)} />
      </article>

      <article className="insights-card">
        <h3><Film size={18} /> Genre suggestions · transparent baseline</h3>
        <p>Based on favourite genre: <strong>{customer.favorite_genre}</strong></p>
        <div className="tag-list">
          {(customer.recommended_genres || []).map((genre) => <span className="genre-tag" key={genre}>{genre}</span>)}
        </div>
        <p className="insights-note">These are fixed genre suggestions, not movie titles and not a model learned from viewing interactions.</p>
      </article>

      <article className="insights-card">
        <h3><Film size={18} /> Movie recommendations from catalogue</h3>
        <p className="panel-description">
          Titles are selected from {movies?.catalogue_source || "the local movie catalogue"} when a catalogue genre matches this customer’s favourite genre.
          The catalogue does not confirm current streaming availability.
        </p>
        {supportedFilters.length > 0 && (
          <div className="catalog-filter-row">
            {supportedFilters.map((filter) => (
              <label className="filter-field" key={filter}>
                <span>{filter.replaceAll("_", " ")}</span>
                <select value={catalogFilters[filter] || ""} onChange={(event) => setCatalogFilters((current) => ({ ...current, [filter]: event.target.value }))}>
                  <option value="">All</option>
                  {(filterOptions[filter] || []).map((option) => <option key={option} value={option}>{option}</option>)}
                </select>
              </label>
            ))}
          </div>
        )}
        {(movieLoading || movieResultQuery !== catalogQuery) && <p role="status">Checking the available catalogue…</p>}
        {movieError && movieResultQuery === catalogQuery && <p className="insights-error" role="alert">{movieError}</p>}
        {movies?.available && movies.recommendations?.length > 0 && (
          <div className="movie-list">
            {movies.recommendations.map((movie, index) => (
              <article className="movie-item" key={`${movie.title}-${index}`}>
                <h4>{movie.title}</h4>
                <p><strong>Genre:</strong> {movie.genre}</p>
                {movie.description && <p>{movie.description}</p>}
                {Object.entries(movie)
                  .filter(([field, value]) => !["title", "genre", "description"].includes(field) && value !== "")
                  .map(([field, value]) => (
                    <p key={field}><strong>{field.replaceAll("_", " ")}:</strong> {String(value)}</p>
                  ))}
                <p className="panel-description">{movies.explanation}</p>
              </article>
            ))}
          </div>
        )}
        {!movieLoading && movieResultQuery === catalogQuery && !movieError && movies && movies.recommendations?.length === 0 && (
          <div className="catalog-empty" role="status">
            {movies.message || "No catalogue titles match this customer's favourite genre and selected filters."}
          </div>
        )}
      </article>
    </section>
  );
}

export default CustomerInsights;
