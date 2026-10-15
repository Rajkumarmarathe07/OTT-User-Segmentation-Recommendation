
import { useCallback, useEffect, useState } from "react";
import axios from "axios";
import {
  LayoutDashboard,
  Users,
  Lightbulb,
  Search,
  RefreshCw,
  Activity,
  Clock,
  UserRound,
  AlertCircle,
} from "lucide-react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from "recharts";
import "./App.css";

const API = "http://127.0.0.1:5000/api";

function App() {
  const [page, setPage] = useState("Dashboard");
  const [dashboard, setDashboard] = useState(null);
  const [segments, setSegments] = useState([]);
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [customerId, setCustomerId] = useState("");
  const [customer, setCustomer] = useState(null);
  const [searchError, setSearchError] = useState("");
  const [searching, setSearching] = useState(false);

  const loadDashboard = useCallback(async () => {
    setLoading(true);
    setError("");

    try {
      const [
        dashboardResponse,
        segmentsResponse,
        recommendationsResponse,
      ] = await Promise.all([
        axios.get(`${API}/dashboard`),
        axios.get(`${API}/segments`),
        axios.get(`${API}/recommendations`),
      ]);

      setDashboard(dashboardResponse.data);
      setSegments(segmentsResponse.data);
      setRecommendations(recommendationsResponse.data);
    } catch (err) {
      setError(
        err.response?.data?.error ||
          "Cannot connect to Flask. Check that python app.py is running on port 5000."
      );
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
      // eslint-disable-next-line react-hooks/set-state-in-effect
    loadDashboard();
  }, [loadDashboard]);

  const searchCustomer = async (event) => {
    event.preventDefault();
    setCustomer(null);
    setSearchError("");

    if (!customerId.trim()) {
      setSearchError("Please enter a customer ID.");
      return;
    }

    setSearching(true);

    try {
      const response = await axios.get(
        `${API}/customer/${encodeURIComponent(customerId.trim())}`
      );
      setCustomer(response.data);
    } catch (err) {
      setSearchError(
        err.response?.status === 404
          ? "Customer not found. Enter the complete customer ID."
          : err.response?.data?.error || "Unable to search for this customer."
      );
    } finally {
      setSearching(false);
    }
  };

  const menuItems = [
    { name: "Dashboard", icon: LayoutDashboard },
    { name: "User Segments", icon: Users },
    { name: "Recommendations", icon: Lightbulb },
    { name: "Customer Search", icon: Search },
  ];

  const renderDashboard = () => {
    if (!dashboard) return null;

    const cards = [
      {
        title: "Total Customers",
        value: dashboard.total_users ?? 0,
        icon: Users,
        note: "Customers in dataset",
      },
      {
        title: "Favorite Genres",
        value: dashboard.total_genres ?? 0,
        icon: Lightbulb,
        note: "Different genres",
      },
      {
        title: "Average Watch Hours",
        value: dashboard.average_watch_hours ?? "N/A",
        icon: Clock,
        note: "Average per customer",
      },
      {
        title: "Churn Rate",
        value:
          dashboard.churn_rate == null
            ? "N/A"
            : `${dashboard.churn_rate}%`,
        icon: Activity,
        note: "Customers marked as churned",
      },
    ];

    return (
      <>
        <div className="page-heading">
          <div>
            <h1>OTT Dashboard</h1>
            <p>Machine learning insights from your customer data.</p>
          </div>

          <button className="secondary-button" onClick={loadDashboard}>
            <RefreshCw size={16} />
            Refresh
          </button>
        </div>

        <div className="stats-grid">
          {cards.map((card) => {
            const Icon = card.icon;

            return (
              <div className="stat-card" key={card.title}>
                <div className="stat-top">
                  <span>{card.title}</span>
                  <span className="stat-icon">
                    <Icon size={20} />
                  </span>
                </div>
                <h2>{card.value}</h2>
                <p>{card.note}</p>
              </div>
            );
          })}
        </div>

        <div className="charts-grid">
          <section className="panel">
            <h2>Customers by Favorite Genre</h2>
            <p className="panel-description">
              Number of customers in each genre
            </p>

            <div className="chart-area">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={dashboard.genres || []}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#303047" />
                  <XAxis
                    dataKey="name"
                    tick={{ fill: "#b8b8ca", fontSize: 11 }}
                    angle={-20}
                    textAnchor="end"
                    height={55}
                  />
                  <YAxis tick={{ fill: "#b8b8ca", fontSize: 12 }} />
                  <Tooltip
                    contentStyle={{
                      background: "#202033",
                      border: "1px solid #41415a",
                      borderRadius: "8px",
                      color: "#fff",
                    }}
                  />
                  <Bar
                    dataKey="users"
                    fill="#9b87f5"
                    radius={[5, 5, 0, 0]}
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </section>

          <section className="panel">
            <h2>Customer Segments</h2>
            <p className="panel-description">
              Customers grouped using K-Means clustering
            </p>

            <div className="chart-area">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={dashboard.clusters || []}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#303047" />
                  <XAxis
                    dataKey="name"
                    tick={{ fill: "#b8b8ca", fontSize: 12 }}
                  />
                  <YAxis tick={{ fill: "#b8b8ca", fontSize: 12 }} />
                  <Tooltip
                    contentStyle={{
                      background: "#202033",
                      border: "1px solid #41415a",
                      borderRadius: "8px",
                      color: "#fff",
                    }}
                  />
                  <Bar
                    dataKey="users"
                    name="Customers"
                    fill="#53c6a2"
                    radius={[5, 5, 0, 0]}
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </section>
        </div>
      </>
    );
  };

  const renderSegments = () => (
    <>
      <div className="page-heading">
        <div>
          <h1>User Segments</h1>
          <p>Understand customer behaviour in each cluster.</p>
        </div>
      </div>

      {segments.length === 0 ? (
        <div className="empty-state">
          No segments available. Run clustering.py first.
        </div>
      ) : (
        <div className="segment-grid">
          {segments.map((segment) => (
            <section className="panel segment-card" key={segment.cluster}>
              <div className="segment-heading">
                <div>
                  <span className="eyebrow">CUSTOMER GROUP</span>
                  <h2>{segment.name}</h2>
                </div>
                <span className="cluster-number">{segment.cluster}</span>
              </div>

              <div className="segment-users">
                <Users size={19} />
                <strong>{segment.users}</strong> customers
              </div>

              <div className="detail-list">
                <div>
                  <span>Average age</span>
                  <strong>{segment.avg_age ?? "N/A"}</strong>
                </div>
                <div>
                  <span>Average watch hours</span>
                  <strong>{segment.avg_watch_hours ?? "N/A"}</strong>
                </div>
                <div>
                  <span>Days since last login</span>
                  <strong>{segment.avg_last_login_days ?? "N/A"}</strong>
                </div>
                <div>
                  <span>Average monthly fee</span>
                  <strong>
                    {segment.avg_monthly_fee == null
                      ? "N/A"
                      : `₹${segment.avg_monthly_fee}`}
                  </strong>
                </div>
                <div>
                  <span>Average daily watch time</span>
                  <strong>{segment.avg_daily_watch_time ?? "N/A"}</strong>
                </div>
                <div>
                  <span>Churn rate</span>
                  <strong>
                    {segment.churn_rate == null
                      ? "N/A"
                      : `${(segment.churn_rate * 100).toFixed(1)}%`}
                  </strong>
                </div>
              </div>

              <div className="genre-tag">
                Most common genre: {segment.top_genre || "Unknown"}
              </div>
            </section>
          ))}
        </div>
      )}
    </>
  );

  const renderRecommendations = () => (
    <>
      <div className="page-heading">
        <div>
          <h1>Genre Recommendations</h1>
          <p>Suggested genres based on customer preferences.</p>
        </div>
      </div>

      <div className="recommendation-grid">
        {recommendations.map((item) => (
          <section className="panel recommendation-card" key={item.genre}>
            <div className="recommendation-title">
              <span className="genre-symbol">
                <Lightbulb size={20} />
              </span>
              <div>
                <h2>{item.genre}</h2>
                <p>{item.customers} customers</p>
              </div>
            </div>

            <h3>Suggested genres</h3>
            <div className="tag-list">
              {(item.recommended_genres || []).map((genre) => (
                <span className="genre-tag" key={genre}>
                  {genre}
                </span>
              ))}
            </div>
          </section>
        ))}
      </div>

      <p className="notice">
        These are genre-based suggestions, not recommendations for specific
        movies. Movie-level viewing history is needed for personalized movie
        recommendations.
      </p>
    </>
  );

  const renderCustomerSearch = () => (
    <>
      <div className="page-heading">
        <div>
          <h1>Customer Search</h1>
          <p>Find a customer using their full customer ID.</p>
        </div>
      </div>

      <section className="panel search-panel">
        <form className="search-form" onSubmit={searchCustomer}>
          <input
            type="text"
            value={customerId}
            onChange={(event) => setCustomerId(event.target.value)}
            placeholder="Enter the full customer UUID"
            aria-label="Customer ID"
          />

          <button
            className="primary-button"
            type="submit"
            disabled={searching}
          >
            <Search size={17} />
            {searching ? "Searching..." : "Search"}
          </button>
        </form>

        {searchError && <p className="error-message">{searchError}</p>}
      </section>

      {customer && (
        <section className="panel customer-panel">
          <div className="customer-title">
            <span className="customer-avatar">
              <UserRound size={24} />
            </span>
            <div>
              <h2>Customer Details</h2>
              <p className="customer-id">{customer.customer_id}</p>
            </div>
          </div>

          <div className="customer-grid">
            {[
              ["Age", customer.age],
              ["Gender", customer.gender],
              ["Subscription", customer.subscription_type],
              ["Favorite genre", customer.favorite_genre],
              ["Watch hours", customer.watch_hours],
              ["Last login (days)", customer.last_login_days],
              ["Region", customer.region],
              ["Device", customer.device],
              ["Monthly fee", customer.monthly_fee],
              ["Payment method", customer.payment_method],
              ["Number of profiles", customer.number_of_profiles],
              ["Cluster", customer.cluster ?? "Not assigned"],
              [
                "Churned",
                customer.churned == null
                  ? "Unknown"
                  : Number(customer.churned) === 1
                    ? "Yes"
                    : "No",
              ],
            ].map(([label, value]) => (
              <div className="customer-field" key={label}>
                <span>{label}</span>
                <strong>{value ?? "N/A"}</strong>
              </div>
            ))}
          </div>

          <div className="customer-recommendations">
            <h3>Recommended genres</h3>
            <div className="tag-list">
              {(customer.recommended_genres || []).map((genre) => (
                <span className="genre-tag" key={genre}>
                  {genre}
                </span>
              ))}
            </div>
          </div>
        </section>
      )}
    </>
  );

  return (
    <div className="app-layout">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-logo">O</div>
          <div>
            <h2>OTT Insight</h2>
            <p>ML Analytics Platform</p>
          </div>
        </div>

        <p className="menu-label">MAIN MENU</p>

        <nav className="navigation">
          {menuItems.map((item) => {
            const Icon = item.icon;

            return (
              <button
                key={item.name}
                className={`nav-item ${page === item.name ? "active" : ""}`}
                onClick={() => setPage(item.name)}
              >
                <Icon size={19} />
                <span>{item.name}</span>
              </button>
            );
          })}
        </nav>

        <div className="sidebar-footer">
          <span className="status-dot" />
          Flask API · Port 5000
          <p>OTT User Segmentation Project</p>
        </div>
      </aside>

      <main className="main-content">
        {error ? (
          <div className="error-panel">
            <AlertCircle size={28} />
            <h2>Could not load dashboard data</h2>
            <p>{error}</p>
            <button className="primary-button" onClick={loadDashboard}>
              <RefreshCw size={16} />
              Try again
            </button>
          </div>
        ) : loading ? (
          <div className="loading-state">
            <RefreshCw size={25} className="spin" />
            <p>Loading your OTT analytics...</p>
          </div>
        ) : (
          <>
            {page === "Dashboard" && renderDashboard()}
            {page === "User Segments" && renderSegments()}
            {page === "Recommendations" && renderRecommendations()}
            {page === "Customer Search" && renderCustomerSearch()}
          </>
        )}
      </main>
    </div>
  );
}

export default App;
