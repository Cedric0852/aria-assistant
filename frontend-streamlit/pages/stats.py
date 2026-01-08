import os

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(page_title="API Stats", page_icon="📊", layout="wide")


def format_time(ms: float) -> str:
    """Format milliseconds to human-readable string."""
    if ms is None or ms == 0:
        return "-"
    if ms >= 1000:
        return f"{ms/1000:.1f}s"
    return f"{ms:.0f}ms"


def get_time_in_seconds(ms: float) -> float:
    """Convert milliseconds to seconds."""
    if ms is None:
        return 0
    return ms / 1000

API_URL = os.environ.get("API_URL", "http://localhost:8000")


@st.cache_data(ttl=10)
def fetch_stats():
    """Fetch stats from the backend API."""
    response = requests.get(f"{API_URL}/api/stats", timeout=10)
    response.raise_for_status()
    return response.json()


st.title("API Stats Dashboard")

if st.button("Refresh"):
    st.cache_data.clear()
    st.rerun()

try:
    stats = fetch_stats()
    summary = stats.get("summary", {})
    endpoints = stats.get("endpoints", {})

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Requests", summary.get("total_requests", 0))

    with col2:
        st.metric("Cache Hit Rate", summary.get("cache_hit_rate", "0%"))

    with col3:
        st.metric("Cache Hits", summary.get("total_cache_hits", 0))

    with col4:
        st.metric("Endpoints Tracked", summary.get("endpoints_tracked", 0))

    st.divider()

    st.subheader("Endpoint Performance")

    if endpoints:
        table_data = []
        for endpoint_name, endpoint_data in endpoints.items():
            endpoint = endpoint_data.get("endpoint", endpoint_name)
            short_name = endpoint.replace("/api/query/", "").replace("?include_audio=", " (audio=")
            if "audio=" in short_name:
                short_name += ")"

            table_data.append(
                {
                    "Endpoint": short_name,
                    "Avg Response": format_time(endpoint_data.get("avg_response_ms", 0)),
                    "First Response": format_time(endpoint_data.get("first_response_ms", 0)),
                    "Cached Response": format_time(endpoint_data.get("cached_response_ms", 0)),
                    "Speedup": endpoint_data.get("improvement", "-"),
                    "Cache Hits": endpoint_data.get("cache_hits", 0),
                    "Total Requests": endpoint_data.get("total_requests", 0),
                }
            )

        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.divider()

        st.subheader("Response Time Comparison")

        endpoints_list = []
        first_times = []
        cached_times = []

        for endpoint_name, endpoint_data in endpoints.items():
            endpoint = endpoint_data.get("endpoint", endpoint_name)
            short_name = endpoint.replace("/api/query/", "").replace("?include_audio=", "\n(audio=")
            if "audio=" in short_name:
                short_name += ")"

            endpoints_list.append(short_name)
            first_times.append(get_time_in_seconds(endpoint_data.get("first_response_ms", 0)))
            cached_times.append(get_time_in_seconds(endpoint_data.get("cached_response_ms", 0)))

        fig = go.Figure()

        fig.add_trace(go.Bar(
            name='First Response',
            x=endpoints_list,
            y=first_times,
            text=[f"{t:.1f}s" if t >= 1 else f"{t*1000:.0f}ms" for t in first_times],
            textposition='outside',
            marker_color='#FF6B6B',
        ))

        fig.add_trace(go.Bar(
            name='Cached Response',
            x=endpoints_list,
            y=cached_times,
            text=[f"{t:.1f}s" if t >= 1 else f"{t*1000:.0f}ms" for t in cached_times],
            textposition='outside',
            marker_color='#4ECDC4',
        ))

        fig.update_layout(
            barmode='group',
            title='First Response vs Cached Response Times',
            yaxis_title='Time (seconds)',
            xaxis_title='Endpoint',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=500,
        )

        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("Cache Performance Summary")

        col1, col2 = st.columns(2)

        with col1:
            best_improvement = None
            best_endpoint = None
            for endpoint_name, endpoint_data in endpoints.items():
                improvement = endpoint_data.get("improvement", "")
                if improvement and "x" in improvement:
                    try:
                        factor = float(improvement.replace("x faster", "").strip())
                        if best_improvement is None or factor > best_improvement:
                            best_improvement = factor
                            best_endpoint = endpoint_data.get("endpoint", endpoint_name).replace("/api/query/", "")
                    except:
                        pass

            if best_improvement:
                st.success(f"**Best Speedup:** {best_improvement:.1f}x faster on `{best_endpoint}`")
            else:
                st.info("No cache speedup data yet")

        with col2:
            total_first = sum(get_time_in_seconds(e.get("first_response_ms", 0)) for e in endpoints.values())
            total_cached = sum(get_time_in_seconds(e.get("cached_response_ms", 0)) for e in endpoints.values())
            count = len(endpoints)

            if count > 0 and total_cached > 0:
                avg_first = total_first / count
                avg_cached = total_cached / count
                st.info(f"**Avg First:** {avg_first:.1f}s | **Avg Cached:** {avg_cached:.2f}s")
            else:
                st.info("Collecting performance data...")

    else:
        st.info("No endpoint data available yet.")

except requests.exceptions.RequestException as e:
    st.error(f"Failed to fetch stats from API: {e}")
except Exception as e:
    st.error(f"An error occurred: {e}")
