import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import os
import calendar
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from streamlit_extras.colored_header import colored_header
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.chart_container import chart_container

# --- Configuration ---
st.set_page_config(
    layout="wide",
    page_title="Movie Genre Performance Analyzer",
    page_icon="🎬",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        color: #1E3A8A !important;
        margin-bottom: 0.5rem !important;
        text-align: center;
    }
    .sub-header {
        font-size: 1.1rem !important;
        color: #6B7280 !important;
        font-style: italic;
        margin-bottom: 2rem !important;
        text-align: center;
    }
    .metric-card {
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        background-color: #f8fafc;
    }
    .metric-label {
        font-size: 0.9rem;
        font-weight: 600;
        color: #4B5563;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1E3A8A;
    }
    .highlight-text {
        background-color: #DBEAFE;
        padding: 0.2rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: 600;
    }
    .tab-content {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f9fafb;
    }
    .small-text {
        font-size: 0.8rem;
        color: #6B7280;
        font-style: italic;
    }
    .section-divider {
        margin-top: 2rem;
        margin-bottom: 2rem;
        border-top: 1px solid #E5E7EB;
    }
    div[data-testid="stDecoration"] {
        background-image: linear-gradient(to right, #1E3A8A, #3B82F6);
        height: 0.3rem;
    }
    div[data-testid="stExpander"] {
        border-radius: 0.5rem;
        border: 1px solid #E5E7EB;
    }
    div[data-testid="stMetric"] {
        background-color: #f9fafb;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
    }
    h1, h2, h3, h4, h5, h6 {
        color: #1E3A8A !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 4px 4px 0px 0px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        background-color: #EFF6FF;
    }
    .stTabs [aria-selected="true"] {
        background-color: #DBEAFE !important;
        color: #1E40AF !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Data Loading Constants ---
DATA_FILE_PARQUET = "output/df_processed_for_streamlit.parquet"
DATA_FILE_CSV = "output/df_processed_for_streamlit.csv"

MIN_SAMPLE_SIZE = 20  # Minimum movies needed in a genre/group for analysis
MIN_SAMPLE_SIZE_ENTITY = 5  # Minimum movies needed for a specific company/country within the genre


# --- Helper functions ---
def format_number(num, prefix=""):
    """Format numbers with appropriate suffixes"""
    if num >= 1_000_000_000:
        return f"{prefix}{num / 1_000_000_000:.1f}B"
    elif num >= 1_000_000:
        return f"{prefix}{num / 1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{prefix}{num / 1_000:.1f}K"
    else:
        return f"{prefix}{num:.2f}"


def get_color_scale(max_items=10):
    """Generate a color scale for charts"""
    return alt.Scale(range=["#1E40AF", "#3B82F6", "#60A5FA", "#93C5FD", "#BFDBFE",
                            "#DBEAFE", "#E0F2FE", "#F0F9FF", "#F8FAFC", "#F9FAFB"][:max_items])


# --- Data Loading (Cached) ---
@st.cache_data
def load_data():
    """Load and prepare the movie data, with progress indicator"""

    # Show loading indicator
    with st.spinner("Loading movie data... This may take a moment."):
        if os.path.exists(DATA_FILE_PARQUET):
            try:
                df = pd.read_parquet(DATA_FILE_PARQUET)
                # Ensure required raw columns exist for display if log versions are used
                for log_col in ['revenue_log', 'roi_log', 'popularity_log', 'vote_count_log', 'budget_log']:
                    raw_col = log_col.replace('_log', '')
                    if log_col in df.columns and raw_col not in df.columns:
                        # Calculate raw from log (approximation for display)
                        if raw_col == 'roi':  # Handle potential shift in ROI log transform
                            df[raw_col] = np.expm1(df[log_col])
                        else:
                            df[raw_col] = np.expm1(df[log_col])
                        st.sidebar.warning(f"Raw column '{raw_col}' calculated from '{log_col}' for display.")
                return df
            except Exception as e:
                st.error(f"Error loading Parquet file: {e}")
                pass

        if os.path.exists(DATA_FILE_CSV):
            try:
                df = pd.read_csv(DATA_FILE_CSV)
                # Add similar logic here to ensure raw columns exist if loading from CSV
                for log_col in ['revenue_log', 'roi_log', 'popularity_log', 'vote_count_log', 'budget_log']:
                    raw_col = log_col.replace('_log', '')
                    if log_col in df.columns and raw_col not in df.columns:
                        if raw_col == 'roi':
                            df[raw_col] = np.expm1(df[log_col])
                        else:
                            df[raw_col] = np.expm1(df[log_col])
                        st.sidebar.warning(f"Raw column '{raw_col}' calculated from '{log_col}' for display.")
                return df
            except Exception as e:
                st.error(f"Error loading CSV file: {e}")
                return None
        else:
            st.error(
                f"Error: Data file not found. Ensure '{DATA_FILE_PARQUET}' or '{DATA_FILE_CSV}' exists in 'output/'.")
            st.stop()
            return None


# --- Helper Function for Entity Analysis ---
def analyze_entity(df_genre, entity_prefix, metric, min_samples, title, genre_name):
    colored_header(label=title, description=f"Analyzing performance by {title.lower()}", color_name="blue-70")

    entity_cols = [col for col in df_genre.columns if col.startswith(entity_prefix)]

    if not entity_cols:
        st.info(f"No {entity_prefix.replace('_', ' ')} data found in the dataset.")
        return

    entity_perf = {}
    overall_mean = df_genre[metric].mean()

    for entity_col in entity_cols:
        # Movies within the genre produced by this entity
        df_entity_subset = df_genre[df_genre[entity_col] == 1]
        count = len(df_entity_subset)

        if count >= min_samples:
            mean_metric_val = df_entity_subset[metric].mean()
            # Get clean entity name
            entity_name = entity_col.replace(entity_prefix, '').replace('_', ' ')
            entity_perf[entity_name] = {'mean': mean_metric_val, 'count': count}

    if entity_perf:
        entity_df = pd.DataFrame(entity_perf).T
        entity_df['diff_from_overall'] = entity_df['mean'] - overall_mean
        entity_df = entity_df.sort_values('mean', ascending=False)

        # Display metric in a custom card
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            st.metric(
                label=f"Overall Avg {metric.replace('_log', '')} for {genre_name}",
                value=f"{overall_mean:.2f}"
            )

        # Display top N entities
        top_n_display = 8

        # Create a more visual table with conditional formatting
        styled_df = entity_df.head(top_n_display).copy()
        styled_df['diff_icon'] = styled_df['diff_from_overall'].apply(
            lambda x: "🔼" if x > 0 else "🔽"
        )
        styled_df['formatted_diff'] = styled_df['diff_from_overall'].apply(
            lambda x: f"{styled_df['diff_icon'].iloc[0]} {abs(x):.2f}"
        )

        display_cols = ['mean', 'count', 'formatted_diff']
        display_names = {
            'mean': f'Avg {metric.replace("_log", "")}',
            'count': 'Movie Count',
            'formatted_diff': 'Diff from Genre Avg'
        }

        with st.container():
            chart_container_entity = chart_container(styled_df.head(top_n_display))

            # Altair Horizontal Bar Chart with improved styling
            plot_df_reset = entity_df.head(top_n_display).reset_index().rename(columns={'index': 'entity_name'})

            # Create color gradient based on performance
            colors = alt.condition(
                alt.datum.mean > overall_mean,
                alt.value("#3B82F6"),  # Blue for above average
                alt.value("#F87171")  # Red for below average
            )

            bars = alt.Chart(plot_df_reset).mark_bar().encode(
                x=alt.X('mean', axis=alt.Axis(title=f'Average {metric.replace("_log", "")}')),
                y=alt.Y('entity_name', axis=alt.Axis(title=title.split(" Performance")[0]), sort='-x'),
                color=colors,
                tooltip=[
                    alt.Tooltip('entity_name', title=title.split(" Performance")[0]),
                    alt.Tooltip('mean', title=f'Avg. {metric.replace("_log", "")}', format='.2f'),
                    alt.Tooltip('count', title='Number of Movies'),
                    alt.Tooltip('diff_from_overall', title='Diff. from Genre Avg.', format='+.2f')
                ]
            ).properties(
                title=f'Top {top_n_display} {title} for {genre_name}',
                height=300
            )

            # Add a reference line for the overall mean
            rule = alt.Chart(pd.DataFrame({'overall_mean': [overall_mean]})).mark_rule(
                color='black',
                strokeDash=[3, 3],
                strokeWidth=1.5
            ).encode(x='overall_mean')

            # Add text annotation for the reference line
            text = alt.Chart(pd.DataFrame({'overall_mean': [overall_mean], 'y': [0]})).mark_text(
                align='left',
                baseline='top',
                dx=5,
                dy=-8,
                fontSize=12,
                fontStyle='italic'
            ).encode(
                x='overall_mean',
                y='y:Q',
                text=alt.value(f"Genre avg: {overall_mean:.2f}")
            )

            # Combine the charts
            final_chart = alt.layer(bars, rule, text).configure_view(
                strokeOpacity=0
            ).configure_axis(
                labelFontSize=12,
                titleFontSize=14,
                titleFontWeight='bold',
                grid=False
            ).configure_title(
                fontSize=16,
                fontWeight='bold',
                anchor='middle'
            )

            st.altair_chart(final_chart, use_container_width=True)

            with st.expander("See detailed performance data"):
                st.dataframe(
                    styled_df[display_cols].rename(columns=display_names),
                    column_config={
                        'Avg {}'.format(metric.replace("_log", "")): st.column_config.NumberColumn(
                            format="%.2f",
                            help=f"Average {metric.replace('_log', '')} for movies in this category"
                        ),
                        'Movie Count': st.column_config.NumberColumn(
                            format="%d",
                            help="Number of movies in the dataset"
                        ),
                        'Diff from Genre Avg': st.column_config.TextColumn(
                            help="Difference from the overall genre average"
                        )
                    },
                    height=300
                )
    else:
        st.info(
            f"Not enough movies per {entity_prefix.replace('_', ' ')} (min {min_samples}) within this genre for reliable analysis.")


# --- Time Analysis Functions ---
def analyze_year_performance(df_genre, metric, metric_name):
    """Analyze and visualize performance by year"""
    year_perf = df_genre.groupby('release_year')[metric].agg(['mean', 'count']).reset_index()
    year_perf = year_perf[year_perf['count'] >= MIN_SAMPLE_SIZE / 4]  # Min count per year

    if not year_perf.empty:
        best_year = year_perf.loc[year_perf['mean'].idxmax(), 'release_year']
        best_year_value = year_perf.loc[year_perf['mean'].idxmax(), 'mean']

        # Use Plotly for a more interactive chart
        fig = go.Figure()

        # Add scatter plot with line
        fig.add_trace(go.Scatter(
            x=year_perf['release_year'],
            y=year_perf['mean'],
            mode='lines+markers',
            marker=dict(
                size=8,
                color=year_perf['mean'],
                colorscale='Blues',
                showscale=True,
                colorbar=dict(title=f"Avg {metric_name}"),
            ),
            line=dict(width=2, color='#3B82F6'),
            hovertemplate="<b>Year:</b> %{x}<br>" +
                          f"<b>Avg {metric_name}:</b> %{{y:.2f}}<br>" +
                          "<b>Movies:</b> %{text}<extra></extra>",
            text=year_perf['count']
        ))

        # Customize layout
        fig.update_layout(
            title=f"<b>{metric_name} Performance by Year</b>",
            xaxis_title="Release Year",
            yaxis_title=f"Average {metric_name}",
            template="plotly_white",
            hovermode="x unified",
            hoverlabel=dict(bgcolor="white", font_size=12),
            height=400,
            margin=dict(l=0, r=0, t=50, b=0),
        )

        st.plotly_chart(fig, use_container_width=True)

        # Add best year metric with styling
        st.metric(
            label=f"Best Year for {metric_name}",
            value=f"{best_year}",
            delta=f"{best_year_value:.2f}"
        )

        return best_year, best_year_value
    else:
        st.info("Not enough data per year for reliable trend analysis.")
        return None, None


def analyze_season_performance(df_genre, metric, metric_name):
    """Analyze and visualize performance by season"""
    season_perf = df_genre.groupby('release_season')[metric].agg(['mean', 'count']).reset_index()
    season_perf = season_perf[season_perf['count'] >= MIN_SAMPLE_SIZE / 4]

    if not season_perf.empty:
        best_season = season_perf.loc[season_perf['mean'].idxmax(), 'release_season']
        best_season_value = season_perf.loc[season_perf['mean'].idxmax(), 'mean']

        # Season colors
        season_colors = {
            'Spring': '#4ADE80',  # Green for Spring
            'Summer': '#F59E0B',  # Amber for Summer
            'Fall': '#B45309',  # Brown for Fall
            'Winter': '#93C5FD'  # Light blue for Winter
        }

        # For bar colors
        colors = [season_colors.get(season, '#3B82F6') for season in season_perf['release_season']]

        # Create Plotly bar chart
        fig = go.Figure()

        # Add bar chart
        fig.add_trace(go.Bar(
            x=season_perf['release_season'],
            y=season_perf['mean'],
            marker_color=colors,
            text=season_perf['count'],
            texttemplate="%{text} movies",
            textposition="outside",
            hovertemplate="<b>Season:</b> %{x}<br>" +
                          f"<b>Avg {metric_name}:</b> %{{y:.2f}}<br>" +
                          "<b>Movies:</b> %{text}<extra></extra>",
        ))

        # Customize layout
        fig.update_layout(
            title=f"<b>{metric_name} Performance by Season</b>",
            xaxis_title="Release Season",
            yaxis_title=f"Average {metric_name}",
            template="plotly_white",
            hoverlabel=dict(bgcolor="white", font_size=12),
            height=400,
            margin=dict(l=0, r=0, t=50, b=0),
            xaxis=dict(
                categoryorder='array',
                categoryarray=['Spring', 'Summer', 'Fall', 'Winter']
            )
        )

        st.plotly_chart(fig, use_container_width=True)

        # Add best season metric
        st.metric(
            label=f"Best Season for {metric_name}",
            value=f"{best_season}",
            delta=f"{best_season_value:.2f}"
        )

        return best_season, best_season_value
    else:
        st.info("Not enough data per season for reliable analysis.")
        return None, None


def analyze_month_performance(df_genre, metric, metric_name):
    """Analyze and visualize performance by month"""
    month_perf = df_genre.groupby('release_month')[metric].agg(['mean', 'count']).reset_index()
    month_perf = month_perf[month_perf['count'] >= MIN_SAMPLE_SIZE / 12]  # Relax min samples for months

    if not month_perf.empty:
        best_month_num = month_perf.loc[month_perf['mean'].idxmax(), 'release_month']
        best_month_name = calendar.month_name[best_month_num]
        best_month_value = month_perf.loc[month_perf['mean'].idxmax(), 'mean']

        # Add month names
        month_perf['month_name'] = month_perf['release_month'].apply(lambda x: calendar.month_name[x])

        # Create a nice multi-color bar chart with Plotly
        fig = go.Figure()

        # Get color gradient based on performance
        month_perf['norm_value'] = (month_perf['mean'] - month_perf['mean'].min()) / (
                    month_perf['mean'].max() - month_perf['mean'].min())

        # Add bar chart with hover info
        fig.add_trace(go.Bar(
            x=month_perf['month_name'],
            y=month_perf['mean'],
            marker=dict(
                color=month_perf['norm_value'],
                colorscale='Blues',
                showscale=False
            ),
            text=month_perf['count'],
            texttemplate="%{text} movies",
            textposition="outside",
            hovertemplate="<b>Month:</b> %{x}<br>" +
                          f"<b>Avg {metric_name}:</b> %{{y:.2f}}<br>" +
                          "<b>Movies:</b> %{text}<extra></extra>",
        ))

        # Customize layout
        fig.update_layout(
            title=f"<b>{metric_name} Performance by Release Month</b>",
            xaxis_title="Release Month",
            yaxis_title=f"Average {metric_name}",
            template="plotly_white",
            hoverlabel=dict(bgcolor="white", font_size=12),
            height=400,
            margin=dict(l=0, r=0, t=50, b=0),
            xaxis=dict(
                categoryorder='array',
                categoryarray=[calendar.month_name[i] for i in range(1, 13)]
            )
        )

        st.plotly_chart(fig, use_container_width=True)

        # Add best month metric
        st.metric(
            label=f"Best Month for {metric_name}",
            value=f"{best_month_name}",
            delta=f"{best_month_value:.2f}"
        )

        return best_month_name, best_month_value
    else:
        st.info("Not enough data per month for reliable analysis.")
        return None, None


# --- Analysis Function ---
def run_analysis(df_processed, genre, metric):
    genre_name = genre.replace('genre_', '')
    metric_name = metric.replace('_log', '')

    # Create a container with styled header
    with st.container():
        colored_header(
            label=f"Analysis Dashboard: {genre_name} Genre",
            description=f"Measuring success by: {metric_name}",
            color_name="blue-100"
        )

        # Movies count and filtering info
        df_genre = df_processed[df_processed[genre] == 1].copy()

        if len(df_genre) < MIN_SAMPLE_SIZE:
            st.warning(f"Only {len(df_genre)} movies found for genre '{genre_name}'. Results may not be reliable.")
            return

        # Handle potential NaNs in the metric column for this subset
        df_genre = df_genre.dropna(subset=[metric])
        if len(df_genre) == 0:
            st.error(f"No valid data points found for metric '{metric_name}' within this genre after dropping NaNs.")
            return

        # Show dataset summary
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.metric(
                label="Total Movies Analyzed",
                value=f"{len(df_genre)}",
                delta=f"{len(df_genre) / len(df_processed) * 100:.1f}% of dataset"
            )
        with col2:
            avg_metric = df_genre[metric].mean()
            overall_avg = df_processed[metric].mean()
            st.metric(
                label=f"Average {metric_name}",
                value=f"{avg_metric:.2f}",
                delta=f"{avg_metric - overall_avg:.2f} vs. all movies" if overall_avg != 0 else None,
                delta_color="normal"
            )
        with col3:
            year_range = f"{df_genre['release_year'].min()} - {df_genre['release_year'].max()}"
            st.metric(
                label="Year Range",
                value=year_range
            )

    # Apply styling to metrics
    style_metric_cards()
    add_vertical_space(1)

    # --- Tabbed Layout with better styling ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Performance Summary",
        "🕒 Timing Analysis",
        "🎭 Co-Genres Impact",
        "💰 Budget Analysis",
        "🏢 Production Insights"
    ])

    with tab1:
        # Performance Summary Dashboard
        st.subheader("Genre Performance At A Glance")

        # Key performance indicators
        with st.container():
            df_top_movies = df_genre.sort_values(metric, ascending=False).head(3)

            # Show the top 3 movies in this genre
            st.markdown("### 🏆 Top Performing Movies")
            for i, (idx, movie) in enumerate(df_top_movies.iterrows()):
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.markdown(f"**#{i + 1}**")
                    st.markdown(f"**{movie.get('title', 'Unknown Title')}** ({movie.get('release_year', 'N/A')})")
                with col2:
                    metrics_cols = st.columns(3)
                    with metrics_cols[0]:
                        raw_metric_val = movie.get(metric.replace('_log', ''), np.expm1(movie.get(metric, 0)))
                        st.markdown(f"**{metric_name}:** {raw_metric_val:.2f}")
                    with metrics_cols[1]:
                        if 'budget' in movie:
                            st.markdown(f"**Budget:** ${movie['budget']:,.0f}")
                        elif 'budget_log' in movie:
                            st.markdown(f"**Budget:** ${np.expm1(movie['budget_log']):,.0f}")
                    with metrics_cols[2]:
                        if 'vote_average' in movie:
                            st.markdown(f"**Rating:** {movie['vote_average']:.1f}/10")

            add_vertical_space(1)

            # Show best timing insights
            st.markdown("### ⏰ Best Release Timing")
            timing_cols = st.columns(3)

            # Pre-calculate best timing to display in summary
            year_perf = df_genre.groupby('release_year')[metric].agg(['mean', 'count'])
            year_perf = year_perf[year_perf['count'] >= MIN_SAMPLE_SIZE / 4]
            if not year_perf.empty:
                best_year = year_perf['mean'].idxmax()
                best_year_value = year_perf.loc[best_year, 'mean']
                with timing_cols[0]:
                    st.metric("Best Year", f"{best_year}", f"{best_year_value:.2f}")

            season_perf = df_genre.groupby('release_season')[metric].agg(['mean', 'count'])
            season_perf = season_perf[season_perf['count'] >= MIN_SAMPLE_SIZE / 4]
            if not season_perf.empty:
                best_season = season_perf['mean'].idxmax()
                best_season_value = season_perf.loc[best_season, 'mean']
                with timing_cols[1]:
                    st.metric("Best Season", f"{best_season}", f"{best_season_value:.2f}")

            month_perf = df_genre.groupby('release_month')[metric].agg(['mean', 'count'])
            month_perf = month_perf[month_perf['count'] >= MIN_SAMPLE_SIZE / 12]
            if not month_perf.empty:
                best_month = month_perf['mean'].idxmax()
                best_month_name = calendar.month_name[best_month]
                best_month_value = month_perf.loc[best_month, 'mean']
                with timing_cols[2]:
                    st.metric("Best Month", f"{best_month_name}", f"{best_month_value:.2f}")

            add_vertical_space(1)

            # Budget insights summary
            st.markdown("### 💵 Budget Insights")
            if 'budget' in df_genre.columns or 'budget_log' in df_genre.columns:
                budget_col_name = 'budget' if 'budget' in df_genre.columns else 'budget_log'
                is_log = budget_col_name.endswith('_log')

                budget_cols = st.columns(3)

                # Calculate budget stats
                with budget_cols[0]:
                    if is_log:
                        median_budget = np.expm1(df_genre[budget_col_name].median())
                    else:
                        median_budget = df_genre[budget_col_name].median()
                    st.metric("Median Budget", f"${median_budget:,.0f}")

                with budget_cols[1]:
                    # Get top performing quartile
                    top_perf_quantile = 0.75
                    top_movies_budget = df_genre[df_genre[metric] >= df_genre[metric].quantile(top_perf_quantile)]

                    if is_log:
                        top_median_budget = np.expm1(top_movies_budget[budget_col_name].median())
                    else:
                        top_median_budget = top_movies_budget[budget_col_name].median()

                    st.metric(
                        "Top Performers Budget",
                        f"${top_median_budget:,.0f}",
                        f"{(top_median_budget / median_budget - 1) * 100:.1f}% vs median" if median_budget > 0 else None
                    )

                with budget_cols[2]:
                    # Calculate ROI estimate if we have revenue data
                    if 'revenue' in df_genre.columns and 'budget' in df_genre.columns:
                        avg_roi = ((df_genre['revenue'] / df_genre['budget']) - 1).mean()
                        st.metric("Avg ROI", f"{avg_roi:.2f}x")
                    elif 'roi' in df_genre.columns:
                        avg_roi = df_genre['roi'].mean()
                        st.metric("Avg ROI", f"{avg_roi:.2f}x")
                    else:
                        st.metric("Budget Range",
                                  f"${df_genre[budget_col_name].min():,.0f} - ${df_genre[budget_col_name].max():,.0f}")

    with tab2:
        # Time Period Analysis with improved visualizations
        colored_header(
            label="Timing Analysis",
            description="When is the best time to release movies in this genre?",
            color_name="blue-70"
        )

        with st.container():
            st.markdown("""
                Understanding the optimal timing for movie releases can significantly impact performance.
                This analysis examines historical patterns by year, season, and month.
            """)

            # Year Analysis
            analyze_year_performance(df_genre, metric, metric_name)

            st.markdown("---")

            # Season and Month in columns
            col1, col2 = st.columns(2)

            with col1:
                analyze_season_performance(df_genre, metric, metric_name)

            with col2:
                analyze_month_performance(df_genre, metric, metric_name)

            st.markdown("---")

            # Add explanation text
            with st.expander("How to interpret timing results"):
                st.markdown("""
                    **Interpreting the timing analysis:**
                    - **Year trends** may indicate changing audience preferences or market conditions
                    - **Seasonal patterns** often reflect audience availability and competition
                    - **Monthly patterns** can reveal more specific release window opportunities

                    Keep in mind that correlation doesn't imply causation. These patterns should be considered
                    alongside other factors like competition, marketing strategies, and current market trends.
                """)

    with tab3:
        # Co-occurring Genre Analysis with better visuals
        colored_header(
            label="Co-Genre Analysis",
            description="Which genre combinations perform best?",
            color_name="blue-70"
        )

        other_genres = [g for g in [col for col in df_processed.columns if col.startswith('genre_')] if g != genre]
        co_genre_perf = {}
        overall_mean_cogenre = df_genre[metric].mean()

        for other_g in other_genres:
            df_co_genre = df_genre[df_genre[other_g] == 1]
            count = len(df_co_genre)
            if count >= MIN_SAMPLE_SIZE / 2:
                mean_metric_val = df_co_genre[metric].mean()
                co_genre_perf[other_g.replace('genre_', '')] = {
                    'mean': mean_metric_val,
                    'count': count,
                    'diff': mean_metric_val - overall_mean_cogenre
                }

        if co_genre_perf:
            co_genre_df = pd.DataFrame(co_genre_perf).T.reset_index()
            co_genre_df.columns = ['co_genre', 'mean', 'count', 'diff']
            co_genre_df = co_genre_df.sort_values('mean', ascending=False)

            st.metric(
                label=f"Overall Avg {metric_name} for {genre_name}",
                value=f"{overall_mean_cogenre:.2f}"
            )

            # Create interactive Plotly chart
            top_n_display_co = 12
            plot_df = co_genre_df.head(top_n_display_co)

            # Color bars based on performance vs. overall
            colors = ['#3B82F6' if val > 0 else '#F87171' for val in plot_df['diff']]

            fig = go.Figure()

            # Add bars
            fig.add_trace(go.Bar(
                x=plot_df['co_genre'],
                y=plot_df['mean'],
                marker_color=colors,
                text=plot_df['count'],
                textposition="outside",
                texttemplate="%{text} movies",
                hovertemplate="<b>Genre:</b> %{x}<br>" +
                              f"<b>Avg {metric_name}:</b> %{{y:.2f}}<br>" +
                              "<b>Movies:</b> %{text}<br>" +
                              "<b>Diff from base genre:</b> %{customdata:.2f}<extra></extra>",
                customdata=plot_df['diff']
            ))

            # Add reference line for genre average
            fig.add_shape(
                type="line",
                x0=-0.5,
                x1=len(plot_df) - 0.5,
                y0=overall_mean_cogenre,
                y1=overall_mean_cogenre,
                line=dict(color="black", width=2, dash="dash")
            )

            # Add annotation for reference line
            fig.add_annotation(
                x=len(plot_df) - 1,
                y=overall_mean_cogenre + 0.05,
                text=f"Genre average: {overall_mean_cogenre:.2f}",
                showarrow=False,
                font=dict(color="black")
            )

            # Customize layout
            fig.update_layout(
                title=f"<b>Impact of Co-Genres on {metric_name} Performance</b>",
                xaxis_title="Combined Genre",
                yaxis_title=f"Average {metric_name}",
                template="plotly_white",
                hoverlabel=dict(bgcolor="white", font_size=12),
                height=500,
                margin=dict(l=10, r=10, t=50, b=10),
                xaxis=dict(tickangle=45)
            )

            st.plotly_chart(fig, use_container_width=True)

            with st.expander("See detailed co-genre data"):
                # Format for display
                display_df = plot_df.copy()
                display_df['impact'] = display_df['diff'].apply(
                    lambda x: f"{'🔼' if x > 0 else '🔽'} {abs(x):.2f}"
                )

                st.dataframe(
                    display_df[['co_genre', 'mean', 'count', 'impact']],
                    column_config={
                        'co_genre': st.column_config.TextColumn("Co-Genre"),
                        'mean': st.column_config.NumberColumn(f"Avg {metric_name}", format="%.2f"),
                        'count': st.column_config.NumberColumn("Movies", format="%d"),
                        'impact': st.column_config.TextColumn("Impact vs. Base Genre")
                    },
                    height=400
                )

            # Insights section
            st.subheader("Key Insights")

            # Best combinations
            best_combos = co_genre_df.head(3)['co_genre'].tolist()
            worst_combos = co_genre_df.tail(3)['co_genre'].tolist()

            insight_cols = st.columns(2)
            with insight_cols[0]:
                st.markdown("#### Best Genre Combinations")
                st.markdown(f"Combining **{genre_name}** with these genres tends to improve performance:")
                for i, genre_combo in enumerate(best_combos):
                    st.markdown(f"{i + 1}. **{genre_combo}** (+{co_genre_perf[genre_combo]['diff']:.2f})")

            with insight_cols[1]:
                st.markdown("#### Less Effective Combinations")
                st.markdown(f"These genres may reduce {metric_name} when combined with **{genre_name}**:")
                for i, genre_combo in enumerate(worst_combos):
                    st.markdown(f"{i + 1}. **{genre_combo}** ({co_genre_perf[genre_combo]['diff']:.2f})")

        else:
            st.info("Not enough data with co-occurring genres to provide reliable analysis.")

    with tab4:
        # Budget Range Analysis with better visualization
        colored_header(
            label="Budget Analysis",
            description=f"How does budget affect {metric_name} in the {genre_name} genre?",
            color_name="blue-70"
        )

        budget_col_name = 'budget_log'
        raw_budget_col = 'budget'

        if budget_col_name not in df_genre.columns:
            st.error(f"Budget column '{budget_col_name}' not found.")
        else:
            try:
                # Bin budget (using quantiles on log-budget)
                num_bins = 5
                df_genre['budget_bin'], bin_edges = pd.qcut(
                    df_genre[budget_col_name],
                    q=num_bins,
                    labels=False,
                    retbins=True,
                    duplicates='drop'
                )

                budget_perf = df_genre.groupby('budget_bin').agg({
                    metric: ['mean', 'count'],
                    budget_col_name: ['min', 'max', 'mean']
                })

                # Flatten multi-index columns
                budget_perf.columns = [f'{col[0]}_{col[1]}' for col in budget_perf.columns]
                budget_perf = budget_perf.reset_index()

                # Convert log budget to raw for display
                budget_perf['min_budget_raw'] = np.expm1(budget_perf[f'{budget_col_name}_min'])
                budget_perf['max_budget_raw'] = np.expm1(budget_perf[f'{budget_col_name}_max'])
                budget_perf['mean_budget_raw'] = np.expm1(budget_perf[f'{budget_col_name}_mean'])

                # Add budget range label for display
                budget_perf['budget_range'] = budget_perf.apply(
                    lambda x: f"${x['min_budget_raw']:,.0f} - ${x['max_budget_raw']:,.0f}",
                    axis=1
                )

                if not budget_perf.empty and len(budget_perf) > 1:
                    best_budget_bin_idx = budget_perf[f'{metric}_mean'].idxmax()
                    best_budget_bin = budget_perf.loc[best_budget_bin_idx, 'budget_bin']

                    # Create more informative visualization with Plotly
                    fig = go.Figure()

                    # Add bar chart
                    fig.add_trace(go.Bar(
                        x=budget_perf['budget_range'],
                        y=budget_perf[f'{metric}_mean'],
                        marker=dict(
                            color=budget_perf[f'{metric}_mean'],
                            colorscale='Blues',
                            showscale=False
                        ),
                        text=budget_perf[f'{metric}_count'],
                        textposition="outside",
                        texttemplate="%{text} movies",
                        hovertemplate="<b>Budget Range:</b> %{x}<br>" +
                                      f"<b>Avg {metric_name}:</b> %{{y:.2f}}<br>" +
                                      "<b>Movies:</b> %{text}<extra></extra>"
                    ))

                    # Add reference points
                    for i, row in budget_perf.iterrows():
                        fig.add_annotation(
                            x=i,
                            y=row[f'{metric}_mean'] + 0.05,
                            text=f"{row[f'{metric}_mean']:.2f}",
                            showarrow=False,
                            font=dict(color="black", size=12)
                        )

                    # Customize layout
                    fig.update_layout(
                        title=f"<b>{metric_name} Performance by Budget Range</b>",
                        xaxis_title="Budget Range",
                        yaxis_title=f"Average {metric_name}",
                        template="plotly_white",
                        hoverlabel=dict(bgcolor="white", font_size=12),
                        height=500,
                        margin=dict(l=10, r=10, t=50, b=10),
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Highlight the best budget bin
                    best_bin_data = budget_perf.loc[best_budget_bin_idx]

                    st.metric(
                        label=f"Optimal Budget Range for {genre_name} Genre",
                        value=best_bin_data['budget_range'],
                        delta=f"Avg {metric_name}: {best_bin_data[f'{metric}_mean']:.2f}"
                    )

                    # Additional budget insights
                    with st.expander("More budget insights"):
                        # Calculate correlation
                        corr = df_genre[[budget_col_name, metric]].corr().iloc[0, 1]

                        st.markdown(f"**Budget to {metric_name} Correlation:** {corr:.3f}")

                        if abs(corr) > 0.5:
                            st.markdown(
                                f"There's a **strong {'positive' if corr > 0 else 'negative'}** correlation between budget and {metric_name}.")
                        elif abs(corr) > 0.3:
                            st.markdown(
                                f"There's a **moderate {'positive' if corr > 0 else 'negative'}** correlation between budget and {metric_name}.")
                        else:
                            st.markdown(
                                f"There's a **weak {'positive' if corr > 0 else 'negative'}** correlation between budget and {metric_name}.")

                        # Show ROI by budget bin if we have ROI data
                        if 'roi' in df_genre.columns or 'roi_log' in df_genre.columns:
                            roi_col = 'roi' if 'roi' in df_genre.columns else 'roi_log'
                            roi_raw = roi_col == 'roi'

                            roi_by_budget = df_genre.groupby('budget_bin')[roi_col].mean().reset_index()

                            if not roi_raw:
                                roi_by_budget['roi_raw'] = np.expm1(roi_by_budget[roi_col])
                            else:
                                roi_by_budget['roi_raw'] = roi_by_budget[roi_col]

                            # Add budget range
                            roi_by_budget = roi_by_budget.merge(
                                budget_perf[['budget_bin', 'budget_range']],
                                on='budget_bin'
                            )

                            st.markdown("#### ROI by Budget Range")

                            # Create ROI by budget chart
                            fig_roi = go.Figure()

                            # Add bar chart
                            fig_roi.add_trace(go.Bar(
                                x=roi_by_budget['budget_range'],
                                y=roi_by_budget['roi_raw'],
                                marker=dict(
                                    color=roi_by_budget['roi_raw'],
                                    colorscale='RdBu',
                                    showscale=False
                                ),
                                hovertemplate="<b>Budget Range:</b> %{x}<br>" +
                                              "<b>Avg ROI:</b> %{y:.2f}x<extra></extra>"
                            ))

                            # Customize layout
                            fig_roi.update_layout(
                                title="<b>Return on Investment by Budget Range</b>",
                                xaxis_title="Budget Range",
                                yaxis_title="Average ROI",
                                template="plotly_white",
                                hoverlabel=dict(bgcolor="white", font_size=12),
                                height=400,
                                margin=dict(l=10, r=10, t=50, b=10),
                            )

                            st.plotly_chart(fig_roi, use_container_width=True)

                    # Show budget stats for top performing movies
                    top_perf_quantile = 0.75
                    top_movies_budget = df_genre[df_genre[metric] >= df_genre[metric].quantile(top_perf_quantile)]

                    if not top_movies_budget.empty and raw_budget_col in top_movies_budget.columns:
                        median_budget_top = top_movies_budget[raw_budget_col].median()
                        iqr_budget_top = top_movies_budget[raw_budget_col].quantile([0.25, 0.75])

                        st.markdown(f"### Budget Profile: Top {(1 - top_perf_quantile) * 100:.0f}% Performing Movies")

                        stats_cols = st.columns(3)
                        with stats_cols[0]:
                            st.metric(
                                label="Median Budget",
                                value=f"${median_budget_top:,.0f}"
                            )
                        with stats_cols[1]:
                            st.metric(
                                label="25th Percentile",
                                value=f"${iqr_budget_top.iloc[0]:,.0f}"
                            )
                        with stats_cols[2]:
                            st.metric(
                                label="75th Percentile",
                                value=f"${iqr_budget_top.iloc[1]:,.0f}"
                            )

                        # Compare to overall genre
                        overall_median = df_genre[raw_budget_col].median()
                        st.markdown(
                            f"Top performing {genre_name} movies have a **{median_budget_top / overall_median:.1f}x** {'higher' if median_budget_top > overall_median else 'lower'} median budget compared to all {genre_name} movies.")

                else:
                    st.info("Not enough data per budget bin for reliable analysis.")

            except Exception as e:
                st.error(f"Error during budget analysis: {e}")
                st.exception(e)

    with tab5:
        # Entity Analysis with improved visualization
        analyze_entity(df_genre, "production_companies_", metric, MIN_SAMPLE_SIZE_ENTITY, "Production Companies",
                       genre_name)

        st.markdown("---")

        analyze_entity(df_genre, "production_countries_", metric, MIN_SAMPLE_SIZE_ENTITY, "Production Countries",
                       genre_name)

        st.markdown("---")

        # Top Movies Table with better formatting
        colored_header(
            label=f"Top {genre_name} Movies",
            description=f"Highest {metric_name} performers in this genre",
            color_name="blue-70"
        )

        top_n_movies = 15
        df_top_movies = df_genre.sort_values(metric, ascending=False).head(top_n_movies)

        # Select columns to display - include raw metric if log version was used
        display_cols = ['title', 'release_year']
        if metric.endswith('_log'):
            raw_metric_col = metric.replace('_log', '')
            if raw_metric_col in df_top_movies.columns:
                display_cols.extend([raw_metric_col])
            else:
                display_cols.append(metric)  # Only show log if raw isn't available
        else:
            display_cols.append(metric)  # Show the metric itself if not log

        # Add raw budget for context
        if raw_budget_col in df_top_movies.columns:
            display_cols.append(raw_budget_col)
        elif budget_col_name in df_top_movies.columns:  # Fallback to log budget
            display_cols.append(budget_col_name)

        # Add vote average if available
        if 'vote_average' in df_top_movies.columns:
            display_cols.append('vote_average')

        # Ensure all display columns exist
        display_cols = [col for col in display_cols if col in df_top_movies.columns]

        # Create a nice formatted dataframe
        df_display = df_top_movies[display_cols].copy()

        # Rename columns for display
        rename_dict = {
            'title': 'Movie Title',
            'release_year': 'Year',
            metric: f'{metric_name} (log)',
            raw_metric_col if metric.endswith('_log') else metric: metric_name,
            raw_budget_col: 'Budget',
            budget_col_name: 'Budget (log)',
            'vote_average': 'Rating'
        }

        # Filter rename_dict to only include columns that exist
        rename_dict = {k: v for k, v in rename_dict.items() if k in df_display.columns}

        # Create column configs for nicer formatting
        column_config = {
            'Movie Title': st.column_config.TextColumn(
                "Movie Title",
                help="The title of the movie"
            ),
            'Year': st.column_config.NumberColumn(
                "Release Year",
                format="%d",
                help="Year the movie was released"
            ),
            metric_name: st.column_config.NumberColumn(
                metric_name,
                format="%.2f",
                help=f"The {metric_name} metric value"
            ),
            f'{metric_name} (log)': st.column_config.NumberColumn(
                f"{metric_name} (log)",
                format="%.2f",
                help=f"Log-transformed {metric_name} value"
            ),
            'Budget': st.column_config.NumberColumn(
                "Budget ($)",
                format="$%d",
                help="Movie production budget in USD"
            ),
            'Budget (log)': st.column_config.NumberColumn(
                "Budget (log)",
                format="%.2f",
                help="Log-transformed budget value"
            ),
            'Rating': st.column_config.NumberColumn(
                "Rating",
                format="%.1f",
                help="Average user rating out of 10"
            )
        }

        # Filter column_config to only include columns that exist
        column_config = {k: v for k, v in column_config.items() if k in rename_dict.values()}

        # Display the formatted dataframe
        st.dataframe(
            df_display.rename(columns=rename_dict),
            column_config=column_config,
            height=400
        )

        with st.expander("Download top movies data"):
            # Add CSV download button
            csv = df_display.rename(columns=rename_dict).to_csv(index=False)
            st.download_button(
                label=f"Download Top {genre_name} Movies Data",
                data=csv,
                file_name=f"top_{genre_name.lower().replace(' ', '_')}_movies.csv",
                mime="text/csv"
            )

    # Final insights and recommendations
    st.markdown("---")


# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/movie.png", width=80)
    st.title("Analysis Controls")

    st.markdown("---")

    # Load Data
    df_processed = load_data()

    # Extract available genres/metrics after loading
    if df_processed is not None:
        available_genres = sorted([col for col in df_processed.columns if col.startswith('genre_')])

        # Create more user-friendly metric names
        metric_options = {
            'revenue_log': 'Revenue',
            'roi_log': 'Return on Investment',
            'popularity_log': 'Popularity',
            'vote_count_log': 'Vote Count',
            'vote_average': 'Average Rating'
        }

        # Only include metrics that are in the dataframe
        available_metrics = sorted([
            metric for metric in metric_options.keys()
            if metric in df_processed.columns
        ])

        budget_col = 'budget_log'
        raw_budget_col = 'budget'  # Used for display

        # Genre selector with better formatting
        st.subheader("1. Select Genre")
        selected_genre = st.selectbox(
            "Choose the primary genre to analyze:",
            options=available_genres,
            format_func=lambda x: x.replace('genre_', ''),
            index=0,
            help="Select the main genre you want to analyze"
        )

        # Metric selector with better formatting
        st.subheader("2. Select Success Metric")
        selected_metric = st.selectbox(
            "Choose how to measure success:",
            options=available_metrics,
            format_func=lambda x: metric_options.get(x, x),
            index=0,
            help="Select the metric that defines success for your analysis"
        )

        st.markdown("---")

        # Add some explanation about the metrics
        with st.expander("About the Metrics"):
            st.markdown("""
            - **Revenue**: Box office performance in dollars
            - **ROI**: Return on Investment (Revenue/Budget - 1)
            - **Popularity**: General audience interest and engagement
            - **Vote Count**: Number of ratings received
            - **Average Rating**: Quality score from 1-10

            *Note: Most metrics use log-transformation to normalize distribution.*
            """)

        # Dataset info
        with st.expander("Dataset Information"):
            st.markdown(f"""
            - **Total Movies**: {len(df_processed)}
            - **Year Range**: {df_processed['release_year'].min()} - {df_processed['release_year'].max()}
            - **Genres**: {len(available_genres)}
            """)

    else:
        st.error("Failed to load dataset. Please check file paths.")

# --- Main App Content ---
if 'df_processed' in locals() and df_processed is not None:
    # App Header
    # st.markdown('<h1 class="main-header">🎬 Movie Genre Performance Analyzer</h1>', unsafe_allow_html=True)
    # st.markdown('<p class="sub-header">Unlock data-driven insights for optimizing movie production decisions</p>',
    #             unsafe_allow_html=True)

    # Run analysis button from sidebar or when variables are set
    if ('analyze_button' in locals()) or (
            'selected_genre' in locals() and 'selected_metric' in locals()):
        if selected_genre and selected_metric:
            run_analysis(df_processed, selected_genre, selected_metric)
        else:
            st.info("Please select both a genre and a metric in the sidebar to begin analysis.")