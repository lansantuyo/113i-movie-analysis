import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import calendar
import os
from datetime import datetime
import importlib

# --- Must be first Streamlit command ---
st.set_page_config(layout="wide", page_title="Movie Performance Analysis",
                   page_icon="🎬", initial_sidebar_state="expanded")

# Try to import streamlit_extras; if not available, create placeholder functions
try:
    from streamlit_extras.colored_header import colored_header
    from streamlit_extras.metric_cards import style_metric_cards
    from streamlit_extras.add_vertical_space import add_vertical_space
    from streamlit_extras.chart_container import chart_container
except ImportError:
    # Create placeholder functions to avoid errors
    def colored_header(label, description, color_name="blue-70"):
        st.header(label)
        st.write(description)


    def style_metric_cards():
        pass


    def add_vertical_space(n=1):
        for _ in range(n):
            st.write("")


    def chart_container(data):
        return st.container()

# --- Configuration ---
DATA_FILE = "output/test_predictions.parquet"
MIN_SAMPLE_SIZE = 5  # Minimum movies needed for aggregation

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

    /* Time period indicators */
    .period-recent {
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 0.2rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: 600;
        display: inline-block;
    }
    .period-older {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 0.2rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: 600;
        display: inline-block;
    }

    /* Card styling */
    .movie-card {
        background-color: white;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        padding: 1rem;
        margin-bottom: 1rem;
        transition: transform 0.2s ease-in-out;
    }
    .movie-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    .movie-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
    }
    .movie-info {
        font-size: 0.9rem;
        color: #4B5563;
    }
    .movie-metric {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1E3A8A;
    }

    /* Filter sidebar styling */
    .filter-section {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .filter-title {
        font-weight: 600;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
    }

    /* Time period comparison styling */
    .time-period1 {
        color: #1E40AF;
        background-color: #DBEAFE;
    }
    .time-period2 {
        color: #92400E;
        background-color: #FEF3C7;
    }
</style>
""", unsafe_allow_html=True)


# --- Helper Functions ---
def format_value(value, metric):
    """Format values based on metric type"""
    if value is None:
        return "N/A"

    if metric == 'revenue':
        if value >= 1_000_000_000:
            return f"${value / 1_000_000_000:.1f}B"
        elif value >= 1_000_000:
            return f"${value / 1_000_000:.1f}M"
        else:
            return f"${value:,.0f}"
    elif metric == 'roi':
        return f"{value:.2f}x"
    elif metric in ['popularity', 'vote_average']:
        return f"{value:.1f}"
    else:
        return f"{value:.2f}"


def get_performance_color(value, df, metric_col):
    """Get color based on percentile of value within dataset"""
    if df is None or df.empty or metric_col not in df.columns:
        return "#3B82F6"  # Default blue

    percentile = (df[metric_col] <= value).mean() * 100

    if percentile > 90:
        return "#16A34A"  # Green
    elif percentile > 75:
        return "#22C55E"  # Light green
    elif percentile > 50:
        return "#3B82F6"  # Blue
    elif percentile > 25:
        return "#F97316"  # Orange
    else:
        return "#EF4444"  # Red


# --- Data Loading (Cached) ---
@st.cache_data
def load_data():
    """Load movie data with visual loading state"""
    with st.spinner("Loading movie data... Please wait."):
        try:
            # Try to load the parquet file if it exists
            if os.path.exists(DATA_FILE):
                df = pd.read_parquet(DATA_FILE)
                return df
            else:
                st.error(f"File not found: {DATA_FILE}")
                return None
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None


# --- Data Processing Functions ---
def get_production_companies(row):
    """Extract production companies from a row"""
    companies = []
    for col in row.index:
        if col.startswith('production_companies_') and row[col] == 1:
            companies.append(col.replace('production_companies_', '').replace('_', ' '))
    return ', '.join(companies) if companies else '—'


def get_genres(row):
    """Extract genres from a row"""
    genres = []
    for col in row.index:
        if col.startswith('genre_') and row[col] == 1:
            genres.append(col.replace('genre_', ''))
    return ', '.join(genres) if genres else '—'


def extract_all_genres(df):
    """Extract all unique genres from the dataframe"""
    if df is None or df.empty:
        return []

    genre_cols = [col for col in df.columns if col.startswith('genre_')]
    genres = [col.replace('genre_', '') for col in genre_cols]
    return genres


def extract_all_companies(df):
    """Extract all unique companies from the dataframe"""
    if df is None or df.empty:
        return []

    company_cols = [col for col in df.columns if col.startswith('production_companies_')]
    companies = [col.replace('production_companies_', '').replace('_', ' ') for col in company_cols]
    return companies


def filter_dataframe(df, filters):
    """Apply filters to dataframe"""
    if df is None or df.empty:
        return df

    filtered_df = df.copy()

    # Apply genre filter
    if filters['genre'] != 'All Genres':
        genre_col = f"genre_{filters['genre']}"
        if genre_col in filtered_df.columns:
            filtered_df = filtered_df[filtered_df[genre_col] == 1]

    # Apply company filter
    if filters['company'] != 'All Companies':
        company_name = filters['company'].replace(' ', '_')
        company_col = f"production_companies_{company_name}"
        if company_col in filtered_df.columns:
            filtered_df = filtered_df[filtered_df[company_col] == 1]

    # Apply year filter
    if filters['min_year'] is not None and 'release_year' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['release_year'] >= filters['min_year']]
    if filters['max_year'] is not None and 'release_year' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['release_year'] <= filters['max_year']]

    # Apply season filter
    if filters['season'] != 'All Seasons' and 'release_season' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['release_season'] == filters['season']]

    # Apply budget filter
    if filters['min_budget'] is not None and 'budget' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['budget'] >= filters['min_budget']]
    if filters['max_budget'] is not None and 'budget' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['budget'] <= filters['max_budget']]

    return filtered_df


def filter_by_time_period(df, year_range):
    """Filter dataframe by time period (year range)"""
    if df is None or df.empty or 'release_year' not in df.columns:
        return df

    filtered_df = df[(df['release_year'] >= year_range[0]) & (df['release_year'] <= year_range[1])]
    return filtered_df


# --- Visualization Functions ---
def display_top_movies(df, metric, n=10, time_label="All Movies", expanded_view=False):
    """Display top n movies based on the selected metric"""
    if df is None or df.empty:
        st.warning(f"No data available for {time_label}.")
        return None

    # Create a clean display dataframe
    metric_col = f'predicted_{metric}'
    if metric_col not in df.columns:
        st.error(f"Prediction column '{metric_col}' not found in the dataset.")
        return None

    # Sort by the prediction metric
    df_sorted = df.sort_values(metric_col, ascending=False).head(n)

    # Check if we have any movies to display
    if df_sorted.empty:
        st.info(f"No movies found with the current filters for {time_label}.")
        return None

    # Add production companies and genres as columns
    df_sorted['Production Companies'] = df_sorted.apply(get_production_companies, axis=1)
    df_sorted['Genres'] = df_sorted.apply(get_genres, axis=1)

    # Select and rename columns for display
    display_cols = [
        'title', metric_col, 'budget',
        'release_year', 'release_season', 'release_month',
        'Production Companies', 'Genres'
    ]

    # Ensure all display columns exist in the dataframe
    display_cols = [col for col in display_cols if col in df_sorted.columns]

    # Create a display dataframe with renamed columns
    df_display = df_sorted[display_cols].copy()

    # Rename columns for display
    rename_map = {
        'title': 'Title',
        metric_col: f'Predicted {metric.capitalize()}',
        'budget': 'Budget',
        'release_year': 'Year',
        'release_season': 'Season',
        'release_month': 'Month'
    }
    df_display = df_display.rename(columns=rename_map)

    # Format month names
    if 'Month' in df_display.columns:
        df_display['Month'] = df_display['Month'].apply(
            lambda x: calendar.month_name[x] if isinstance(x, (int, float)) and 1 <= x <= 12 else x)

    # If we're showing an expanded view, use a more visual layout
    if expanded_view:
        # Create visual cards for each movie
        for i, (idx, movie) in enumerate(df_sorted.iterrows()):
            with st.container():
                col1, col2 = st.columns([3, 1])

                with col1:
                    title = movie.get('title', 'Unknown Title')
                    year = movie.get('release_year', 'TBD')
                    genres = get_genres(movie)
                    companies = get_production_companies(movie)

                    st.markdown(f"### {i + 1}. {title} ({year})")
                    st.markdown(f"**Genres:** {genres}")
                    st.markdown(f"**Production:** {companies}")

                    # Release timing
                    release_info = []
                    if 'release_season' in movie and not pd.isna(movie['release_season']):
                        release_info.append(f"Season: {movie['release_season']}")
                    if 'release_month' in movie and not pd.isna(movie['release_month']) and isinstance(
                            movie['release_month'], (int, float)):
                        release_info.append(f"Month: {calendar.month_name[int(movie['release_month'])]}")

                    if release_info:
                        st.markdown(f"**Release:** {', '.join(release_info)}")

                with col2:
                    # Display the prediction metric prominently
                    prediction_value = movie[metric_col]
                    color = get_performance_color(prediction_value, df, metric_col)

                    st.markdown(f"""
                    <div style="background-color: {color}; color: white; padding: 1rem; border-radius: 0.5rem; text-align: center;">
                        <div style="font-size: 0.9rem; margin-bottom: 0.5rem;">Predicted {metric.capitalize()}</div>
                        <div style="font-size: 1.8rem; font-weight: 700;">{format_value(prediction_value, metric)}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Display budget if available
                    if 'budget' in movie and not pd.isna(movie['budget']):
                        st.markdown(f"""
                        <div style="background-color: #F3F4F6; padding: 1rem; border-radius: 0.5rem; text-align: center; margin-top: 0.5rem;">
                            <div style="font-size: 0.9rem; margin-bottom: 0.5rem;">Budget</div>
                            <div style="font-size: 1.4rem; font-weight: 600;">${movie['budget']:,.0f}</div>
                        </div>
                        """, unsafe_allow_html=True)

                st.markdown("---")
    else:
        # Standard table view
        # Format numeric columns
        format_dict = {}
        if f'Predicted {metric.capitalize()}' in df_display.columns:
            if metric == 'revenue':
                format_dict[f'Predicted {metric.capitalize()}'] = "${:,.0f}"
            elif metric == 'roi':
                format_dict[f'Predicted {metric.capitalize()}'] = "{:.2f}x"
            else:
                format_dict[f'Predicted {metric.capitalize()}'] = "{:.2f}"

        if 'Budget' in df_display.columns:
            format_dict['Budget'] = "${:,.0f}"

        # Create column config for better display
        column_config = {
            'Title': st.column_config.TextColumn(
                "Movie Title",
                help="The title of the movie"
            ),
            'Year': st.column_config.NumberColumn(
                "Release Year",
                format="%d",
                help="The release year"
            ),
            'Month': st.column_config.TextColumn(
                "Release Month",
                help="The release month"
            ),
            'Season': st.column_config.TextColumn(
                "Release Season",
                help="The release season"
            )
        }

        # Add metric-specific formatting
        if f'Predicted {metric.capitalize()}' in df_display.columns:
            if metric == 'revenue':
                column_config[f'Predicted {metric.capitalize()}'] = st.column_config.NumberColumn(
                    f"Predicted {metric.capitalize()}",
                    format="$%d",
                    help=f"The predicted {metric} value"
                )
            elif metric == 'roi':
                column_config[f'Predicted {metric.capitalize()}'] = st.column_config.NumberColumn(
                    f"Predicted {metric.capitalize()}",
                    format="%.2fx",
                    help=f"The predicted {metric} value"
                )
            else:
                column_config[f'Predicted {metric.capitalize()}'] = st.column_config.NumberColumn(
                    f"Predicted {metric.capitalize()}",
                    format="%.2f",
                    help=f"The predicted {metric} value"
                )

        if 'Budget' in df_display.columns:
            column_config['Budget'] = st.column_config.NumberColumn(
                "Budget",
                format="$%d",
                help="The production budget"
            )

        # Display the dataframe with enhanced formatting
        st.dataframe(
            df_display,
            column_config=column_config,
            use_container_width=True,
            height=min(400, 50 + 35 * len(df_display))
        )

    return df_sorted


def create_time_analysis(df, selected_metric):
    """Create visualizations for time-based analysis"""
    metric_col = f'predicted_{selected_metric}'

    if df is None or df.empty or metric_col not in df.columns:
        st.info("Not enough data available for time analysis.")
        return

    col1, col2 = st.columns(2)

    # Season analysis
    with col1:
        st.subheader("Performance by Season")

        if 'release_season' in df.columns:
            season_data = df.groupby('release_season')[metric_col].agg(['mean', 'count']).reset_index()

            if not season_data.empty:
                # Define season order and colors
                season_order = ['Winter', 'Spring', 'Summer', 'Fall']
                season_colors = {
                    'Spring': '#4ADE80',  # Green
                    'Summer': '#F59E0B',  # Amber
                    'Fall': '#B45309',  # Brown
                    'Winter': '#93C5FD'  # Light blue
                }

                # Create Plotly bar chart
                fig = px.bar(
                    season_data,
                    x='release_season',
                    y='mean',
                    color='release_season',
                    labels={
                        'release_season': 'Season',
                        'mean': f'Avg. {selected_metric.capitalize()}'
                    },
                    text='count',
                    category_orders={"release_season": season_order},
                    color_discrete_map=season_colors
                )

                # Update layout
                fig.update_layout(
                    xaxis_title='Season',
                    yaxis_title=f'Avg. {selected_metric.capitalize()}',
                    showlegend=False,
                    height=400,
                    plot_bgcolor='white'
                )

                # Format y-axis based on metric
                if selected_metric == 'revenue':
                    fig.update_yaxes(tickprefix='$', tickformat=',.0f')

                # Add count labels
                fig.update_traces(
                    texttemplate='%{text} movies',
                    textposition='outside',
                    hovertemplate=(
                            "<b>Season:</b> %{x}<br>" +
                            f"<b>Avg. {selected_metric.capitalize()}:</b> " +
                            ("$%{y:,.0f}" if selected_metric == 'revenue' else "%{y:.2f}") +
                            "<br><b>Movies:</b> %{text}<extra></extra>"
                    )
                )

                # Display the chart
                st.plotly_chart(fig, use_container_width=True)

                # Find best season
                best_season_idx = season_data['mean'].idxmax()
                best_season = season_data.loc[best_season_idx, 'release_season']
                best_season_value = season_data.loc[best_season_idx, 'mean']

                st.metric(
                    label="Best Season for Release",
                    value=best_season,
                    delta=f"{format_value(best_season_value, selected_metric)}"
                )
            else:
                st.info("Not enough season data for analysis.")
        else:
            st.info("No season information found in the dataset.")

    # Month analysis
    with col2:
        st.subheader("Performance by Month")

        if 'release_month' in df.columns:
            month_data = df.groupby('release_month')[metric_col].agg(['mean', 'count']).reset_index()

            if not month_data.empty:
                # Add month names
                month_data['month_name'] = month_data['release_month'].apply(
                    lambda x: calendar.month_name[int(x)] if isinstance(x, (int, float)) and 1 <= x <= 12 else str(x)
                )

                # Create Plotly bar chart
                fig = px.bar(
                    month_data,
                    x='release_month',
                    y='mean',
                    color='mean',  # Color based on value for gradient effect
                    labels={
                        'release_month': 'Month',
                        'mean': f'Avg. {selected_metric.capitalize()}'
                    },
                    text='count',
                    color_continuous_scale='Blues'
                )

                # Update layout
                fig.update_layout(
                    xaxis_title='Month',
                    xaxis=dict(
                        tickmode='array',
                        tickvals=list(range(1, 13)),
                        ticktext=[calendar.month_abbr[i] for i in range(1, 13)]
                    ),
                    yaxis_title=f'Avg. {selected_metric.capitalize()}',
                    showlegend=False,
                    height=400,
                    plot_bgcolor='white',
                    coloraxis_showscale=False
                )

                # Format y-axis based on metric
                if selected_metric == 'revenue':
                    fig.update_yaxes(tickprefix='$', tickformat=',.0f')

                # Add count labels
                fig.update_traces(
                    texttemplate='%{text} movies',
                    textposition='outside',
                    hovertemplate=(
                            "<b>Month:</b> %{customdata}<br>" +
                            f"<b>Avg. {selected_metric.capitalize()}:</b> " +
                            ("$%{y:,.0f}" if selected_metric == 'revenue' else "%{y:.2f}") +
                            "<br><b>Movies:</b> %{text}<extra></extra>"
                    ),
                    customdata=month_data['month_name']
                )

                # Display the chart
                st.plotly_chart(fig, use_container_width=True)

                # Find best month
                best_month_idx = month_data['mean'].idxmax()
                best_month_num = month_data.loc[best_month_idx, 'release_month']
                best_month_name = calendar.month_name[int(best_month_num)] if isinstance(best_month_num, (
                    int, float)) and 1 <= best_month_num <= 12 else str(best_month_num)
                best_month_value = month_data.loc[best_month_idx, 'mean']

                st.metric(
                    label="Best Month for Release",
                    value=best_month_name,
                    delta=f"{format_value(best_month_value, selected_metric)}"
                )
            else:
                st.info("Not enough month data for analysis.")
        else:
            st.info("No month information found in the dataset.")


def create_genre_analysis(df, selected_metric):
    """Create visualizations for genre analysis"""
    metric_col = f'predicted_{selected_metric}'

    if df is None or df.empty or metric_col not in df.columns:
        st.info("Not enough data available for genre analysis.")
        return

    # Get all genre columns
    genre_cols = [col for col in df.columns if col.startswith('genre_')]

    if not genre_cols:
        st.info("No genre information found in the dataset.")
        return

    # Calculate average metric value for each genre
    genre_data = []
    for genre in genre_cols:
        genre_movies = df[df[genre] == 1]
        if len(genre_movies) >= MIN_SAMPLE_SIZE:  # Only include genres with enough movies
            avg_value = genre_movies[metric_col].mean()
            genre_data.append({
                'genre': genre.replace('genre_', ''),
                'avg_value': avg_value,
                'count': len(genre_movies)
            })

    if not genre_data:
        st.info("Not enough data per genre for analysis.")
        return

    genre_df = pd.DataFrame(genre_data).sort_values('avg_value', ascending=False)

    # Create Plotly horizontal bar chart
    fig = go.Figure()

    # Add bars
    fig.add_trace(go.Bar(
        y=genre_df['genre'],
        x=genre_df['avg_value'],
        orientation='h',
        marker=dict(
            color=genre_df['avg_value'],
            colorscale='Blues',
            showscale=False
        ),
        text=genre_df['count'],
        textposition="outside",
        texttemplate="%{text} movies",
        hovertemplate=(
                "<b>Genre:</b> %{y}<br>" +
                f"<b>Avg. {selected_metric.capitalize()}:</b> " +
                ("%{x:$.2f}" if selected_metric == 'revenue' else "%{x:.2f}") +
                "<br><b>Movies:</b> %{text}<extra></extra>"
        )
    ))

    # Update layout
    fig.update_layout(
        title=f"Genre Performance Analysis: {selected_metric.capitalize()}",
        xaxis_title=f"Average {selected_metric.capitalize()}" + (
            " ($)" if selected_metric == 'revenue' else ""),
        yaxis_title="Genre",
        yaxis=dict(categoryorder='total ascending'),
        height=max(400, 50 + 25 * len(genre_df)),
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor='white'
    )

    fig.update_xaxes(
        gridcolor='#F3F4F6',
        zerolinecolor='#E5E7EB'
    )

    fig.update_yaxes(
        gridcolor='#F3F4F6'
    )

    st.plotly_chart(fig, use_container_width=True)

    # Show top 3 and bottom 3 genres with insights
    if len(genre_df) >= 6:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Top Performing Genres")
            top_genres = genre_df.head(3)

            for i, (idx, genre) in enumerate(top_genres.iterrows()):
                st.markdown(f"""
                <div style="padding: 0.5rem; margin-bottom: 0.5rem; background-color: #EFF6FF; border-radius: 0.25rem;">
                    <div style="font-weight: 600;">#{i + 1}: {genre['genre']}</div>
                    <div style="display: flex; justify-content: space-between;">
                        <span>{format_value(genre['avg_value'], selected_metric)}</span>
                        <span>{genre['count']} movies</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with col2:
            st.subheader("Lower Performing Genres")
            bottom_genres = genre_df.tail(3).iloc[::-1]  # Reverse to show worst first

            for i, (idx, genre) in enumerate(bottom_genres.iterrows()):
                st.markdown(f"""
                <div style="padding: 0.5rem; margin-bottom: 0.5rem; background-color: #FEF2F2; border-radius: 0.25rem;">
                    <div style="font-weight: 600;">#{i + 1}: {genre['genre']}</div>
                    <div style="display: flex; justify-content: space-between;">
                        <span>{format_value(genre['avg_value'], selected_metric)}</span>
                        <span>{genre['count']} movies</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)


def create_company_analysis(df, selected_metric):
    """Create visualizations for production company analysis"""
    metric_col = f'predicted_{selected_metric}'

    if df is None or df.empty or metric_col not in df.columns:
        st.info("Not enough data available for company analysis.")
        return

    # Get all production company columns
    company_cols = [col for col in df.columns if col.startswith('production_companies_')]

    if not company_cols:
        st.info("No production company information found in the dataset.")
        return

    # Calculate average metric value for each company
    company_data = []
    for company in company_cols:
        company_movies = df[df[company] == 1]
        if len(company_movies) >= MIN_SAMPLE_SIZE:  # Only include companies with enough movies
            avg_value = company_movies[metric_col].mean()
            company_data.append({
                'company': company.replace('production_companies_', '').replace('_', ' '),
                'avg_value': avg_value,
                'count': len(company_movies)
            })

    if not company_data:
        st.info("Not enough data per production company for analysis.")
        return

    company_df = pd.DataFrame(company_data).sort_values('avg_value', ascending=False)

    # Limit to top 12 companies for cleaner visualization
    top_n = min(12, len(company_df))
    top_companies = company_df.head(top_n)

    # Create Plotly horizontal bar chart
    fig = go.Figure()

    # Add bars
    fig.add_trace(go.Bar(
        y=top_companies['company'],
        x=top_companies['avg_value'],
        orientation='h',
        marker=dict(
            color=top_companies['avg_value'],
            colorscale='Blues',
            showscale=False
        ),
        text=top_companies['count'],
        textposition="outside",
        texttemplate="%{text} movies",
        hovertemplate=(
                "<b>Company:</b> %{y}<br>" +
                f"<b>Avg. {selected_metric.capitalize()}:</b> " +
                ("%{x:$.2f}" if selected_metric == 'revenue' else "%{x:.2f}") +
                "<br><b>Movies:</b> %{text}<extra></extra>"
        )
    ))

    # Update layout
    fig.update_layout(
        title=f"Top {top_n} Production Companies by {selected_metric.capitalize()}",
        xaxis_title=f"Average {selected_metric.capitalize()}" + (
            " ($)" if selected_metric == 'revenue' else ""),
        yaxis_title="Production Company",
        yaxis=dict(categoryorder='total ascending'),
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor='white'
    )

    fig.update_xaxes(
        gridcolor='#F3F4F6',
        zerolinecolor='#E5E7EB'
    )

    fig.update_yaxes(
        gridcolor='#F3F4F6'
    )

    st.plotly_chart(fig, use_container_width=True)

    # Show top 3 companies with cards
    top_3_companies = company_df.head(3)

    st.subheader("Leading Production Companies")

    cols = st.columns(3)
    for i, (idx, company) in enumerate(top_3_companies.iterrows()):
        with cols[i]:
            st.markdown(f"""
            <div style="background-color: white; padding: 1rem; border-radius: 0.5rem; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); text-align: center;">
                <div style="font-size: 1rem; font-weight: 600; color: #1E3A8A; margin-bottom: 0.5rem;">
                    {company['company']}
                </div>
                <div style="font-size: 1.25rem; font-weight: 700; color: #1E3A8A;">
                    {format_value(company['avg_value'], selected_metric)}
                </div>
                <div style="font-size: 0.8rem; color: #6B7280; margin-top: 0.25rem;">
                    Based on {company['count']} movies
                </div>
            </div>
            """, unsafe_allow_html=True)


def create_budget_analysis(df, selected_metric):
    """Create visualizations for budget analysis"""
    metric_col = f'predicted_{selected_metric}'

    if df is None or df.empty or metric_col not in df.columns or 'budget' not in df.columns:
        st.info("Budget information not available for analysis.")
        return

    # Drop any rows with missing budget or metric values
    df_clean = df.dropna(subset=[metric_col, 'budget'])

    if len(df_clean) < 5:
        st.info("Not enough complete data for budget analysis.")
        return

    # Create Plotly scatter plot
    fig = px.scatter(
        df_clean,
        x='budget',
        y=metric_col,
        hover_name='title',
        color=metric_col,
        color_continuous_scale='Blues',
        size='budget',
        size_max=30,
        log_x=True,
        log_y=selected_metric == 'revenue',  # Log scale for revenue makes more sense
        labels={
            'budget': 'Budget ($)',
            metric_col: f'{selected_metric.capitalize()}' + (' ($)' if selected_metric == 'revenue' else '')
        },
        title=f'Budget vs. {selected_metric.capitalize()}'
    )

    # Add a trendline
    fig.update_layout(
        height=500,
        plot_bgcolor='white',
        coloraxis_colorbar=dict(title=f'{selected_metric.capitalize()}'),
        margin=dict(l=20, r=20, t=40, b=20)
    )

    fig.update_traces(
        marker=dict(line=dict(width=1, color='DarkSlateGrey')),
        selector=dict(mode='markers')
    )

    # Add a trendline
    fig.update_xaxes(
        gridcolor='#F3F4F6',
        zerolinecolor='#E5E7EB'
    )

    fig.update_yaxes(
        gridcolor='#F3F4F6',
        zerolinecolor='#E5E7EB'
    )

    # Add a trend line
    try:
        trendline = px.scatter(
            df_clean,
            x='budget',
            y=metric_col,
            trendline='ols',
            log_x=True,
            log_y=selected_metric == 'revenue'
        )

        for trace in trendline.data:
            if trace.mode == 'lines':
                trace.line.color = 'red'
                trace.line.width = 2
                trace.line.dash = 'dash'
                trace.showlegend = True
                trace.name = 'Trend'
                fig.add_trace(trace)
    except Exception as e:
        st.warning(f"Could not generate trendline: {e}")

    st.plotly_chart(fig, use_container_width=True)

    # Calculate correlation
    corr = df_clean[[metric_col, 'budget']].corr().iloc[0, 1]

    # Budget insights
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="Budget to Performance Correlation",
            value=f"{corr:.2f}",
            delta=None
        )

        correlation_strength = ""
        if abs(corr) > 0.7:
            correlation_strength = "Strong"
        elif abs(corr) > 0.4:
            correlation_strength = "Moderate"
        else:
            correlation_strength = "Weak"

        correlation_direction = "positive" if corr > 0 else "negative"

        st.markdown(
            f"There is a **{correlation_strength} {correlation_direction}** correlation between budget and {selected_metric}.")

    with col2:
        # Find optimal budget range
        try:
            # Create budget bins
            df_clean['budget_bin'] = pd.qcut(df_clean['budget'], q=4, labels=False)
            budget_performance = df_clean.groupby('budget_bin').agg({
                'budget': ['min', 'max', 'mean'],
                metric_col: 'mean'
            })

            # Find best performing bin
            best_bin_idx = budget_performance[metric_col]['mean'].idxmax()
            min_budget = budget_performance['budget']['min'][best_bin_idx]
            max_budget = budget_performance['budget']['max'][best_bin_idx]

            st.metric(
                label="Optimal Budget Range",
                value=f"${min_budget / 1_000_000:.1f}M - ${max_budget / 1_000_000:.1f}M"
            )

            st.markdown(
                f"Movies with budgets in this range show the strongest {selected_metric} performance.")
        except Exception as e:
            st.error(f"Error calculating optimal budget: {e}")


def create_summary_dashboard(df, selected_metric):
    """Create a summary dashboard with key insights"""
    metric_col = f'predicted_{selected_metric}'

    if df is None or df.empty or metric_col not in df.columns:
        st.error("No data available for analysis.")
        return

    # Key metrics
    total_movies = len(df)
    avg_metric = df[metric_col].mean() if metric_col in df.columns else None

    # Year range
    min_year = int(df['release_year'].min()) if 'release_year' in df.columns else None
    max_year = int(df['release_year'].max()) if 'release_year' in df.columns else None
    year_range = f"{min_year} - {max_year}" if min_year and max_year else "N/A"

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total Movies",
            value=total_movies
        )

    with col2:
        if avg_metric is not None:
            st.metric(
                label=f"Avg {selected_metric.capitalize()}",
                value=format_value(avg_metric, selected_metric)
            )

    with col3:
        st.metric(
            label="Year Range",
            value=year_range
        )

    with col4:
        # Find top movie
        if metric_col in df.columns:
            top_movie = df.nlargest(1, metric_col)
            top_movie_title = top_movie['title'].iloc[0] if 'title' in top_movie.columns else "Unknown"
            st.metric(
                label="Top Movie",
                value=top_movie_title
            )

    # Style metrics
    style_metric_cards()

    # Overall stats
    st.markdown("---")

    # Top performers card
    if metric_col in df.columns:
        top_movie = df.nlargest(1, metric_col)
        top_movie_title = top_movie['title'].iloc[0] if 'title' in top_movie.columns else "Unknown"
        top_movie_value = top_movie[metric_col].iloc[0]
        top_movie_year = int(top_movie['release_year'].iloc[0]) if 'release_year' in top_movie.columns else None

        st.markdown(f"""
        <div style="background-color: #DBEAFE; padding: 1rem; border-radius: 0.5rem;">
            <div style="font-size: 0.9rem; color: #1E40AF;">Top Performing Movie</div>
            <div style="font-size: 1.2rem; font-weight: 600; color: #1E3A8A; margin: 0.5rem 0;">{top_movie_title} ({top_movie_year})</div>
            <div style="font-size: 1.5rem; font-weight: 700; color: #1E3A8A;">
                {format_value(top_movie_value, selected_metric)}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Create combined charts for insights
    st.markdown("---")

    # Timing and genre insights
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("When to Release")

        # Timing analysis
        if 'release_season' in df.columns and metric_col in df.columns:
            season_data = df.groupby('release_season')[metric_col].mean().reset_index()

            if not season_data.empty:
                season_colors = {
                    'Spring': '#4ADE80',
                    'Summer': '#F59E0B',
                    'Fall': '#B45309',
                    'Winter': '#93C5FD'
                }

                color_map = {row['release_season']: season_colors.get(row['release_season'], '#3B82F6')
                             for _, row in season_data.iterrows()}

                fig = px.bar(
                    season_data,
                    x='release_season',
                    y=metric_col,
                    title="Performance by Season",
                    color='release_season',
                    color_discrete_map=color_map,
                    labels={
                        'release_season': 'Season',
                        metric_col: f'{selected_metric.capitalize()}'
                    }
                )

                fig.update_layout(
                    height=300,
                    showlegend=False,
                    plot_bgcolor='white',
                    margin=dict(l=20, r=20, t=40, b=20)
                )

                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough timing data available.")

    with col2:
        st.subheader("Top Genres")

        # Genre analysis
        genre_cols = [col for col in df.columns if col.startswith('genre_')]

        if genre_cols and metric_col in df.columns:
            genre_data = []

            for genre in genre_cols:
                genre_movies = df[df[genre] == 1]
                if len(genre_movies) >= MIN_SAMPLE_SIZE:
                    avg_value = genre_movies[metric_col].mean()
                    genre_data.append({
                        'genre': genre.replace('genre_', ''),
                        'avg_value': avg_value,
                        'count': len(genre_movies)
                    })

            if genre_data:
                genre_df = pd.DataFrame(genre_data).sort_values('avg_value', ascending=False).head(5)

                fig = px.bar(
                    genre_df,
                    x='genre',
                    y='avg_value',
                    title="Top 5 Genres",
                    color='avg_value',
                    color_continuous_scale='Blues',
                    labels={
                        'genre': 'Genre',
                        'avg_value': f'{selected_metric.capitalize()}'
                    }
                )

                fig.update_layout(
                    height=300,
                    coloraxis_showscale=False,
                    plot_bgcolor='white',
                    margin=dict(l=20, r=20, t=40, b=20)
                )

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough genre data available.")
        else:
            st.info("No genre data available.")


def create_time_period_comparison(df, selected_metric, period1_range, period2_range, period1_label, period2_label):
    """Create comparison between two time periods"""
    metric_col = f'predicted_{selected_metric}'

    if df is None or df.empty or metric_col not in df.columns or 'release_year' not in df.columns:
        st.error("Cannot perform time period comparison: missing required data.")
        return

    # Filter data for each time period
    period1_df = filter_by_time_period(df, period1_range)
    period2_df = filter_by_time_period(df, period2_range)

    if period1_df.empty or period2_df.empty:
        st.error(
            f"Not enough data for one or both time periods. Period 1: {len(period1_df)} movies, Period 2: {len(period2_df)} movies.")
        return

    # Display comparison metrics
    period1_mean = period1_df[metric_col].mean() if not period1_df.empty else None
    period2_mean = period2_df[metric_col].mean() if not period2_df.empty else None

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        if period1_mean is not None and period2_mean is not None:
            diff_pct = (period2_mean - period1_mean) / period1_mean * 100 if period1_mean != 0 else 0

            st.markdown(f"""
            <div style="background-color: white; padding: 1rem; border-radius: 0.5rem; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); text-align: center; margin-bottom: 1rem;">
                <div style="font-size: 1rem; color: #4B5563;">Average {selected_metric.capitalize()} Comparison</div>
                <div style="display: flex; justify-content: space-around; margin: 1rem 0;">
                    <div>
                        <div style="font-size: 0.9rem; color: #92400E;">{period1_label} ({period1_range[0]}-{period1_range[1]})</div>
                        <div style="font-size: 1.5rem; font-weight: 700; color: #92400E;">{format_value(period1_mean, selected_metric)}</div>
                    </div>
                    <div style="font-size: 1.5rem; font-weight: 700; color: #6B7280;">vs.</div>
                    <div>
                        <div style="font-size: 0.9rem; color: #1E40AF;">{period2_label} ({period2_range[0]}-{period2_range[1]})</div>
                        <div style="font-size: 1.5rem; font-weight: 700; color: #1E40AF;">{format_value(period2_mean, selected_metric)}</div>
                    </div>
                </div>
                <div style="font-size: 0.9rem; color: #4B5563;">
                    {period2_label} average is <span style="font-weight: 600; color: {'#16A34A' if diff_pct >= 0 else '#EF4444'};">
                        {abs(diff_pct):.1f}% {'higher' if diff_pct >= 0 else 'lower'}
                    </span> than {period1_label}.
                </div>
            </div>
            """, unsafe_allow_html=True)

    # 1. Genre comparison
    st.subheader("Genre Performance by Time Period")

    # Get genre data for each time period
    period1_genres = []
    period2_genres = []

    genre_cols = [col for col in df.columns if col.startswith('genre_')]

    for genre in genre_cols:
        genre_name = genre.replace('genre_', '')

        # Period 1 stats
        if not period1_df.empty:
            genre_movies = period1_df[period1_df[genre] == 1]
            if len(genre_movies) >= MIN_SAMPLE_SIZE:
                period1_genres.append({
                    'genre': genre_name,
                    'avg_value': genre_movies[metric_col].mean() if metric_col in genre_movies.columns else 0,
                    'count': len(genre_movies),
                    'period': period1_label
                })

        # Period 2 stats
        if not period2_df.empty:
            genre_movies = period2_df[period2_df[genre] == 1]
            if len(genre_movies) >= MIN_SAMPLE_SIZE:
                period2_genres.append({
                    'genre': genre_name,
                    'avg_value': genre_movies[metric_col].mean() if metric_col in genre_movies.columns else 0,
                    'count': len(genre_movies),
                    'period': period2_label
                })

    # Combine genre data
    all_genres = period1_genres + period2_genres

    if all_genres:
        genres_df = pd.DataFrame(all_genres)

        # Get genres that exist in both periods
        common_genres = set(g['genre'] for g in period1_genres) & set(g['genre'] for g in period2_genres)

        if common_genres:
            # Filter to common genres for fair comparison
            genres_df = genres_df[genres_df['genre'].isin(common_genres)]

            # Create grouped bar chart
            fig = px.bar(
                genres_df,
                x='genre',
                y='avg_value',
                color='period',
                barmode='group',
                color_discrete_map={
                    period1_label: '#FCD34D',  # Amber
                    period2_label: '#3B82F6'  # Blue
                },
                labels={
                    'genre': 'Genre',
                    'avg_value': f'Avg. {selected_metric.capitalize()}',
                    'period': 'Time Period'
                }
            )

            fig.update_layout(
                height=400,
                plot_bgcolor='white',
                margin=dict(l=20, r=20, t=40, b=20)
            )

            st.plotly_chart(fig, use_container_width=True)

            # Identify genres with biggest differences
            genres_wide = genres_df.pivot(index='genre', columns='period', values='avg_value').reset_index()
            genres_wide['difference'] = genres_wide[period2_label] - genres_wide[period1_label]
            genres_wide['difference_pct'] = genres_wide['difference'] / genres_wide[period1_label] * 100

            # Find biggest positive and negative differences
            biggest_increase = genres_wide.nlargest(1, 'difference_pct')
            biggest_decrease = genres_wide.nsmallest(1, 'difference_pct')

            col1, col2 = st.columns(2)

            with col1:
                if not biggest_increase.empty:
                    genre_name = biggest_increase['genre'].iloc[0]
                    diff_pct = biggest_increase['difference_pct'].iloc[0]

                    st.markdown(f"""
                    <div style="background-color: #DCFCE7; padding: 0.75rem; border-radius: 0.5rem;">
                        <div style="font-size: 0.9rem; color: #166534;">Most Improved Genre</div>
                        <div style="font-size: 1.2rem; font-weight: 600; color: #166534;">{genre_name}</div>
                        <div style="font-size: 0.9rem; color: #166534;">
                            {diff_pct:.1f}% higher in {period2_label} vs. {period1_label}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            with col2:
                if not biggest_decrease.empty:
                    genre_name = biggest_decrease['genre'].iloc[0]
                    diff_pct = biggest_decrease['difference_pct'].iloc[0]

                    st.markdown(f"""
                    <div style="background-color: #FEE2E2; padding: 0.75rem; border-radius: 0.5rem;">
                        <div style="font-size: 0.9rem; color: #B91C1C;">Declining Genre</div>
                        <div style="font-size: 1.2rem; font-weight: 600; color: #B91C1C;">{genre_name}</div>
                        <div style="font-size: 0.9rem; color: #B91C1C;">
                            {abs(diff_pct):.1f}% lower in {period2_label} vs. {period1_label}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info(f"No common genres between {period1_label} and {period2_label} with enough data for comparison.")
    else:
        st.info("Not enough genre data available for comparison.")

    # 2. Budget impact comparison
    st.markdown("---")
    st.subheader("Budget Impact by Time Period")

    if 'budget' in df.columns:
        # Create scatter plot with both time periods
        period1_budget = period1_df.dropna(subset=['budget', metric_col])
        period2_budget = period2_df.dropna(subset=['budget', metric_col])

        if not period1_budget.empty and not period2_budget.empty:
            # Combine for visualization
            period1_budget = period1_budget.copy()
            period2_budget = period2_budget.copy()

            # Add period column for coloring
            period1_budget['period'] = period1_label
            period2_budget['period'] = period2_label

            combined_budget = pd.concat([period1_budget, period2_budget])

            # Create scatter plot
            fig = px.scatter(
                combined_budget,
                x='budget',
                y=metric_col,
                color='period',
                hover_name='title',
                log_x=True,
                log_y=selected_metric == 'revenue',
                color_discrete_map={
                    period1_label: '#FCD34D',
                    period2_label: '#3B82F6'
                },
                labels={
                    'budget': 'Budget ($)',
                    metric_col: f'{selected_metric.capitalize()}',
                    'period': 'Time Period'
                },
                trendline='ols',
                trendline_scope='trace'
            )

            fig.update_layout(
                height=500,
                plot_bgcolor='white',
                margin=dict(l=20, r=20, t=40, b=20)
            )

            st.plotly_chart(fig, use_container_width=True)

            # Calculate correlations
            period1_corr = period1_budget[['budget', metric_col]].corr().iloc[0, 1]
            period2_corr = period2_budget[['budget', metric_col]].corr().iloc[0, 1]

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    label=f"{period1_label} Budget Correlation",
                    value=f"{period1_corr:.2f}"
                )

            with col2:
                st.metric(
                    label=f"{period2_label} Budget Correlation",
                    value=f"{period2_corr:.2f}"
                )

            # Interpretation
            corr_diff = period2_corr - period1_corr

            st.markdown(f"""
            <div style="background-color: #F3F4F6; padding: 1rem; border-radius: 0.5rem; margin-top: 1rem;">
                <div style="font-weight: 600; color: #1E3A8A; margin-bottom: 0.5rem;">Budget Impact Interpretation</div>
                <div>
                    The correlation between budget and {selected_metric} is 
                    <span style="font-weight: 600; color: {'#16A34A' if corr_diff > 0 else '#EF4444'};">
                        {abs(corr_diff):.2f} points {'stronger' if corr_diff > 0 else 'weaker'}
                    </span> 
                    in {period2_label} compared to {period1_label}.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Not enough budget data for both time periods to make a comparison.")
    else:
        st.info("Budget information not available for comparison.")


# --- Main App ---
def main():
    st.markdown('<h1 class="main-header">🎬 Movie Performance Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Explore historical movie performance patterns and compare time periods</p>',
                unsafe_allow_html=True)

    # Load data
    df = load_data()

    # Show error if no data
    if df is None:
        st.error("Failed to load data. Please check the file path and format.")
        st.info("To use this app, you need to save your movie data as a Parquet file.")

        # Instructions
        with st.expander("How to prepare your data"):
            st.markdown("""
            1. After preprocessing your movie data, save the dataframe as a Parquet file:
            ```python
            df.to_parquet('output/movie_predictions.parquet')
            ```

            2. Make sure the dataframe contains:
               - Movie titles in a 'title' column
               - Values in a 'predicted_' column (e.g., 'predicted_revenue', 'predicted_roi')
               - Release information (year, month, season)
               - Genre columns starting with 'genre_'
               - Production company columns starting with 'production_companies_'
               - Budget information in a 'budget' column (optional)
            """)

        # Mock data option for demonstration
        if st.button("Use Sample Data for Demonstration", type="primary"):
            st.info("This would load sample data in a real implementation")

        return

    # Determine available metrics
    prediction_cols = [col for col in df.columns if col.startswith('predicted_')]
    available_metrics = [col.replace('predicted_', '') for col in prediction_cols]

    # Default to revenue if available
    default_metric = 'revenue' if 'predicted_revenue' in prediction_cols else available_metrics[
        0] if available_metrics else None

    if not available_metrics:
        st.error(
            "No prediction columns found in the data. Make sure your dataframe contains columns starting with 'predicted_'.")
        st.stop()

    # --- Sidebar Controls ---
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/movie-projector.png", width=80)
        st.title("Analysis Controls")

        # Metric selection
        st.header("1. Choose Metric")

        metric_descriptions = {
            'revenue': 'Box office earnings',
            'roi': 'Return on Investment',
            'popularity': 'Audience interest',
            'vote_count': 'Number of ratings',
            'vote_average': 'Average rating'
        }

        selected_metric = st.selectbox(
            "Measure success by:",
            options=available_metrics,
            index=available_metrics.index(default_metric) if default_metric in available_metrics else 0,
            format_func=lambda x: f"{x.capitalize()}"  # {metric_descriptions.get(x, '')}
        )

        # Number of movies to display
        st.header("2. Display Options")

        top_n = st.slider(
            "Number of top movies:",
            min_value=5,
            max_value=50,
            value=10,
            step=5
        )

        expanded_view = st.checkbox("Show expanded movie cards", value=False)

        # Time period selection
        st.header("3. Time Period Comparison")

        # Find min/max year in the dataset
        if 'release_year' in df.columns:
            overall_min_year = int(df['release_year'].min())
            overall_max_year = int(df['release_year'].max())
            year_span = overall_max_year - overall_min_year + 1

            # Set default ranges for comparison
            half_span = year_span // 2
            default_period1 = (overall_min_year, overall_min_year + half_span - 1)
            default_period2 = (overall_min_year + half_span, overall_max_year)

            # Create controls for time period 1
            st.subheader("First Time Period")
            period1_range = st.slider(
                "Year range:",
                min_value=overall_min_year,
                max_value=overall_max_year,
                value=(default_period1[0], default_period1[1]),
                key="period1_slider"
            )
            period1_label = st.text_input("Label for period:", "Older Period", key="period1_label")

            # Create controls for time period 2
            st.subheader("Second Time Period")
            period2_range = st.slider(
                "Year range:",
                min_value=overall_min_year,
                max_value=overall_max_year,
                value=(default_period2[0], default_period2[1]),
                key="period2_slider"
            )
            period2_label = st.text_input("Label for period:", "Recent Period", key="period2_label")
        else:
            st.warning("Release year data not found. Time period comparison will be disabled.")
            period1_range = (0, 0)
            period2_range = (0, 0)
            period1_label = "Period 1"
            period2_label = "Period 2"

        # Advanced filters section
        st.markdown("---")
        st.header("4. Filter Movies")

        # Extract all unique genres and companies
        all_genres = extract_all_genres(df)
        all_companies = extract_all_companies(df)

        # Filter by genre
        genre_filter = st.selectbox(
            "Filter by genre:",
            options=['All Genres'] + sorted(all_genres),
            index=0
        )

        # Filter by company
        company_filter = st.selectbox(
            "Filter by production company:",
            options=['All Companies'] + sorted(all_companies),
            index=0
        )

        # Year range filter
        min_year, max_year = None, None

        if 'release_year' in df.columns:
            min_year_data = int(df['release_year'].min())
            max_year_data = int(df['release_year'].max())

            year_range = st.slider(
                "Release year range:",
                min_value=min_year_data,
                max_value=max_year_data,
                value=(min_year_data, max_year_data),
                key="main_year_filter"
            )

            min_year, max_year = year_range

        # Season filter
        season_filter = st.selectbox(
            "Filter by season:",
            options=['All Seasons', 'Spring', 'Summer', 'Fall', 'Winter'],
            index=0
        )

        # Budget range filter
        min_budget, max_budget = None, None

        if 'budget' in df.columns:
            min_budget_data = float(df['budget'].min())
            max_budget_data = float(df['budget'].max())

            # Use log scale for budget slider to handle wide ranges
            budget_range = st.slider(
                "Budget range ($):",
                min_value=min_budget_data,
                max_value=max_budget_data,
                value=(min_budget_data, max_budget_data),
                format="$%d"
            )

            min_budget, max_budget = budget_range

        # Collect all filters
        filters = {
            'genre': genre_filter,
            'company': company_filter,
            'min_year': min_year,
            'max_year': max_year,
            'season': season_filter,
            'min_budget': min_budget,
            'max_budget': max_budget
        }

        # Apply filters to main dataframe
        filtered_df = filter_dataframe(df, filters)

        # Show filter summary
        st.markdown("---")
        if genre_filter != 'All Genres' or company_filter != 'All Companies' or season_filter != 'All Seasons':
            st.markdown("**Active Filters:**")
            filter_text = []
            if genre_filter != 'All Genres':
                filter_text.append(f"Genre: {genre_filter}")
            if company_filter != 'All Companies':
                filter_text.append(f"Company: {company_filter}")
            if season_filter != 'All Seasons':
                filter_text.append(f"Season: {season_filter}")
            st.markdown(", ".join(filter_text))

            # Reset filters button
            if st.button("Reset All Filters"):
                st.experimental_rerun()

    # --- Main Content ---
    # Use tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview",
        "🎬 Top Movies",
        "🔍 Detailed Analysis",
        "📈 Time Period Comparison"
    ])

    with tab1:
        # Overview dashboard with key insights
        colored_header(
            label="Performance Overview",
            description=f"Key insights based on {selected_metric} analysis",
            color_name="blue-70"
        )

        # Summary dashboard
        create_summary_dashboard(filtered_df, selected_metric)

        # Show data summary
        st.markdown("---")
        st.subheader("Data Summary")

        if filtered_df is not None and not filtered_df.empty:
            st.markdown(f"**Movies in dataset:** {len(filtered_df)}")

            if 'release_year' in filtered_df.columns:
                years = filtered_df['release_year'].unique()
                years = sorted([y for y in years if not pd.isna(y)])
                st.markdown(f"**Release Years:** {', '.join(map(str, years[:10]))}{'...' if len(years) > 10 else ''}")

            if 'budget' in filtered_df.columns:
                avg_budget = filtered_df['budget'].mean()
                st.markdown(f"**Average Budget:** ${avg_budget:,.0f}")

            # Add genre distribution
            genre_cols = [col for col in filtered_df.columns if col.startswith('genre_')]
            genre_counts = {col.replace('genre_', ''): filtered_df[col].sum() for col in genre_cols}
            genre_df = pd.DataFrame({'genre': list(genre_counts.keys()), 'count': list(genre_counts.values())})
            genre_df = genre_df.sort_values('count', ascending=False)

            st.markdown("**Top Genres in Dataset:**")
            fig = px.bar(
                genre_df.head(10),
                x='genre',
                y='count',
                title="Genre Distribution",
                color='count',
                color_continuous_scale='Blues',
            )
            fig.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown("**No movies match the current filters.**")

    with tab2:
        colored_header(
            label="Top Performing Movies",
            description=f"Movies with highest {selected_metric}",
            color_name="blue-70"
        )

        # Display top movies
        display_top_movies(
            filtered_df,
            selected_metric,
            n=top_n,
            time_label="All Time Periods",
            expanded_view=expanded_view
        )

    with tab3:
        colored_header(
            label=f"Detailed Analysis: {selected_metric.capitalize()}",
            description="Explore factors that impact movie performance",
            color_name="blue-70"
        )

        if filtered_df is None or filtered_df.empty:
            st.info("No movies match the current filters.")
        else:
            # Time-based analysis
            create_time_analysis(filtered_df, selected_metric)

            st.markdown("---")

            # Genre and Company Analysis
            col1, col2 = st.columns(2)

            with col1:
                create_genre_analysis(filtered_df, selected_metric)

            with col2:
                create_company_analysis(filtered_df, selected_metric)

            st.markdown("---")

            # Budget analysis
            create_budget_analysis(filtered_df, selected_metric)

    with tab4:
        colored_header(
            label="Time Period Comparison",
            description="Compare movie performance across different eras",
            color_name="blue-70"
        )

        # Create time period filters
        period1_df = filter_by_time_period(filtered_df, period1_range)
        period2_df = filter_by_time_period(filtered_df, period2_range)

        period1_count = len(period1_df) if period1_df is not None else 0
        period2_count = len(period2_df) if period2_df is not None else 0

        # Show time period info
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
            <div class="period-older" style="text-align: center; padding: 0.5rem;">
                {period1_label}: {period1_range[0]}-{period1_range[1]} ({period1_count} movies)
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="period-recent" style="text-align: center; padding: 0.5rem;">
                {period2_label}: {period2_range[0]}-{period2_range[1]} ({period2_count} movies)
            </div>
            """, unsafe_allow_html=True)

        # Check if we have data for comparison
        if period1_count < MIN_SAMPLE_SIZE or period2_count < MIN_SAMPLE_SIZE:
            st.warning(
                f"Insufficient data for time period comparison. Each period needs at least {MIN_SAMPLE_SIZE} movies.")

            # Show counts for debugging
            st.markdown(f"Movies in {period1_label}: {period1_count}")
            st.markdown(f"Movies in {period2_label}: {period2_count}")
        else:
            # Run time period comparison
            create_time_period_comparison(
                filtered_df,
                selected_metric,
                period1_range,
                period2_range,
                period1_label,
                period2_label
            )

            # Show top movies from each period
            st.markdown("---")
            st.subheader("Top Movies from Each Period")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"<h4 style='color: #92400E;'>{period1_label} Top Movies</h4>", unsafe_allow_html=True)
                display_top_movies(
                    period1_df,
                    selected_metric,
                    n=5,
                    time_label=period1_label,
                    expanded_view=False
                )

            with col2:
                st.markdown(f"<h4 style='color: #1E40AF;'>{period2_label} Top Movies</h4>", unsafe_allow_html=True)
                display_top_movies(
                    period2_df,
                    selected_metric,
                    n=5,
                    time_label=period2_label,
                    expanded_view=False
                )

    # --- Footer ---
    st.markdown("---")
    st.caption(f"Movie Performance Analysis • Analysis date: {datetime.now().strftime('%B %d, %Y')}")


if __name__ == "__main__":
    main()