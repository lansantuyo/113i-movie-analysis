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
st.set_page_config(layout="wide", page_title="Model Prediction Results",
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
IN_PRODUCTION_FILE = "output/in_prod_predictions.parquet"
POST_PRODUCTION_FILE = "output/post_prod_predictions.parquet"

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

    /* Status indicators */
    .status-production {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 0.2rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: 600;
        display: inline-block;
    }
    .status-post-production {
        background-color: #DCFCE7;
        color: #166534;
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
    """Load movie prediction data with visual loading state"""
    with st.spinner("Loading movie prediction data... Please wait."):
        try:
            # Try to load the parquet files if they exist
            if os.path.exists(IN_PRODUCTION_FILE) and os.path.exists(POST_PRODUCTION_FILE):
                in_prod = pd.read_parquet(IN_PRODUCTION_FILE)
                post_prod = pd.read_parquet(POST_PRODUCTION_FILE)

                # Add a stage column for filtering
                in_prod['stage'] = 'In Production'
                post_prod['stage'] = 'Post-Production'

                return in_prod, post_prod
            else:
                if not os.path.exists(IN_PRODUCTION_FILE):
                    st.error(f"File not found: {IN_PRODUCTION_FILE}")
                if not os.path.exists(POST_PRODUCTION_FILE):
                    st.error(f"File not found: {POST_PRODUCTION_FILE}")
                return None, None
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None, None


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


# --- Visualization Functions ---
def display_top_movies(df, metric, n=10, stage="In Production", expanded_view=False):
    """Display top n movies based on the selected metric"""
    if df is None or df.empty:
        st.warning(f"No data available for {stage} movies.")
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
        st.info(f"No {stage.lower()} movies found with the current filters.")
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
                help="The planned release year"
            ),
            'Month': st.column_config.TextColumn(
                "Release Month",
                help="The planned release month"
            ),
            'Season': st.column_config.TextColumn(
                "Release Season",
                help="The planned release season"
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
                        'mean': f'Avg. Predicted {selected_metric.capitalize()}'
                    },
                    text='count',
                    category_orders={"release_season": season_order},
                    color_discrete_map=season_colors
                )

                # Update layout
                fig.update_layout(
                    xaxis_title='Season',
                    yaxis_title=f'Avg. Predicted {selected_metric.capitalize()}',
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
                            f"<b>Avg. Predicted {selected_metric.capitalize()}:</b> " +
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
                        'mean': f'Avg. Predicted {selected_metric.capitalize()}'
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
                    yaxis_title=f'Avg. Predicted {selected_metric.capitalize()}',
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
                            f"<b>Avg. Predicted {selected_metric.capitalize()}:</b> " +
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
                f"<b>Avg. Predicted {selected_metric.capitalize()}:</b> " +
                ("%{x:$.2f}" if selected_metric == 'revenue' else "%{x:.2f}") +
                "<br><b>Movies:</b> %{text}<extra></extra>"
        )
    ))

    # Update layout
    fig.update_layout(
        title=f"Genre Performance Analysis: {selected_metric.capitalize()}",
        xaxis_title=f"Average Predicted {selected_metric.capitalize()}" + (
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
                f"<b>Avg. Predicted {selected_metric.capitalize()}:</b> " +
                ("%{x:$.2f}" if selected_metric == 'revenue' else "%{x:.2f}") +
                "<br><b>Movies:</b> %{text}<extra></extra>"
        )
    ))

    # Update layout
    fig.update_layout(
        title=f"Top {top_n} Production Companies by {selected_metric.capitalize()}",
        xaxis_title=f"Average Predicted {selected_metric.capitalize()}" + (
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
            metric_col: f'Predicted {selected_metric.capitalize()}' + (' ($)' if selected_metric == 'revenue' else '')
        },
        title=f'Budget vs. Predicted {selected_metric.capitalize()}'
    )

    # Add a trendline
    fig.update_layout(
        height=500,
        plot_bgcolor='white',
        coloraxis_colorbar=dict(title=f'Predicted {selected_metric.capitalize()}'),
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
            f"There is a **{correlation_strength} {correlation_direction}** correlation between budget and predicted {selected_metric}.")

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
                f"Movies with budgets in this range show the strongest predicted {selected_metric} performance.")
        except Exception as e:
            st.error(f"Error calculating optimal budget: {e}")


def create_summary_dashboard(in_prod_df, post_prod_df, selected_metric):
    """Create a summary dashboard with key insights"""
    metric_col = f'predicted_{selected_metric}'

    if ((in_prod_df is None or in_prod_df.empty) and
            (post_prod_df is None or post_prod_df.empty)):
        st.error("No data available for analysis.")
        return

    # Combine dataframes for overall analysis
    all_movies = pd.concat([
        in_prod_df if in_prod_df is not None else pd.DataFrame(),
        post_prod_df if post_prod_df is not None else pd.DataFrame()
    ]).reset_index(drop=True)

    if metric_col not in all_movies.columns:
        st.error(f"Prediction column '{metric_col}' not found in the dataset.")
        return

    # Key metrics
    total_movies = len(all_movies)
    avg_prediction = all_movies[metric_col].mean() if metric_col in all_movies.columns else None
    in_prod_count = len(in_prod_df) if in_prod_df is not None else 0
    post_prod_count = len(post_prod_df) if post_prod_df is not None else 0

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total Movies",
            value=total_movies
        )

    with col2:
        if avg_prediction is not None:
            st.metric(
                label=f"Avg {selected_metric.capitalize()}",
                value=format_value(avg_prediction, selected_metric)
            )

    with col3:
        st.metric(
            label="In Production",
            value=in_prod_count,
            delta=f"{in_prod_count / total_movies * 100:.1f}%" if total_movies > 0 else None
        )

    with col4:
        st.metric(
            label="Post-Production",
            value=post_prod_count,
            delta=f"{post_prod_count / total_movies * 100:.1f}%" if total_movies > 0 else None
        )

    # Style metrics
    style_metric_cards()

    # Overall stats
    st.markdown("---")

    if in_prod_df is not None and not in_prod_df.empty and metric_col in in_prod_df.columns:
        top_in_prod = in_prod_df.nlargest(1, metric_col)
        top_in_prod_title = top_in_prod['title'].iloc[0] if 'title' in top_in_prod.columns else "Unknown"
        top_in_prod_value = top_in_prod[metric_col].iloc[0]
    else:
        top_in_prod_title = "N/A"
        top_in_prod_value = None

    if post_prod_df is not None and not post_prod_df.empty and metric_col in post_prod_df.columns:
        top_post_prod = post_prod_df.nlargest(1, metric_col)
        top_post_prod_title = top_post_prod['title'].iloc[0] if 'title' in top_post_prod.columns else "Unknown"
        top_post_prod_value = top_post_prod[metric_col].iloc[0]
    else:
        top_post_prod_title = "N/A"
        top_post_prod_value = None

    # Top performers cards
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div style="background-color: #DBEAFE; padding: 1rem; border-radius: 0.5rem;">
            <div style="font-size: 0.9rem; color: #1E40AF;">Top In-Production Movie</div>
            <div style="font-size: 1.2rem; font-weight: 600; color: #1E3A8A; margin: 0.5rem 0;">{top_in_prod_title}</div>
            <div style="font-size: 1.5rem; font-weight: 700; color: #1E3A8A;">
                {format_value(top_in_prod_value, selected_metric) if top_in_prod_value is not None else "N/A"}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="background-color: #DCFCE7; padding: 1rem; border-radius: 0.5rem;">
            <div style="font-size: 0.9rem; color: #166534;">Top Post-Production Movie</div>
            <div style="font-size: 1.2rem; font-weight: 600; color: #166534; margin: 0.5rem 0;">{top_post_prod_title}</div>
            <div style="font-size: 1.5rem; font-weight: 700; color: #166534;">
                {format_value(top_post_prod_value, selected_metric) if top_post_prod_value is not None else "N/A"}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Create combined charts for insights
    st.markdown("---")

    # Timing and genre insights
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("When to Release")

        # Combined timing analysis
        if 'release_season' in all_movies.columns and metric_col in all_movies.columns:
            season_data = all_movies.groupby('release_season')[metric_col].mean().reset_index()

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
                        metric_col: f'Predicted {selected_metric.capitalize()}'
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
        genre_cols = [col for col in all_movies.columns if col.startswith('genre_')]

        if genre_cols and metric_col in all_movies.columns:
            genre_data = []

            for genre in genre_cols:
                genre_movies = all_movies[all_movies[genre] == 1]
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
                        'avg_value': f'Predicted {selected_metric.capitalize()}'
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


# --- Main App ---
def main():
    st.markdown('<h1 class="main-header">🎬 Model Prediction Results</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">This dashboard explores movies predicted to be the most successful based on various metrics. Select a success metric and explore top movies in production and post-production stages.</p>',
                unsafe_allow_html=True)

    # Load data
    in_prod_df, post_prod_df = load_data()

    # Show error if no data
    if in_prod_df is None and post_prod_df is None:
        st.error("Failed to load data. Please check the file paths and data formats.")
        st.info("To use this app, you need to save your dataframes with predictions as Parquet files.")

        # Instructions
        with st.expander("How to prepare your data"):
            st.markdown("""
            1. After running your prediction code, save the dataframes with predictions:
            ```python
            in_prod_df.to_parquet('output/in_prod_predictions.parquet')
            post_prod_df.to_parquet('output/post_prod_predictions.parquet')
            ```

            2. Make sure the dataframes contain:
               - Movie titles in a 'title' column
               - Prediction columns starting with 'predicted_'
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
    prediction_cols = []

    if in_prod_df is not None and not in_prod_df.empty:
        prediction_cols.extend([col for col in in_prod_df.columns if col.startswith('predicted_')])

    if post_prod_df is not None and not post_prod_df.empty:
        prediction_cols.extend([col for col in post_prod_df.columns if col.startswith('predicted_')])

    prediction_cols = list(set(prediction_cols))  # Remove duplicates
    available_metrics = [col.replace('predicted_', '') for col in prediction_cols]

    # Default to revenue if available
    default_metric = 'revenue' if 'predicted_revenue' in prediction_cols else available_metrics[
        0] if available_metrics else None

    if not available_metrics:
        st.error(
            "No prediction columns found in the data. Make sure your dataframes contain columns starting with 'predicted_'.")
        st.stop()

    # --- Sidebar Controls ---
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/movie-projector.png", width=80)
        st.title("Analysis Controls")

        # Metric selection
        st.header("1. Choose Metric")

        metric_descriptions = {
            'revenue': '',
            'roi': 'Return on Investment',
            'popularity': 'Audience interest and engagement',
            'vote_count': '',
            'vote_average': 'Average rating score'
        }

        selected_metric = st.selectbox(
            "Measure success by:",
            options=available_metrics,
            index=available_metrics.index(default_metric) if default_metric in available_metrics else 0,
            format_func=lambda x: f"{x.capitalize()}" # ({metric_descriptions.get(x, '')})
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

        # Advanced filters section
        st.markdown("---")
        st.header("3. Filter Movies")

        # Extract all unique genres and companies
        all_genres = extract_all_genres(pd.concat([
            in_prod_df if in_prod_df is not None else pd.DataFrame(),
            post_prod_df if post_prod_df is not None else pd.DataFrame()
        ]))

        all_companies = extract_all_companies(pd.concat([
            in_prod_df if in_prod_df is not None else pd.DataFrame(),
            post_prod_df if post_prod_df is not None else pd.DataFrame()
        ]))

        # Filter by genre
        genre_filter = st.selectbox(
            "Filter by genre:",
            options=['All Genres'] + sorted(all_genres),
            index=0
        )

        # Filter by company (with search)
        company_filter = st.selectbox(
            "Filter by production company:",
            options=['All Companies'] + sorted(all_companies),
            index=0
        )

        # Year range filter (if applicable)
        min_year, max_year = None, None

        year_data = pd.concat([
            in_prod_df[[
                'release_year']] if in_prod_df is not None and 'release_year' in in_prod_df.columns else pd.DataFrame(),
            post_prod_df[[
                'release_year']] if post_prod_df is not None and 'release_year' in post_prod_df.columns else pd.DataFrame()
        ])

        if not year_data.empty:
            min_year_data = int(year_data['release_year'].min())
            max_year_data = int(year_data['release_year'].max())

            year_range = st.slider(
                "Release year range:",
                min_value=min_year_data,
                max_value=max_year_data,
                value=(min_year_data, max_year_data)
            )

            min_year, max_year = year_range

        # Season filter
        season_filter = st.selectbox(
            "Filter by season:",
            options=['All Seasons', 'Spring', 'Summer', 'Fall', 'Winter'],
            index=0
        )

        # Budget range filter (if applicable)
        min_budget, max_budget = None, None

        budget_data = pd.concat([
            in_prod_df[['budget']] if in_prod_df is not None and 'budget' in in_prod_df.columns else pd.DataFrame(),
            post_prod_df[
                ['budget']] if post_prod_df is not None and 'budget' in post_prod_df.columns else pd.DataFrame()
        ])

        if not budget_data.empty:
            min_budget_data = float(budget_data['budget'].min())
            max_budget_data = float(budget_data['budget'].max())

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

        # Apply filters to dataframes
        filtered_in_prod = filter_dataframe(in_prod_df, filters) if in_prod_df is not None else None
        filtered_post_prod = filter_dataframe(post_prod_df, filters) if post_prod_df is not None else None

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
        "📈 Comparisons"
    ])

    with tab1:
        # Overview dashboard with key insights
        colored_header(
            label="Prediction Overview",
            description=f"Key insights based on {selected_metric} predictions",
            color_name="blue-70"
        )

        # Summary dashboard
        create_summary_dashboard(filtered_in_prod, filtered_post_prod, selected_metric)

        # Show data summary
        st.markdown("---")
        st.subheader("Data Summary")

        col1, col2 = st.columns(2)

        with col1:
            if filtered_in_prod is not None and not filtered_in_prod.empty:
                st.markdown(f"**In Production Movies:** {len(filtered_in_prod)}")
                if 'release_year' in filtered_in_prod.columns:
                    years = filtered_in_prod['release_year'].unique()
                    years = sorted([y for y in years if not pd.isna(y)])
                    st.markdown(f"**Planned Release Years:** {', '.join(map(str, years))}")
            else:
                st.markdown("**No in-production movies match the current filters.**")

        with col2:
            if filtered_post_prod is not None and not filtered_post_prod.empty:
                st.markdown(f"**Post-Production Movies:** {len(filtered_post_prod)}")
                if 'release_year' in filtered_post_prod.columns:
                    years = filtered_post_prod['release_year'].unique()
                    years = sorted([y for y in years if not pd.isna(y)])
                    st.markdown(f"**Planned Release Years:** {', '.join(map(str, years))}")
            else:
                st.markdown("**No post-production movies match the current filters.**")

    with tab2:
        colored_header(
            label="Top Predicted Movies",
            description=f"Movies with highest predicted {selected_metric}",
            color_name="blue-70"
        )

        # Top Movies view with two columns
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <div style="background-color: #FEF3C7; padding: 0.5rem; border-radius: 0.25rem; margin-bottom: 1rem;">
                <div style="font-weight: 600; color: #92400E;">🎬 In Production Movies</div>
            </div>
            """, unsafe_allow_html=True)

            top_in_prod = display_top_movies(
                filtered_in_prod,
                selected_metric,
                n=top_n,
                stage="In Production",
                expanded_view=expanded_view
            )

        with col2:
            st.markdown("""
            <div style="background-color: #DCFCE7; padding: 0.5rem; border-radius: 0.25rem; margin-bottom: 1rem;">
                <div style="font-weight: 600; color: #166534;">🎥 Post-Production Movies</div>
            </div>
            """, unsafe_allow_html=True)

            top_post_prod = display_top_movies(
                filtered_post_prod,
                selected_metric,
                n=top_n,
                stage="Post-Production",
                expanded_view=expanded_view
            )

    with tab3:
        colored_header(
            label=f"Detailed Analysis: {selected_metric.capitalize()}",
            description="Explore factors that impact movie success predictions",
            color_name="blue-70"
        )

        # Create analysis tabs for production stages
        stage_tab1, stage_tab2 = st.tabs([
            "In Production Movies",
            "Post-Production Movies"
        ])

        with stage_tab1:
            if filtered_in_prod is None or filtered_in_prod.empty:
                st.info("No in-production movies match the current filters.")
            else:
                # Time-based analysis
                create_time_analysis(filtered_in_prod, selected_metric)

                st.markdown("---")

                # Genre and Company Analysis
                col1, col2 = st.columns(2)

                with col1:
                    create_genre_analysis(filtered_in_prod, selected_metric)

                with col2:
                    create_company_analysis(filtered_in_prod, selected_metric)

                st.markdown("---")

                # Budget analysis
                create_budget_analysis(filtered_in_prod, selected_metric)

        with stage_tab2:
            if filtered_post_prod is None or filtered_post_prod.empty:
                st.info("No post-production movies match the current filters.")
            else:
                # Time-based analysis
                create_time_analysis(filtered_post_prod, selected_metric)

                st.markdown("---")

                # Genre and Company Analysis
                col1, col2 = st.columns(2)

                with col1:
                    create_genre_analysis(filtered_post_prod, selected_metric)

                with col2:
                    create_company_analysis(filtered_post_prod, selected_metric)

                st.markdown("---")

                # Budget analysis
                create_budget_analysis(filtered_post_prod, selected_metric)

    with tab4:
        colored_header(
            label="Stage Comparison",
            description="How do in-production and post-production movies compare?",
            color_name="blue-70"
        )

        if ((filtered_in_prod is None or filtered_in_prod.empty) and
                (filtered_post_prod is None or filtered_post_prod.empty)):
            st.info("Insufficient data to make comparisons. Please adjust your filters.")
        else:
            # Prepare data for comparisons
            metric_col = f'predicted_{selected_metric}'

            in_prod_mean = filtered_in_prod[
                metric_col].mean() if filtered_in_prod is not None and not filtered_in_prod.empty and metric_col in filtered_in_prod.columns else None
            post_prod_mean = filtered_post_prod[
                metric_col].mean() if filtered_post_prod is not None and not filtered_post_prod.empty and metric_col in filtered_post_prod.columns else None

            # Display comparison metrics
            col1, col2, col3 = st.columns([1, 2, 1])

            with col2:
                if in_prod_mean is not None and post_prod_mean is not None:
                    diff_pct = (post_prod_mean - in_prod_mean) / in_prod_mean * 100 if in_prod_mean != 0 else 0

                    st.markdown(f"""
                    <div style="background-color: white; padding: 1rem; border-radius: 0.5rem; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); text-align: center; margin-bottom: 1rem;">
                        <div style="font-size: 1rem; color: #4B5563;">Average Predicted {selected_metric.capitalize()}</div>
                        <div style="display: flex; justify-content: space-around; margin: 1rem 0;">
                            <div>
                                <div style="font-size: 0.9rem; color: #92400E;">In Production</div>
                                <div style="font-size: 1.5rem; font-weight: 700; color: #92400E;">{format_value(in_prod_mean, selected_metric)}</div>
                            </div>
                            <div style="font-size: 1.5rem; font-weight: 700; color: #6B7280;">vs.</div>
                            <div>
                                <div style="font-size: 0.9rem; color: #166534;">Post-Production</div>
                                <div style="font-size: 1.5rem; font-weight: 700; color: #166534;">{format_value(post_prod_mean, selected_metric)}</div>
                            </div>
                        </div>
                        <div style="font-size: 0.9rem; color: #4B5563;">
                            Post-production movies are predicted to be <span style="font-weight: 600; color: {'#16A34A' if diff_pct >= 0 else '#EF4444'};">
                                {abs(diff_pct):.1f}% {'higher' if diff_pct >= 0 else 'lower'}
                            </span> than in-production movies.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # Create comparison charts

            # 1. Genre comparison
            st.subheader("Genre Performance by Production Stage")

            # Get genre data for each stage
            in_prod_genres = []
            post_prod_genres = []

            genre_cols = [col for col in filtered_in_prod.columns if
                          col.startswith('genre_')] if filtered_in_prod is not None else []

            for genre in genre_cols:
                genre_name = genre.replace('genre_', '')

                # In Production stats
                if filtered_in_prod is not None and not filtered_in_prod.empty:
                    genre_movies = filtered_in_prod[filtered_in_prod[genre] == 1]
                    if len(genre_movies) >= MIN_SAMPLE_SIZE:
                        in_prod_genres.append({
                            'genre': genre_name,
                            'avg_value': genre_movies[metric_col].mean() if metric_col in genre_movies.columns else 0,
                            'count': len(genre_movies),
                            'stage': 'In Production'
                        })

                # Post Production stats
                if filtered_post_prod is not None and not filtered_post_prod.empty:
                    genre_movies = filtered_post_prod[filtered_post_prod[genre] == 1]
                    if len(genre_movies) >= MIN_SAMPLE_SIZE:
                        post_prod_genres.append({
                            'genre': genre_name,
                            'avg_value': genre_movies[metric_col].mean() if metric_col in genre_movies.columns else 0,
                            'count': len(genre_movies),
                            'stage': 'Post-Production'
                        })

            # Combine genre data
            all_genres = in_prod_genres + post_prod_genres

            if all_genres:
                genres_df = pd.DataFrame(all_genres)

                # Get genres that exist in both stages
                common_genres = set(g['genre'] for g in in_prod_genres) & set(g['genre'] for g in post_prod_genres)

                if common_genres:
                    # Filter to common genres for fair comparison
                    genres_df = genres_df[genres_df['genre'].isin(common_genres)]

                    # Create grouped bar chart
                    fig = px.bar(
                        genres_df,
                        x='genre',
                        y='avg_value',
                        color='stage',
                        barmode='group',
                        color_discrete_map={
                            'In Production': '#FCD34D',
                            'Post-Production': '#4ADE80'
                        },
                        labels={
                            'genre': 'Genre',
                            'avg_value': f'Avg. Predicted {selected_metric.capitalize()}',
                            'stage': 'Production Stage'
                        }
                    )

                    fig.update_layout(
                        height=400,
                        plot_bgcolor='white',
                        margin=dict(l=20, r=20, t=40, b=20)
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Identify genres with biggest differences
                    genres_wide = genres_df.pivot(index='genre', columns='stage', values='avg_value').reset_index()
                    genres_wide['difference'] = genres_wide['Post-Production'] - genres_wide['In Production']
                    genres_wide['difference_pct'] = genres_wide['difference'] / genres_wide['In Production'] * 100

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
                                    Post-production {diff_pct:.1f}% higher than in-production
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
                                    Post-production {abs(diff_pct):.1f}% lower than in-production
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.info("No common genres between production stages with enough data for comparison.")
            else:
                st.info("Not enough genre data available for comparison.")

            # 2. Budget to performance correlation comparison
            st.markdown("---")
            st.subheader("Budget Impact by Production Stage")

            if (filtered_in_prod is not None and 'budget' in filtered_in_prod.columns and
                    filtered_post_prod is not None and 'budget' in filtered_post_prod.columns):

                # Create scatter plot with both stages
                in_prod_budget = filtered_in_prod.dropna(subset=['budget', metric_col])
                post_prod_budget = filtered_post_prod.dropna(subset=['budget', metric_col])

                if not in_prod_budget.empty and not post_prod_budget.empty:
                    # Add stage column to each dataframe
                    in_prod_budget = in_prod_budget.copy()
                    post_prod_budget = post_prod_budget.copy()

                    # Combine for visualization
                    combined_budget = pd.concat([in_prod_budget, post_prod_budget])

                    # Create scatter plot
                    fig = px.scatter(
                        combined_budget,
                        x='budget',
                        y=metric_col,
                        color='stage',
                        hover_name='title',
                        log_x=True,
                        log_y=selected_metric == 'revenue',
                        color_discrete_map={
                            'In Production': '#FCD34D',
                            'Post-Production': '#4ADE80'
                        },
                        labels={
                            'budget': 'Budget ($)',
                            metric_col: f'Predicted {selected_metric.capitalize()}',
                            'stage': 'Production Stage'
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
                    in_prod_corr = in_prod_budget[['budget', metric_col]].corr().iloc[0, 1]
                    post_prod_corr = post_prod_budget[['budget', metric_col]].corr().iloc[0, 1]

                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric(
                            label="In Production Budget Correlation",
                            value=f"{in_prod_corr:.2f}"
                        )

                    with col2:
                        st.metric(
                            label="Post-Production Budget Correlation",
                            value=f"{post_prod_corr:.2f}"
                        )

                    # Interpretation
                    corr_diff = post_prod_corr - in_prod_corr

                    st.markdown(f"""
                    <div style="background-color: #F3F4F6; padding: 1rem; border-radius: 0.5rem; margin-top: 1rem;">
                        <div style="font-weight: 600; color: #1E3A8A; margin-bottom: 0.5rem;">Budget Impact Interpretation</div>
                        <div>
                            The correlation between budget and predicted {selected_metric} is 
                            <span style="font-weight: 600; color: {'#16A34A' if corr_diff > 0 else '#EF4444'};">
                                {abs(corr_diff):.2f} points {'stronger' if corr_diff > 0 else 'weaker'}
                            </span> 
                            for post-production movies compared to in-production movies.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("Not enough budget data for both stages to make a comparison.")
            else:
                st.info("Budget information not available for comparison.")


if __name__ == "__main__":
    main()
