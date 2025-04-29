import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import calendar
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io

# --- Configuration ---
IN_PRODUCTION_FILE = "output/in_prod_predictions.parquet"
POST_PRODUCTION_FILE = "output/post_prod_predictions.parquet"

MIN_SAMPLE_SIZE = 5  # Minimum movies needed for aggregation

# --- Page Config ---
st.set_page_config(layout="wide", page_title="Model Prediction Results",
                   page_icon="🎬", initial_sidebar_state="collapsed")

# --- Custom CSS ---
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
    }
    .highlight-text {
        color: #FF4B4B;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 5px 5px 0px 0px;
        padding-left: 20px;
        padding-right: 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ff4b4b30;
    }
    .st-emotion-cache-1629p8f {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# --- Data Loading (Cached) ---
@st.cache_data
def load_data():
    try:
        # Try to load the parquet files if they exist
        if os.path.exists(IN_PRODUCTION_FILE) and os.path.exists(POST_PRODUCTION_FILE):
            in_prod = pd.read_parquet(IN_PRODUCTION_FILE)
            post_prod = pd.read_parquet(POST_PRODUCTION_FILE)

            # Log successful load
            st.sidebar.success(f"✅ Successfully loaded data from parquet files")
            st.sidebar.info(f"📊 Found {len(in_prod)} in-production and {len(post_prod)} post-production movies")

            return in_prod, post_prod

        else:
            st.sidebar.warning("⚠️ Data files not found. Creating sample data for demonstration.")
            return create_demo_data()

    except Exception as e:
        st.sidebar.error(f"❌ Error loading data: {e}")
        st.sidebar.warning("⚠️ Falling back to sample data")
        return create_demo_data()


def create_demo_data():
    """Create sample data for demonstration purposes"""
    # Generate current date info for realistic future releases
    current_year = datetime.now().year

    # Create a more diverse set of sample data
    genres = ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary',
              'Drama', 'Family', 'Fantasy', 'Horror', 'Mystery', 'Romance',
              'SciFi', 'Thriller', 'Western']

    companies = ['Universal', 'Disney', 'Warner', 'Sony', 'Paramount',
                 'Netflix', 'Amazon', 'A24', 'Lionsgate', 'Focus',
                 'Pixar', 'DreamWorks', 'MGM', 'Amblin', 'Blumhouse']

    # Generate more sample data for in_prod (50 movies)
    np.random.seed(42)  # For reproducibility

    # Generate 50 sample movies for in-production
    movie_count = 50

    # Base properties
    titles = [f"Sample Movie IP-{i + 1}" for i in range(movie_count)]
    revenues = np.random.lognormal(18, 1, movie_count)  # Log-normal distribution for realistic revenue
    popularities = np.random.normal(7, 2, movie_count)  # Normal distribution for popularity
    vote_avgs = np.random.normal(7, 1, movie_count)  # Normal distribution for vote average
    vote_counts = np.random.lognormal(8, 1, movie_count)  # Log-normal for vote counts
    rois = np.random.normal(2.5, 1, movie_count)  # Normal for ROI
    budgets = revenues / (rois + np.random.normal(0, 0.5, movie_count))  # Budget based on revenue and ROI

    # Time properties
    years = np.random.choice([current_year, current_year + 1], movie_count, p=[0.7, 0.3])
    months = np.random.randint(1, 13, movie_count)
    seasons = [get_season(m) for m in months]

    # Create in_prod dataframe
    in_prod = pd.DataFrame({
        'title': titles,
        'predicted_revenue': revenues,
        'predicted_popularity': popularities,
        'predicted_vote_average': np.clip(vote_avgs, 1, 10),  # Clip to valid range
        'predicted_vote_count': vote_counts,
        'predicted_roi': np.clip(rois, 0.1, 10),  # Clip to reasonable ROI range
        'budget': budgets,
        'release_year': years,
        'release_month': months,
        'release_season': seasons,
    })

    # Generate post-production data (30 movies)
    movie_count_post = 30

    # Base properties - slightly different distributions
    titles_post = [f"Sample Movie PP-{i + 1}" for i in range(movie_count_post)]
    revenues_post = np.random.lognormal(18.2, 0.9, movie_count_post)
    popularities_post = np.random.normal(7.2, 1.8, movie_count_post)
    vote_avgs_post = np.random.normal(7.1, 0.9, movie_count_post)
    vote_counts_post = np.random.lognormal(8.1, 0.9, movie_count_post)
    rois_post = np.random.normal(2.7, 0.9, movie_count_post)
    budgets_post = revenues_post / (rois_post + np.random.normal(0, 0.4, movie_count_post))

    # Time properties - post-production movies are releasing sooner
    years_post = np.random.choice([current_year, current_year + 1], movie_count_post, p=[0.9, 0.1])
    months_post = np.random.randint(1, 13, movie_count_post)
    seasons_post = [get_season(m) for m in months_post]

    # Create post_prod dataframe
    post_prod = pd.DataFrame({
        'title': titles_post,
        'predicted_revenue': revenues_post,
        'predicted_popularity': popularities_post,
        'predicted_vote_average': np.clip(vote_avgs_post, 1, 10),
        'predicted_vote_count': vote_counts_post,
        'predicted_roi': np.clip(rois_post, 0.1, 10),
        'budget': budgets_post,
        'release_year': years_post,
        'release_month': months_post,
        'release_season': seasons_post,
    })

    # Add genre columns to both dataframes
    for genre in genres:
        # Each movie has a 20% chance of having each genre, ensuring multiple genres per movie
        in_prod[f'genre_{genre}'] = np.random.choice([0, 1], movie_count, p=[0.8, 0.2])
        post_prod[f'genre_{genre}'] = np.random.choice([0, 1], movie_count_post, p=[0.8, 0.2])

        # Ensure at least one genre per movie
        has_no_genre_in = (in_prod[[f'genre_{g}' for g in genres]].sum(axis=1) == 0)
        has_no_genre_post = (post_prod[[f'genre_{g}' for g in genres]].sum(axis=1) == 0)

        if has_no_genre_in.any():
            random_genre = np.random.choice(genres, size=has_no_genre_in.sum())
            for i, idx in enumerate(in_prod[has_no_genre_in].index):
                in_prod.loc[idx, f'genre_{random_genre[i]}'] = 1

        if has_no_genre_post.any():
            random_genre = np.random.choice(genres, size=has_no_genre_post.sum())
            for i, idx in enumerate(post_prod[has_no_genre_post].index):
                post_prod.loc[idx, f'genre_{random_genre[i]}'] = 1

    # Add production company columns to both dataframes
    for company in companies:
        # Each movie has a 15% chance of being produced by each company
        in_prod[f'production_companies_{company}'] = np.random.choice([0, 1], movie_count, p=[0.85, 0.15])
        post_prod[f'production_companies_{company}'] = np.random.choice([0, 1], movie_count_post, p=[0.85, 0.15])

        # Ensure at least one production company per movie
        has_no_company_in = (in_prod[[f'production_companies_{c}' for c in companies]].sum(axis=1) == 0)
        has_no_company_post = (post_prod[[f'production_companies_{c}' for c in companies]].sum(axis=1) == 0)

        if has_no_company_in.any():
            random_company = np.random.choice(companies, size=has_no_company_in.sum())
            for i, idx in enumerate(in_prod[has_no_company_in].index):
                in_prod.loc[idx, f'production_companies_{random_company[i]}'] = 1

        if has_no_company_post.any():
            random_company = np.random.choice(companies, size=has_no_company_post.sum())
            for i, idx in enumerate(post_prod[has_no_company_post].index):
                post_prod.loc[idx, f'production_companies_{random_company[i]}'] = 1

    return in_prod, post_prod


def get_season(month):
    """Convert month to season"""
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'


# --- Helper Functions ---
def get_production_companies(row):
    """Extract production companies from a row"""
    companies = []
    for col in row.index:
        if col.startswith('production_companies_') and row[col] == 1:
            companies.append(col.replace('production_companies_', ''))
    return ', '.join(companies) if companies else '—'


def get_genres(row):
    """Extract genres from a row"""
    genres = []
    for col in row.index:
        if col.startswith('genre_') and row[col] == 1:
            genres.append(col.replace('genre_', ''))
    return ', '.join(genres) if genres else '—'


def format_currency(value):
    """Format large currency values with appropriate suffixes"""
    if value >= 1_000_000_000:
        return f"${value / 1_000_000_000:.1f}B"
    elif value >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    elif value >= 1_000:
        return f"${value / 1_000:.1f}K"
    else:
        return f"${value:.0f}"


def display_key_metrics(df, metric):
    """Display key metrics in a visually appealing way"""
    if df is None or df.empty or f'predicted_{metric}' not in df.columns:
        st.warning("Insufficient data to display key metrics.")
        return

    col1, col2, col3, col4 = st.columns(4)

    # Average predicted value
    avg_value = df[f'predicted_{metric}'].mean()
    if metric == 'revenue':
        formatted_avg = format_currency(avg_value)
    elif metric == 'roi':
        formatted_avg = f"{avg_value:.2f}x"
    else:
        formatted_avg = f"{avg_value:.2f}"
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Average {metric.capitalize()}</h4>
            <h2>{formatted_avg}</h2>
        </div>
        """, unsafe_allow_html=True)

    # Best season
    if 'release_season' in df.columns:
        season_data = df.groupby('release_season')[f'predicted_{metric}'].mean()
        best_season = season_data.idxmax() if not season_data.empty else "—"
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Best Season</h4>
                <h2>{best_season}</h2>
                <small>Avg {metric.capitalize()}: {format_metric_value(season_data.max(), metric)}</small>
            </div>
            """, unsafe_allow_html=True)

    # Best month
    if 'release_month' in df.columns:
        month_data = df.groupby('release_month')[f'predicted_{metric}'].mean()
        best_month_num = month_data.idxmax() if not month_data.empty else 0
        best_month = calendar.month_name[best_month_num] if 1 <= best_month_num <= 12 else "—"
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Best Month</h4>
                <h2>{best_month}</h2>
                <small>Avg {metric.capitalize()}: {format_metric_value(month_data.max(), metric)}</small>
            </div>
            """, unsafe_allow_html=True)

    # Best genre
    genre_cols = [col for col in df.columns if col.startswith('genre_')]
    if genre_cols:
        genre_data = []
        for genre in genre_cols:
            genre_movies = df[df[genre] == 1]
            if len(genre_movies) >= MIN_SAMPLE_SIZE:
                avg_value = genre_movies[f'predicted_{metric}'].mean()
                genre_name = genre.replace('genre_', '')
                genre_data.append((genre_name, avg_value, len(genre_movies)))

        if genre_data:
            genre_data.sort(key=lambda x: x[1], reverse=True)
            best_genre, best_genre_avg, count = genre_data[0]
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Top Genre</h4>
                    <h2>{best_genre}</h2>
                    <small>Avg {metric.capitalize()}: {format_metric_value(best_genre_avg, metric)} ({count} movies)</small>
                </div>
                """, unsafe_allow_html=True)


def format_metric_value(value, metric):
    """Format metric values appropriately"""
    if metric == 'revenue':
        return format_currency(value)
    elif metric == 'roi':
        return f"{value:.2f}x"
    else:
        return f"{value:.2f}"


def display_top_movies(df, metric, n=10, stage="In Production", allow_filtering=False):
    """Display top n movies based on the selected metric"""
    if df is None or df.empty:
        st.warning(f"No data available for {stage} movies.")
        return

    # Create a clean display dataframe
    if f'predicted_{metric}' not in df.columns:
        st.error(f"Prediction column 'predicted_{metric}' not found in the dataset.")
        return

    # Apply filters if allowed
    if allow_filtering:
        # Genre filter
        genre_cols = [col for col in df.columns if col.startswith('genre_')]
        if genre_cols:
            genres = [col.replace('genre_', '') for col in genre_cols]
            selected_genres = st.multiselect("Filter by genres:", options=genres)

            if selected_genres:
                # Filter for movies that have ANY of the selected genres
                genre_filter = df[genre_cols].copy()
                genre_filter.columns = [col.replace('genre_', '') for col in genre_filter.columns]
                df = df[genre_filter[selected_genres].any(axis=1)]

                if df.empty:
                    st.warning("No movies match the selected genre filters.")
                    return

        # Year filter
        if 'release_year' in df.columns:
            years = sorted(df['release_year'].unique())
            selected_years = st.multiselect("Filter by release years:", options=years)

            if selected_years:
                df = df[df['release_year'].isin(selected_years)]

                if df.empty:
                    st.warning("No movies match the selected year filters.")
                    return

        # Season filter
        if 'release_season' in df.columns:
            seasons = sorted(df['release_season'].unique())
            selected_seasons = st.multiselect("Filter by seasons:", options=seasons)

            if selected_seasons:
                df = df[df['release_season'].isin(selected_seasons)]

                if df.empty:
                    st.warning("No movies match the selected season filters.")
                    return

    # Sort by the prediction metric
    df_sorted = df.sort_values(f'predicted_{metric}', ascending=False).head(n)

    # Add production companies and genres as columns
    df_sorted['Production Companies'] = df_sorted.apply(get_production_companies, axis=1)
    df_sorted['Genres'] = df_sorted.apply(get_genres, axis=1)

    # Select and rename columns for display
    display_cols = [
        'title', f'predicted_{metric}', 'budget',
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
        f'predicted_{metric}': f'Predicted {metric.capitalize()}',
        'budget': 'Budget',
        'release_year': 'Year',
        'release_season': 'Season',
        'release_month': 'Month'
    }
    df_display = df_display.rename(columns=rename_map)

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

    if 'Month' in df_display.columns:
        df_display['Month'] = df_display['Month'].apply(
            lambda x: calendar.month_name[x] if isinstance(x, (int, float)) and 1 <= x <= 12 else x)

    # Display the dataframe with a download button
    st.dataframe(df_display.style.format(format_dict).highlight_max(subset=[f'Predicted {metric.capitalize()}'],
                                                                    color='lightgreen'),
                 use_container_width=True)


    return df_sorted


def create_time_analysis(df, metric):
    """Create visualizations for time-based analysis"""
    col1, col2 = st.columns(2)

    # Season analysis
    with col1:
        st.subheader("Performance by Season")

        if 'release_season' in df.columns:
            season_data = df.groupby('release_season')[f'predicted_{metric}'].agg(['mean', 'count']).reset_index()

            if not season_data.empty:
                season_order = ['Winter', 'Spring', 'Summer', 'Fall']

                # Create a Plotly bar chart
                fig = px.bar(
                    season_data,
                    x='release_season',
                    y='mean',
                    color='release_season',
                    labels={
                        'release_season': 'Season',
                        'mean': f'Avg. Predicted {metric.capitalize()}'
                    },
                    text='count',
                    category_orders={"release_season": season_order},
                    color_discrete_sequence=px.colors.qualitative.Set1
                )

                # Update layout
                fig.update_layout(
                    title=f'Average Predicted {metric.capitalize()} by Season',
                    xaxis_title='Season',
                    yaxis_title=f'Avg. Predicted {metric.capitalize()}',
                    showlegend=False,
                    height=400
                )

                # Add count labels
                fig.update_traces(texttemplate='%{text} movies', textposition='outside')

                # Display the chart
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough season data for analysis.")
        else:
            st.info("No season information found in the dataset.")

    # Month analysis
    with col2:
        st.subheader("Performance by Month")

        if 'release_month' in df.columns:
            month_data = df.groupby('release_month')[f'predicted_{metric}'].agg(['mean', 'count']).reset_index()

            if not month_data.empty:
                # Add month names
                month_data['month_name'] = month_data['release_month'].apply(
                    lambda x: calendar.month_name[int(x)] if isinstance(x, (int, float)) and 1 <= x <= 12 else str(x)
                )

                # Create a Plotly bar chart
                fig = px.bar(
                    month_data,
                    x='release_month',
                    y='mean',
                    color='month_name',
                    labels={
                        'release_month': 'Month',
                        'mean': f'Avg. Predicted {metric.capitalize()}',
                        'month_name': 'Month Name'
                    },
                    text='count',
                    color_discrete_sequence=px.colors.qualitative.Plotly
                )

                # Update layout
                fig.update_layout(
                    title=f'Average Predicted {metric.capitalize()} by Month',
                    xaxis_title='Month',
                    xaxis=dict(
                        tickmode='array',
                        tickvals=list(range(1, 13)),
                        ticktext=[calendar.month_abbr[i] for i in range(1, 13)]
                    ),
                    yaxis_title=f'Avg. Predicted {metric.capitalize()}',
                    showlegend=False,
                    height=400
                )

                # Add count labels
                fig.update_traces(texttemplate='%{text} movies', textposition='outside')

                # Display the chart
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough month data for analysis.")
        else:
            st.info("No month information found in the dataset.")


def create_genre_analysis(df, metric):
    """Create visualizations for genre analysis"""
    st.subheader("Performance by Genre")

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
            avg_value = genre_movies[f'predicted_{metric}'].mean()
            genre_name = genre.replace('genre_', '')
            genre_data.append({
                'genre': genre_name,
                'avg_value': avg_value,
                'count': len(genre_movies)
            })

    if genre_data:
        genre_df = pd.DataFrame(genre_data).sort_values('avg_value', ascending=False)

        # Create a Plotly horizontal bar chart
        fig = px.bar(
            genre_df,
            y='genre',
            x='avg_value',
            orientation='h',
            color='genre',
            labels={
                'genre': 'Genre',
                'avg_value': f'Avg. Predicted {metric.capitalize()}',
                'count': 'Number of Movies'
            },
            text='count',
            color_discrete_sequence=px.colors.qualitative.Bold,
            hover_data=['count']
        )

        # Update layout
        fig.update_layout(
            title=f'Average Predicted {metric.capitalize()} by Genre',
            xaxis_title=f'Avg. Predicted {metric.capitalize()}',
            yaxis_title='Genre',
            showlegend=False,
            height=500,
            yaxis={'categoryorder': 'total ascending'}
        )

        # Add count labels
        fig.update_traces(texttemplate='%{text} movies', textposition='outside')

        # Display the chart
        st.plotly_chart(fig, use_container_width=True)

        # Also show a bubble chart of genres vs budget (if available)
        if 'budget' in df.columns:
            # Calculate average budget for each genre
            budget_data = []
            for genre in genre_cols:
                genre_movies = df[df[genre] == 1]
                if len(genre_movies) >= MIN_SAMPLE_SIZE:
                    avg_metric = genre_movies[f'predicted_{metric}'].mean()
                    avg_budget = genre_movies['budget'].mean()
                    genre_name = genre.replace('genre_', '')
                    budget_data.append({
                        'genre': genre_name,
                        'avg_metric': avg_metric,
                        'avg_budget': avg_budget,
                        'count': len(genre_movies)
                    })

            if budget_data:
                budget_df = pd.DataFrame(budget_data)

                # Create a bubble chart
                fig = px.scatter(
                    budget_df,
                    x='avg_budget',
                    y='avg_metric',
                    size='count',
                    color='genre',
                    text='genre',
                    labels={
                        'avg_budget': 'Average Budget',
                        'avg_metric': f'Avg. Predicted {metric.capitalize()}',
                        'count': 'Number of Movies',
                        'genre': 'Genre'
                    },
                    size_max=50,
                    color_discrete_sequence=px.colors.qualitative.Bold
                )

                # Update layout
                fig.update_layout(
                    title=f'Budget vs. {metric.capitalize()} by Genre',
                    xaxis_title='Average Budget ($)',
                    yaxis_title=f'Avg. Predicted {metric.capitalize()}',
                    height=500
                )

                # Format x-axis to currency
                fig.update_xaxes(tickprefix='$', tickformat=',.0f')

                # Display the chart
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough data per genre for analysis.")


def create_company_analysis(df, metric):
    """Create visualizations for production company analysis"""
    st.subheader("Performance by Production Company")

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
            avg_value = company_movies[f'predicted_{metric}'].mean()
            company_name = company.replace('production_companies_', '')
            company_data.append({
                'company': company_name,
                'avg_value': avg_value,
                'count': len(company_movies)
            })

    if company_data:
        company_df = pd.DataFrame(company_data).sort_values('avg_value', ascending=False)

        # Get top 10 companies for display
        top_companies = company_df.head(10)

        # Create a Plotly horizontal bar chart
        fig = px.bar(
            top_companies,
            y='company',
            x='avg_value',
            orientation='h',
            color='company',
            labels={
                'company': 'Production Company',
                'avg_value': f'Avg. Predicted {metric.capitalize()}',
                'count': 'Number of Movies'
            },
            text='count',
            color_discrete_sequence=px.colors.qualitative.Vivid,
            hover_data=['count']
        )

        # Update layout
        fig.update_layout(
            title=f'Average Predicted {metric.capitalize()} by Production Company (Top 10)',
            xaxis_title=f'Avg. Predicted {metric.capitalize()}',
            yaxis_title='Production Company',
            showlegend=False,
            height=500,
            yaxis={'categoryorder': 'total ascending'}
        )

        # Add count labels
        fig.update_traces(texttemplate='%{text} movies', textposition='outside')

        # Display the chart
        st.plotly_chart(fig, use_container_width=True)

        # Show company performance by genre
        st.subheader("Company Performance by Genre")

        # Get genre columns
        genre_cols = [col for col in df.columns if col.startswith('genre_')]

        if genre_cols and len(top_companies) > 1:
            # For each top production company
            analysis_data = []

            for _, company_row in top_companies.iterrows():
                company_name = company_row['company']
                company_col = f'production_companies_{company_name}'

                # For each genre, get the average metric
                for genre_col in genre_cols:
                    genre_name = genre_col.replace('genre_', '')
                    # Movies from this company with this genre
                    subset = df[(df[company_col] == 1) & (df[genre_col] == 1)]

                    if len(subset) >= 3:  # Only include if at least 3 movies
                        analysis_data.append({
                            'company': company_name,
                            'genre': genre_name,
                            'avg_value': subset[f'predicted_{metric}'].mean(),
                            'count': len(subset)
                        })

            if analysis_data:
                analysis_df = pd.DataFrame(analysis_data)

                # Create a heatmap
                heatmap_pivot = analysis_df.pivot_table(
                    values='avg_value',
                    index='company',
                    columns='genre',
                    aggfunc='mean'
                ).fillna(0)

                # Get count pivot for annotation
                count_pivot = analysis_df.pivot_table(
                    values='count',
                    index='company',
                    columns='genre',
                    aggfunc='sum'
                ).fillna(0)

                # Create the heatmap
                fig = go.Figure(data=go.Heatmap(
                    z=heatmap_pivot.values,
                    x=heatmap_pivot.columns,
                    y=heatmap_pivot.index,
                    colorscale='YlOrRd',
                    hoverongaps=False,
                    hovertemplate='Company: %{y}<br>Genre: %{x}<br>Avg Value: %{z:.2f}<extra></extra>'
                ))

                # Update layout
                fig.update_layout(
                    title=f'Company Performance by Genre (Avg. Predicted {metric.capitalize()})',
                    xaxis_title='Genre',
                    yaxis_title='Production Company',
                    height=500
                )

                # Add count annotations
                annotations = []
                for i, company in enumerate(heatmap_pivot.index):
                    for j, genre in enumerate(heatmap_pivot.columns):
                        count = count_pivot.loc[company, genre]
                        if count > 0:
                            annotations.append(dict(
                                x=genre,
                                y=company,
                                text=str(int(count)),
                                showarrow=False,
                                font=dict(color='black' if heatmap_pivot.iloc[
                                                               i, j] < heatmap_pivot.values.max() / 1.5 else 'white')
                            ))

                fig.update_layout(annotations=annotations)

                # Display the heatmap
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Numbers in cells represent the count of movies in each company-genre combination.")
            else:
                st.info("Not enough data for company-genre analysis.")
    else:
        st.info("Not enough data per production company for analysis.")


def create_budget_analysis(df, metric):
    """Create visualizations for budget analysis"""
    if 'budget' not in df.columns:
        st.info("Budget information not available in the dataset.")
        return

    st.subheader("Budget vs. Predicted Success")

    # Create a Plotly scatter plot
    fig = px.scatter(
        df,
        x='budget',
        y=f'predicted_{metric}',
        color='release_season' if 'release_season' in df.columns else None,
        size='predicted_popularity' if 'predicted_popularity' in df.columns else None,
        hover_name='title',
        labels={
            'budget': 'Budget ($)',
            f'predicted_{metric}': f'Predicted {metric.capitalize()}',
            'release_season': 'Season',
            'predicted_popularity': 'Predicted Popularity'
        },
        log_x=True,
        log_y=metric == 'revenue',
        trendline='ols',
        trendline_color_override='red'
    )

    # Update layout
    fig.update_layout(
        title=f'Budget vs. Predicted {metric.capitalize()}',
        xaxis_title='Budget ($)',
        yaxis_title=f'Predicted {metric.capitalize()}',
        height=600
    )

    # Format axes
    fig.update_xaxes(tickprefix='$', tickformat=',.0f')
    if metric == 'revenue':
        fig.update_yaxes(tickprefix='$', tickformat=',.0f')

    # Add annotations for correlation
    if len(df) >= 5:  # Only calculate correlation if enough data
        correlation = df['budget'].corr(df[f'predicted_{metric}'])
        fig.add_annotation(
            x=0.02,
            y=0.98,
            xref="paper",
            yref="paper",
            text=f"Correlation: {correlation:.2f}",
            showarrow=False,
            font=dict(size=14),
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="black",
            borderwidth=1
        )

    # Display the chart
    st.plotly_chart(fig, use_container_width=True)

    # Create budget bins and analyze performance
    try:
        # Create 5 budget quantiles
        df['budget_quantile'] = pd.qcut(df['budget'], 5, labels=False)

        # Calculate average metric value for each bin
        budget_analysis = df.groupby('budget_quantile').agg({
            'budget': ['min', 'max', 'mean', 'count'],
            f'predicted_{metric}': 'mean'
        }).reset_index()

        # Format the data for display
        budget_bins = []
        for _, row in budget_analysis.iterrows():
            budget_bins.append({
                'Quantile': f"Q{int(row['budget_quantile']) + 1}",
                'Budget Range': f"${row['budget']['min']:,.0f} - ${row['budget']['max']:,.0f}",
                'Avg Budget': f"${row['budget']['mean']:,.0f}",
                f'Avg {metric.capitalize()}': format_metric_value(row[f'predicted_{metric}']['mean'], metric),
                'Count': int(row['budget']['count'])
            })

        budget_df = pd.DataFrame(budget_bins)

        # Display as a table
        st.subheader("Performance by Budget Quantile")
        st.dataframe(budget_df, use_container_width=True)

        # Reset index to make budget_quantile a regular column
        budget_analysis_reset = budget_analysis.reset_index()

        # Prepare data for the chart in the right format
        chart_data = pd.DataFrame({
            'Budget Quantile': [f"Q{int(q) + 1}" for q in budget_analysis_reset['budget_quantile']],
            'Average Value': budget_analysis_reset[f'predicted_{metric}']['mean'],
            'Count': budget_analysis_reset['budget']['count']
        })

        # Create a bar chart of average metric by budget quantile
        fig = px.bar(
            chart_data,
            x='Budget Quantile',
            y='Average Value',
            labels={
                'Budget Quantile': 'Budget Quantile',
                'Average Value': f'Avg. Predicted {metric.capitalize()}'
            },
            color='Budget Quantile',
            text='Count'
        )

        # Update layout
        fig.update_layout(
            title=f'Average Predicted {metric.capitalize()} by Budget Quantile',
            xaxis_title='Budget Quantile',
            xaxis=dict(
                tickmode='array',
                tickvals=list(range(5)),
                ticktext=[f"Q{i + 1}" for i in range(5)]
            ),
            yaxis_title=f'Avg. Predicted {metric.capitalize()}',
            showlegend=False,
            height=400
        )

        # Display the chart
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.warning(f"Couldn't create budget quantile analysis: {e}")


def compare_production_stages(in_prod_df, post_prod_df, metric):
    """Compare in-production and post-production movies"""
    st.header("Comparing In-Production vs. Post-Production Movies")

    if in_prod_df is None or post_prod_df is None or f'predicted_{metric}' not in in_prod_df.columns or f'predicted_{metric}' not in post_prod_df.columns:
        st.warning("Insufficient data to compare production stages.")
        return

    # Create comparison dataframe
    comparison_data = [
        {
            'Stage': 'In Production',
            'Avg Value': in_prod_df[f'predicted_{metric}'].mean(),
            'Median Value': in_prod_df[f'predicted_{metric}'].median(),
            'Min Value': in_prod_df[f'predicted_{metric}'].min(),
            'Max Value': in_prod_df[f'predicted_{metric}'].max(),
            'Count': len(in_prod_df)
        },
        {
            'Stage': 'Post-Production',
            'Avg Value': post_prod_df[f'predicted_{metric}'].mean(),
            'Median Value': post_prod_df[f'predicted_{metric}'].median(),
            'Min Value': post_prod_df[f'predicted_{metric}'].min(),
            'Max Value': post_prod_df[f'predicted_{metric}'].max(),
            'Count': len(post_prod_df)
        }
    ]

    comparison_df = pd.DataFrame(comparison_data)

    # Format for display
    display_df = comparison_df.copy()

    if metric == 'revenue':
        format_cols = ['Avg Value', 'Median Value', 'Min Value', 'Max Value']
        for col in format_cols:
            display_df[col] = display_df[col].apply(lambda x: f"${x:,.0f}")
    elif metric == 'roi':
        format_cols = ['Avg Value', 'Median Value', 'Min Value', 'Max Value']
        for col in format_cols:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}x")
    else:
        format_cols = ['Avg Value', 'Median Value', 'Min Value', 'Max Value']
        for col in format_cols:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}")

    # Display the comparison
    st.dataframe(display_df, use_container_width=True)

    # Create visualization tabs
    comp_tab1, comp_tab2, comp_tab3 = st.tabs(["📊 Box Plot", "📈 Histograms", "🔍 Top Genres"])

    with comp_tab1:
        # Create a box plot comparison
        box_data = []

        for idx, row in in_prod_df.iterrows():
            box_data.append({
                'Stage': 'In Production',
                'Value': row[f'predicted_{metric}']
            })

        for idx, row in post_prod_df.iterrows():
            box_data.append({
                'Stage': 'Post-Production',
                'Value': row[f'predicted_{metric}']
            })

        box_df = pd.DataFrame(box_data)

        fig = px.box(
            box_df,
            x='Stage',
            y='Value',
            color='Stage',
            points='all',
            labels={
                'Value': f'Predicted {metric.capitalize()}',
                'Stage': 'Production Stage'
            },
            title=f'Distribution of Predicted {metric.capitalize()} by Production Stage'
        )

        # Update layout
        fig.update_layout(height=500)

        # Display the chart
        st.plotly_chart(fig, use_container_width=True)

    with comp_tab2:
        # Create histograms
        fig = go.Figure()

        # Add in-production histogram
        fig.add_trace(go.Histogram(
            x=in_prod_df[f'predicted_{metric}'],
            name='In Production',
            opacity=0.7,
            marker_color='blue'
        ))

        # Add post-production histogram
        fig.add_trace(go.Histogram(
            x=post_prod_df[f'predicted_{metric}'],
            name='Post-Production',
            opacity=0.7,
            marker_color='red'
        ))

        # Update layout
        fig.update_layout(
            title=f'Distribution of Predicted {metric.capitalize()} by Production Stage',
            xaxis_title=f'Predicted {metric.capitalize()}',
            yaxis_title='Count',
            barmode='overlay',
            height=500
        )

        # Display the chart
        st.plotly_chart(fig, use_container_width=True)

    with comp_tab3:
        # Compare top genres between stages
        genre_cols = [col for col in in_prod_df.columns if col.startswith('genre_')]

        if genre_cols:
            # Calculate metrics for in-production genres
            in_prod_genres = []
            for genre in genre_cols:
                genre_movies = in_prod_df[in_prod_df[genre] == 1]
                if len(genre_movies) >= MIN_SAMPLE_SIZE:
                    avg_value = genre_movies[f'predicted_{metric}'].mean()
                    genre_name = genre.replace('genre_', '')
                    in_prod_genres.append({
                        'genre': genre_name,
                        'avg_value': avg_value,
                        'count': len(genre_movies),
                        'stage': 'In Production'
                    })

            # Calculate metrics for post-production genres
            post_prod_genres = []
            for genre in genre_cols:
                genre_movies = post_prod_df[post_prod_df[genre] == 1]
                if len(genre_movies) >= MIN_SAMPLE_SIZE:
                    avg_value = genre_movies[f'predicted_{metric}'].mean()
                    genre_name = genre.replace('genre_', '')
                    post_prod_genres.append({
                        'genre': genre_name,
                        'avg_value': avg_value,
                        'count': len(genre_movies),
                        'stage': 'Post-Production'
                    })

            # Combine the data
            all_genres = in_prod_genres + post_prod_genres

            if all_genres:
                genre_comparison = pd.DataFrame(all_genres)

                # Create a grouped bar chart
                fig = px.bar(
                    genre_comparison,
                    x='genre',
                    y='avg_value',
                    color='stage',
                    barmode='group',
                    labels={
                        'genre': 'Genre',
                        'avg_value': f'Avg. Predicted {metric.capitalize()}',
                        'stage': 'Production Stage'
                    },
                    title=f'Genre Performance Comparison by Production Stage',
                    text='count'
                )

                # Update layout
                fig.update_layout(
                    xaxis_title='Genre',
                    yaxis_title=f'Avg. Predicted {metric.capitalize()}',
                    height=500
                )

                # Add count labels
                fig.update_traces(texttemplate='%{text}', textposition='outside')

                # Display the chart
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Numbers above bars indicate the count of movies in each genre-stage combination.")
            else:
                st.info("Not enough data for genre comparison across production stages.")
        else:
            st.info("No genre information found in the dataset.")


# --- Main App ---
st.title("Model Prediction Results")

st.markdown("""
This dashboard explores movies predicted to be the most successful based on various metrics.
Select a success metric and explore top movies in production and post-production stages.
""")

# Load data
in_prod_df, post_prod_df = load_data()

if in_prod_df is not None and post_prod_df is not None:
    # Determine available metrics
    prediction_cols = [col for col in in_prod_df.columns if col.startswith('predicted_')]
    available_metrics = [col.replace('predicted_', '') for col in prediction_cols]

    # Default to revenue if available
    default_metric = 'revenue' if 'predicted_revenue' in prediction_cols else available_metrics[
        0] if available_metrics else None

    if not available_metrics:
        st.error(
            "No prediction columns found in the data. Make sure your dataframes contain columns starting with 'predicted_'.")
        st.stop()

    # --- User Inputs ---
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        selected_metric = st.selectbox(
            "Select Success Metric:",
            options=available_metrics,
            index=available_metrics.index(default_metric) if default_metric in available_metrics else 0,
            help="Choose which prediction metric to analyze"
        )

    with col2:
        top_n = st.number_input("Number of top movies to show:",
                                min_value=5, max_value=50, value=10, step=5,
                                help="How many top-performing movies to display")

    with col3:
        enable_filtering = st.checkbox("Enable filtering options", value=False,
                                       help="Show additional filtering options for the data")

    # --- Main Content ---
    tab1, tab2, tab3 = st.tabs(["📊 Top Movies", "🔍 Detailed Analysis", "📈 Stage Comparison"])

    with tab1:
        st.header("Top Movies by Predicted Success")

        # Display key metrics
        st.subheader("Key Metrics Overview")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<h4>In Production Movies</h4>", unsafe_allow_html=True)
            display_key_metrics(in_prod_df, selected_metric)

        with col2:
            st.markdown("<h4>Post-Production Movies</h4>", unsafe_allow_html=True)
            display_key_metrics(post_prod_df, selected_metric)

        st.markdown("---")

        # Display top movies
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🎬 In Production Movies")
            top_in_prod = display_top_movies(in_prod_df, selected_metric, n=top_n, stage="In Production",
                                             allow_filtering=enable_filtering)

        with col2:
            st.subheader("🎥 Post-Production Movies")
            top_post_prod = display_top_movies(post_prod_df, selected_metric, n=top_n, stage="Post-Production",
                                               allow_filtering=enable_filtering)

    with tab2:
        st.header(f"Detailed Analysis (Metric: {selected_metric.capitalize()})")

        tab_in_prod, tab_post_prod = st.tabs(["In Production", "Post-Production"])

        with tab_in_prod:
            st.subheader("In Production Movies Analysis")

            # Time-based analysis
            create_time_analysis(in_prod_df, selected_metric)

            st.markdown("---")

            # Create two columns for genre and company analysis
            col1, col2 = st.columns(2)

            with col1:
                create_genre_analysis(in_prod_df, selected_metric)

            with col2:
                create_company_analysis(in_prod_df, selected_metric)

            st.markdown("---")

            # Budget analysis
            create_budget_analysis(in_prod_df, selected_metric)

        with tab_post_prod:
            st.subheader("Post-Production Movies Analysis")

            # Time-based analysis
            create_time_analysis(post_prod_df, selected_metric)

            st.markdown("---")

            # Create two columns for genre and company analysis
            col1, col2 = st.columns(2)

            with col1:
                create_genre_analysis(post_prod_df, selected_metric)

            with col2:
                create_company_analysis(post_prod_df, selected_metric)

            st.markdown("---")

            # Budget analysis
            create_budget_analysis(post_prod_df, selected_metric)

    with tab3:
        # Add comparison between in-production and post-production
        compare_production_stages(in_prod_df, post_prod_df, selected_metric)

    # --- Sidebar Information ---
    st.sidebar.title("About This App")
    st.sidebar.info(
        """
        This app helps analyze movie predictions for both in-production and post-production movies.

        **Data Source:**
        - In-Production Movies: `{}`
        - Post-Production Movies: `{}`

        **Features:**
        - Explore top-performing movies by various metrics
        - Analyze seasonal and monthly trends
        - Identify high-performing genres and production companies
        - Understand the impact of budget on predicted success
        - Compare in-production vs. post-production trends
        """.format(IN_PRODUCTION_FILE, POST_PRODUCTION_FILE)
    )