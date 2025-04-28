import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import calendar
import os

# --- Configuration ---
IN_PRODUCTION_FILE = "output/in_prod_predictions.parquet"
POST_PRODUCTION_FILE = "output/post_prod_predictions.parquet"

MIN_SAMPLE_SIZE = 5  # Minimum movies needed for aggregation

# --- Page Config ---
st.set_page_config(layout="wide", page_title="Movie Success Prediction Explorer",
                   page_icon="🎬", initial_sidebar_state="expanded")


# --- Data Loading (Cached) ---
@st.cache_data
def load_data():
    try:
        # Try to load the parquet files if they exist
        if os.path.exists(IN_PRODUCTION_FILE) and os.path.exists(POST_PRODUCTION_FILE):
            in_prod = pd.read_parquet(IN_PRODUCTION_FILE)
            post_prod = pd.read_parquet(POST_PRODUCTION_FILE)
            return in_prod, post_prod
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None


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


def display_top_movies(df, metric, n=10, stage="In Production"):
    """Display top n movies based on the selected metric"""
    if df is None or df.empty:
        st.warning(f"No data available for {stage} movies.")
        return

    # Create a clean display dataframe
    if f'predicted_{metric}' not in df.columns:
        st.error(f"Prediction column 'predicted_{metric}' not found in the dataset.")
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

    # Display the dataframe
    st.dataframe(df_display.style.format(format_dict), use_container_width=True)

    return df_sorted


def create_time_analysis(df, metric):
    """Create visualizations for time-based analysis"""
    col1, col2 = st.columns(2)

    # Season analysis
    with col1:
        st.subheader("Performance by Season")
        season_data = df.groupby('release_season')[f'predicted_{metric}'].mean().reset_index()

        if not season_data.empty:
            season_order = ['Winter', 'Spring', 'Summer', 'Fall']
            chart = alt.Chart(season_data).mark_bar().encode(
                x=alt.X('release_season', title='Season', sort=season_order),
                y=alt.Y(f'predicted_{metric}', title=f'Avg. Predicted {metric.capitalize()}'),
                color=alt.Color('release_season', scale=alt.Scale(scheme='category10'), legend=None),
                tooltip=[
                    alt.Tooltip('release_season', title='Season'),
                    alt.Tooltip(f'predicted_{metric}', title=f'Avg. Predicted {metric.capitalize()}', format='.2f')
                ]
            ).properties(height=300)
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("Not enough season data for analysis.")

    # Month analysis
    with col2:
        st.subheader("Performance by Month")
        month_data = df.groupby('release_month')[f'predicted_{metric}'].mean().reset_index()

        if not month_data.empty:
            # Add month names
            month_data['month_name'] = month_data['release_month'].apply(
                lambda x: calendar.month_abbr[int(x)] if isinstance(x, (int, float)) and 1 <= x <= 12 else str(x)
            )

            month_order = [calendar.month_abbr[i] for i in range(1, 13)]
            chart = alt.Chart(month_data).mark_bar().encode(
                x=alt.X('month_name', title='Month', sort=month_order),
                y=alt.Y(f'predicted_{metric}', title=f'Avg. Predicted {metric.capitalize()}'),
                color=alt.Color('month_name', scale=alt.Scale(scheme='tableau10'), legend=None),
                tooltip=[
                    alt.Tooltip('month_name', title='Month'),
                    alt.Tooltip(f'predicted_{metric}', title=f'Avg. Predicted {metric.capitalize()}', format='.2f')
                ]
            ).properties(height=300)
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("Not enough month data for analysis.")


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
            genre_data.append({
                'genre': genre.replace('genre_', ''),
                'avg_value': avg_value,
                'count': len(genre_movies)
            })

    if genre_data:
        genre_df = pd.DataFrame(genre_data).sort_values('avg_value', ascending=False)

        chart = alt.Chart(genre_df).mark_bar().encode(
            x=alt.X('avg_value', title=f'Avg. Predicted {metric.capitalize()}'),
            y=alt.Y('genre', title='Genre', sort='-x'),
            color=alt.Color('genre', scale=alt.Scale(scheme='category20'), legend=None),
            tooltip=[
                alt.Tooltip('genre', title='Genre'),
                alt.Tooltip('avg_value', title=f'Avg. Predicted {metric.capitalize()}', format='.2f'),
                alt.Tooltip('count', title='Number of Movies')
            ]
        ).properties(height=400)
        st.altair_chart(chart, use_container_width=True)
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
            company_data.append({
                'company': company.replace('production_companies_', ''),
                'avg_value': avg_value,
                'count': len(company_movies)
            })

    if company_data:
        company_df = pd.DataFrame(company_data).sort_values('avg_value', ascending=False).head(10)  # Top 10 companies

        chart = alt.Chart(company_df).mark_bar().encode(
            x=alt.X('avg_value', title=f'Avg. Predicted {metric.capitalize()}'),
            y=alt.Y('company', title='Production Company', sort='-x'),
            color=alt.Color('company', scale=alt.Scale(scheme='tableau20'), legend=None),
            tooltip=[
                alt.Tooltip('company', title='Company'),
                alt.Tooltip('avg_value', title=f'Avg. Predicted {metric.capitalize()}', format='.2f'),
                alt.Tooltip('count', title='Number of Movies')
            ]
        ).properties(height=400)
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Not enough data per production company for analysis.")


def create_budget_analysis(df, metric):
    """Create visualizations for budget analysis"""
    if 'budget' not in df.columns:
        st.info("Budget information not available in the dataset.")
        return

    st.subheader("Budget vs. Predicted Success")

    # Create scatter plot
    chart = alt.Chart(df).mark_circle(size=60, opacity=0.7).encode(
        x=alt.X('budget', title='Budget', scale=alt.Scale(type='log')),
        y=alt.Y(f'predicted_{metric}', title=f'Predicted {metric.capitalize()}',
                scale=alt.Scale(type='log' if metric == 'revenue' else 'linear')),
        color=alt.value('#1f77b4'),
        tooltip=[
            alt.Tooltip('title', title='Movie'),
            alt.Tooltip('budget', title='Budget', format='$,.0f'),
            alt.Tooltip(f'predicted_{metric}', title=f'Predicted {metric.capitalize()}',
                        format='$,.0f' if metric == 'revenue' else '.2f')
        ]
    ).properties(height=400)

    # Add a regression line
    regression = chart.transform_regression(
        'budget', f'predicted_{metric}'
    ).mark_line(color='red')

    # Combine the scatter plot and regression line
    final_chart = alt.layer(chart, regression)
    st.altair_chart(final_chart, use_container_width=True)


# --- Main App ---
st.title("🎬 Movie Success Prediction Explorer")

st.markdown("""
This app helps you explore movies predicted to be the most successful based on various metrics.
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
            index=available_metrics.index(default_metric) if default_metric in available_metrics else 0
        )

    with col2:
        top_n = st.number_input("Number of top movies to show:", min_value=5, max_value=50, value=10, step=5)

    with col3:
        show_all_metrics = st.checkbox("Show all metrics in tables", value=False)

    # --- Main Content ---
    tab1, tab2 = st.tabs(["📊 Top Movies", "🔍 Detailed Analysis"])

    with tab1:
        st.header("Top Movies by Predicted Success")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🎬 In Production Movies")
            top_in_prod = display_top_movies(in_prod_df, selected_metric, n=top_n, stage="In Production")

        with col2:
            st.subheader("🎥 Post-Production Movies")
            top_post_prod = display_top_movies(post_prod_df, selected_metric, n=top_n, stage="Post-Production")

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

    # --- Footer ---
    st.markdown("---")
    st.caption(
        "Note: This analysis is based on machine learning predictions and should be used as one of many factors in decision making.")
    st.caption(
        "Predictions are estimates and actual performance may vary based on many factors not captured in the model.")

else:
    st.error("Failed to load data. Please check the file paths and data formats.")
    st.info("To use this app, you need to save your dataframes with predictions as Parquet files.")

    # Instructions
    with st.expander("How to prepare your data"):
        st.markdown("""
        1. After running your prediction code, save the dataframes with predictions:
        ```python
        in_prod_df.to_parquet('output/in_prod_df_with_predictions.parquet')
        post_prod_df.to_parquet('output/post_prod_df_with_predictions.parquet')
        ```

        2. Make sure the dataframes contain:
           - Movie titles in a 'title' column
           - Prediction columns starting with 'predicted_'
           - Release information (year, month, season)
           - Genre columns starting with 'genre_'
           - Production company columns starting with 'production_companies_'
           - Budget information in a 'budget' column (optional)
        """)