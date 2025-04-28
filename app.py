import streamlit as st
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt # No longer needed for plots
# import seaborn as sns # No longer needed
import altair as alt # Import Altair
import os
import calendar # For month names

# --- Configuration ---
DATA_FILE_PARQUET = "output/df_processed_for_streamlit.parquet"
DATA_FILE_CSV = "output/df_processed_for_streamlit.csv"

MIN_SAMPLE_SIZE = 20 # Minimum movies needed in a genre/group for analysis

# --- Page Config ---
st.set_page_config(layout="wide", page_title="Movie Genre Performance Analyzer")

# --- Data Loading (Cached) ---
@st.cache_data # Cache the data loading to speed up app interaction
def load_data():
    if os.path.exists(DATA_FILE_PARQUET):
        try:
            df = pd.read_parquet(DATA_FILE_PARQUET)
            return df
        except Exception as e:
            st.error(f"Error loading Parquet file '{DATA_FILE_PARQUET}': {e}")
            pass # Fallback to CSV check

    if os.path.exists(DATA_FILE_CSV):
         try:
             df = pd.read_csv(DATA_FILE_CSV)
             return df
         except Exception as e:
             st.error(f"Error loading CSV file '{DATA_FILE_CSV}': {e}")
             return None # Indicate failure
    else:
         st.error(f"Error: Data file not found. Please ensure '{DATA_FILE_PARQUET}' or '{DATA_FILE_CSV}' exists in the 'output' folder relative to app.py.")
         st.stop() # Stop execution if no data
         return None

# --- Analysis Function ---
def run_analysis(df_processed, genre, metric):
    st.header(f"Analysis for: {genre.replace('genre_', '')} | Metric: {metric}")
    st.markdown("---")

    # 1. Filter Data for the selected genre
    df_genre = df_processed[df_processed[genre] == 1].copy()

    if len(df_genre) < MIN_SAMPLE_SIZE:
        st.warning(f"Only {len(df_genre)} movies found for genre '{genre.replace('genre_', '')}'. Results may not be reliable.")
        return # Stop analysis if too few samples

    # Handle potential NaNs in the metric column for this subset
    df_genre = df_genre.dropna(subset=[metric])
    if len(df_genre) == 0:
         st.error(f"No valid data points found for metric '{metric}' within this genre after dropping NaNs.")
         return
    st.write(f"Analyzing **{len(df_genre)}** movies for this genre (after handling NaNs in '{metric}').")


    # --- 2. Time Period Analysis ---
    st.subheader("Performance Over Time")
    col1, col2, col3 = st.columns(3)

    # Year
    with col1:
        year_perf = df_genre.groupby('release_year')[metric].agg(['mean', 'count'])
        year_perf = year_perf[year_perf['count'] >= MIN_SAMPLE_SIZE / 4]
        if not year_perf.empty:
            best_year = year_perf['mean'].idxmax()
            st.metric(label="Best Year (Avg)", value=f"{best_year}", delta=f"{year_perf.loc[best_year, 'mean']:.2f}")

            # Altair Line Chart for Year Trend
            year_perf_reset = year_perf.reset_index()
            chart = alt.Chart(year_perf_reset).mark_line(point=True).encode(
                x=alt.X('release_year', axis=alt.Axis(title='Year', format='d')), # Format as integer year
                y=alt.Y('mean', axis=alt.Axis(title=f'Avg {metric}')),
                tooltip=[
                    alt.Tooltip('release_year', title='Year'),
                    alt.Tooltip('mean', title=f'Avg. {metric}', format='.2f'),
                    alt.Tooltip('count', title='Count')
                    ]
            ).properties(
                title=f'Avg {metric} by Year'
            ).interactive() # Add zooming/panning
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("Not enough data per year for reliable trend.")

    # Season
    with col2:
        season_perf = df_genre.groupby('release_season')[metric].agg(['mean', 'count'])
        season_perf = season_perf[season_perf['count'] >= MIN_SAMPLE_SIZE / 4]
        if not season_perf.empty:
            best_season = season_perf['mean'].idxmax()
            st.metric(label="Best Season (Avg)", value=f"{best_season}", delta=f"{season_perf.loc[best_season, 'mean']:.2f}")

            # Altair Bar Chart for Season
            season_perf_reset = season_perf.reset_index()
            season_order = ['Spring', 'Summer', 'Fall', 'Winter']
            chart = alt.Chart(season_perf_reset).mark_bar().encode(
                x=alt.X('release_season', axis=alt.Axis(title='Season'), sort=season_order),
                y=alt.Y('mean', axis=alt.Axis(title=f'Avg {metric}')),
                tooltip=[
                    alt.Tooltip('release_season', title='Season'),
                    alt.Tooltip('mean', title=f'Avg. {metric}', format='.2f'),
                    alt.Tooltip('count', title='Count')
                ]
            ).properties(
                title=f'Avg {metric} by Season'
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("Not enough data per season.")

    # Month
    with col3:
        month_perf = df_genre.groupby('release_month')[metric].agg(['mean', 'count'])
        month_perf = month_perf[month_perf['count'] >= MIN_SAMPLE_SIZE / 12]
        if not month_perf.empty:
            best_month = month_perf['mean'].idxmax()
            best_month_name = calendar.month_name[best_month]
            st.metric(label="Best Month (Avg)", value=f"{best_month_name} ({best_month})", delta=f"{month_perf.loc[best_month, 'mean']:.2f}")

            # Altair Bar Chart for Month
            month_perf_reset = month_perf.reset_index()
            month_perf_reset['month_name'] = month_perf_reset['release_month'].apply(lambda x: calendar.month_abbr[x])
            month_order = [calendar.month_abbr[i] for i in range(1, 13)]

            chart = alt.Chart(month_perf_reset).mark_bar().encode(
                # Sort by month number, label with abbr name
                x=alt.X('month_name', axis=alt.Axis(title='Month'), sort=month_order),
                y=alt.Y('mean', axis=alt.Axis(title=f'Avg {metric}')),
                tooltip=[
                    alt.Tooltip('month_name', title='Month'),
                    alt.Tooltip('mean', title=f'Avg. {metric}', format='.2f'),
                    alt.Tooltip('count', title='Count')
                ]
            ).properties(
                title=f'Avg {metric} by Month'
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("Not enough data per month.")

    st.markdown("---") # Separator

    # --- 3. Co-occurring Genre Analysis ---
    st.subheader("Co-occurring Genre Performance")
    other_genres = [g for g in available_genres if g != genre]
    co_genre_perf = {}
    overall_mean = df_genre[metric].mean()

    for other_g in other_genres:
        df_co_genre = df_genre[df_genre[other_g] == 1]
        count = len(df_co_genre)
        if count >= MIN_SAMPLE_SIZE / 2:
            mean_metric_val = df_co_genre[metric].mean()
            co_genre_perf[other_g.replace('genre_', '')] = {'mean': mean_metric_val, 'count': count}

    if co_genre_perf:
        co_genre_df = pd.DataFrame(co_genre_perf).T
        co_genre_df['diff_from_overall'] = co_genre_df['mean'] - overall_mean
        co_genre_df = co_genre_df.sort_values('mean', ascending=False)

        st.metric(label=f"Overall Avg {metric} for {genre.replace('genre_', '')}", value=f"{overall_mean:.2f}")

        # Display top N co-genres
        top_n_display = 10
        st.dataframe(co_genre_df.head(top_n_display))

        # Altair Horizontal Bar Chart for Co-Genres
        top_n_plot = 10
        plot_df_reset = co_genre_df.head(top_n_plot).reset_index().rename(columns={'index': 'co_genre'})

        bars = alt.Chart(plot_df_reset).mark_bar().encode(
            x=alt.X('mean', axis=alt.Axis(title=f'Average {metric}')),
            y=alt.Y('co_genre', axis=alt.Axis(title='Co-Genre'), sort='-x'), # Sort descending by mean value
            tooltip=[
                alt.Tooltip('co_genre', title='Co-Genre'),
                alt.Tooltip('mean', title=f'Avg. {metric}', format='.2f'),
                alt.Tooltip('count', title='Count')
            ]
        ).properties(
             title=f'Performance Impact of Top {top_n_plot} Co-Genres'
        )

        # Rule for the overall mean
        rule = alt.Chart(pd.DataFrame({'overall_mean': [overall_mean]})).mark_rule(color='red', strokeDash=[5,5], size=2).encode(
            x='overall_mean'
        )

        # Text label for the rule (optional, can be tricky to position)
        # text = rule.mark_text(
        #     align='left',
        #     baseline='middle',
        #     dx=7, # Nudges text to right so it doesn't overlap
        #     text=f'Overall Avg ({overall_mean:.2f})'
        # ).encode(text=alt.value(f'Overall Avg ({overall_mean:.2f})')) # Add text label


        # Layer the charts
        chart = alt.layer(bars, rule).resolve_scale(
            # You might need scale resolution if axes clash, but likely okay here
        )
        st.altair_chart(chart, use_container_width=True)
        # Add manual legend text if needed
        st.caption(f"<span style='color:red;'>----</span> Overall Avg ({overall_mean:.2f}) for {genre.replace('genre_', '')}", unsafe_allow_html=True)


    else:
        st.info("Not enough data with co-occurring genres to provide reliable analysis.")

    st.markdown("---") # Separator

    # --- 4. Budget Range Analysis ---
    st.subheader(f"Budget ({budget_col}) Analysis")
    if budget_col not in df_genre.columns:
         st.error(f"Budget column '{budget_col}' not found.")
         return

    try:
        # Bin budget (using quantiles on log-budget)
        num_bins = 5
        df_genre['budget_bin'], bin_edges = pd.qcut(df_genre[budget_col], q=num_bins, labels=False, retbins=True, duplicates='drop')
        budget_perf = df_genre.groupby('budget_bin')[metric].agg(['mean', 'count'])
        budget_perf = budget_perf[budget_perf['count'] >= MIN_SAMPLE_SIZE / num_bins]

        if not budget_perf.empty:
            best_budget_bin_idx = budget_perf['mean'].idxmax()
            lower_bound_log = bin_edges[best_budget_bin_idx]
            upper_bound_log = bin_edges[best_budget_bin_idx + 1]

            # Convert log-budget back to raw budget
            lower_bound_raw = np.expm1(lower_bound_log)
            upper_bound_raw = np.expm1(upper_bound_log)

            st.metric(label=f"Best Budget Bin (Avg {metric})", value=f"Bin {best_budget_bin_idx}", delta=f"{budget_perf.loc[best_budget_bin_idx, 'mean']:.2f}")
            st.write(f"  > Approx. Raw Budget Range: **${lower_bound_raw:,.0f} - ${upper_bound_raw:,.0f}**")

            # Altair Bar chart for Budget Bins
            budget_perf_reset = budget_perf.reset_index()
            # Create labels for bins (for axis and tooltip)
            bin_map = {}
            bin_map_tooltip = {}
            for i in range(num_bins):
                 lower = np.expm1(bin_edges[i])
                 upper = np.expm1(bin_edges[i+1])
                 bin_map[i] = f'Bin {i}' # Keep axis label simple
                 bin_map_tooltip[i] = f"${lower:,.0f} - ${upper:,.0f}"

            budget_perf_reset['bin_label'] = budget_perf_reset['budget_bin'].map(bin_map)
            budget_perf_reset['budget_range_tooltip'] = budget_perf_reset['budget_bin'].map(bin_map_tooltip)

            chart = alt.Chart(budget_perf_reset).mark_bar().encode(
                x=alt.X('bin_label', axis=alt.Axis(title=f'Budget Quantile Bin ({raw_budget_col})'), sort=alt.SortField('budget_bin')), # Sort by original index
                y=alt.Y('mean', axis=alt.Axis(title=f'Avg {metric}')),
                tooltip=[
                    alt.Tooltip('bin_label', title='Bin'),
                    alt.Tooltip('budget_range_tooltip', title='Approx. Budget Range'),
                    alt.Tooltip('mean', title=f'Avg. {metric}', format='.2f'),
                    alt.Tooltip('count', title='Count')
                ]
            ).properties(
                title=f'Avg {metric} by Budget Quantile Bin'
            )
            st.altair_chart(chart, use_container_width=True)

        else:
             st.info("Not enough data per budget bin for reliable analysis.")

        # Show budget stats for top performing movies (text-based is fine here)
        top_perf_quantile = 0.75
        top_movies = df_genre[df_genre[metric] >= df_genre[metric].quantile(top_perf_quantile)]
        if not top_movies.empty and raw_budget_col in top_movies.columns:
            median_budget_top = top_movies[raw_budget_col].median()
            iqr_budget_top = top_movies[raw_budget_col].quantile([0.25, 0.75])
            st.markdown(f"**Budget for Top {(1-top_perf_quantile)*100:.0f}% Performing Movies ({metric}):**")
            st.write(f"  - Median Budget: **${median_budget_top:,.0f}**")
            st.write(f"  - 25th-75th Percentile Budget: **${iqr_budget_top.iloc[0]:,.0f} - ${iqr_budget_top.iloc[1]:,.0f}**")

    except Exception as e:
        st.error(f"Error during budget analysis: {e}")
        st.exception(e) # Show traceback in app for debugging

    st.markdown("---")
    st.caption("Analysis Complete. Findings are based on historical correlations. Use alongside market knowledge.")

# --- Main App ---
st.title("🎬 Movie Genre Performance Analyzer")

st.markdown("""
Select a primary genre and a success metric to explore historical performance trends.
The analysis looks at timing, co-occurring genres, and budget ranges associated with success using Altair interactive charts.
""")

# Load Data
df_processed = load_data()

# --- Extract available genres/metrics AFTER loading ---
if df_processed is not None:
    # Define available genres (extract from columns)
    available_genres = sorted([col for col in df_processed.columns if col.startswith('genre_')])

    # Define success metrics (use log versions where available for better analysis of skewed data)
    available_metrics = sorted([
        'revenue_log', 'roi_log', 'popularity_log', 'vote_count_log', # Log-transformed
        'vote_average' # Original scale (usually more normally distributed)
    ])

    # Define budget column to use for analysis (log is often better)
    budget_col = 'budget_log'
    raw_budget_col = 'budget'

    # --- User Inputs ---
    col1, col2 = st.columns(2)
    with col1:
        selected_genre = st.selectbox(
            "Select Primary Genre:",
            options=available_genres,
            format_func=lambda x: x.replace('genre_', '') # Nicer display name
        )
    with col2:
        selected_metric = st.selectbox(
            "Select Success Metric:",
            options=available_metrics
        )

    # --- Run Analysis ---
    if selected_genre and selected_metric:
        run_analysis(df_processed, selected_genre, selected_metric)
    else:
        st.info("Please select a genre and a metric to start the analysis.")

else:
    st.error("Failed to load data. Cannot proceed with analysis.")