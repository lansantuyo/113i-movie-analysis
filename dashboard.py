import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import os
import calendar

# --- Configuration ---
DATA_FILE_PARQUET = "output/df_processed_for_streamlit.parquet"
DATA_FILE_CSV = "output/df_processed_for_streamlit.csv"

MIN_SAMPLE_SIZE = 20 # Minimum movies needed in a genre/group for analysis
MIN_SAMPLE_SIZE_ENTITY = 5 # Minimum movies needed for a specific company/country within the genre

# --- Page Config ---
st.set_page_config(layout="wide", page_title="Movie Genre Performance Analyzer")

# --- Data Loading (Cached) ---
@st.cache_data
def load_data():
    if os.path.exists(DATA_FILE_PARQUET):
        try:
            df = pd.read_parquet(DATA_FILE_PARQUET)
            # Ensure required raw columns exist for display if log versions are used
            for log_col in ['revenue_log', 'roi_log', 'popularity_log', 'vote_count_log', 'budget_log']:
                 raw_col = log_col.replace('_log', '')
                 if log_col in df.columns and raw_col not in df.columns:
                     # Attempt to calculate raw from log (approximation for display)
                     if raw_col == 'roi': # Handle potential shift in ROI log transform
                         # This requires knowing how roi_log was created. Assuming simple log1p for now.
                         # If ROI was shifted, this calculation needs adjustment.
                         df[raw_col] = np.expm1(df[log_col])
                     else:
                         df[raw_col] = np.expm1(df[log_col])
                     st.sidebar.warning(f"Raw column '{raw_col}' missing, calculated approx. from '{log_col}' for display.")

            return df
        except Exception as e:
            st.error(f"Error loading Parquet file '{DATA_FILE_PARQUET}': {e}")
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
                     st.sidebar.warning(f"Raw column '{raw_col}' missing, calculated approx. from '{log_col}' for display.")
             return df
         except Exception as e:
             st.error(f"Error loading CSV file '{DATA_FILE_CSV}': {e}")
             return None
    else:
         st.error(f"Error: Data file not found. Ensure '{DATA_FILE_PARQUET}' or '{DATA_FILE_CSV}' exists in 'output/'.")
         st.stop()
         return None

# --- Helper Function for Entity Analysis ---
def analyze_entity(df_genre, entity_prefix, metric, min_samples, title):
    st.subheader(title)
    entity_cols = [col for col in df_genre.columns if col.startswith(entity_prefix)]

    if not entity_cols:
        st.info(f"No {entity_prefix.replace('_', ' ')} columns found in the data.")
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

        st.metric(label=f"Overall Avg {metric} for {df_genre.replace('genre_', '')}", value=f"{overall_mean:.2f}")

        # Display top N entities
        top_n_display = 5
        st.dataframe(entity_df.head(top_n_display))

        # Altair Horizontal Bar Chart
        plot_df_reset = entity_df.head(top_n_display).reset_index().rename(columns={'index': 'entity_name'})

        bars = alt.Chart(plot_df_reset).mark_bar().encode(
            x=alt.X('mean', axis=alt.Axis(title=f'Average {metric}')),
            y=alt.Y('entity_name', axis=alt.Axis(title=title.split(" Performance")[0]), sort='-x'),
            tooltip=[
                alt.Tooltip('entity_name', title=title.split(" Performance")[0]),
                alt.Tooltip('mean', title=f'Avg. {metric}', format='.2f'),
                alt.Tooltip('count', title='Count')
            ]
        ).properties(
             title=f'Top {top_n_display} {title}'
        )
        st.altair_chart(bars, use_container_width=True)

    else:
        st.info(f"Not enough movies per {entity_prefix.replace('_', ' ')} (min {min_samples}) within this genre for reliable analysis.")

# --- Analysis Function ---
def run_analysis(df_processed, genre, metric):
    st.header(f"Analysis for: {genre.replace('genre_', '')} | Metric: {metric}")
    st.markdown("---")

    # 1. Filter Data for the selected genre
    df_genre = df_processed[df_processed[genre] == 1].copy()

    if len(df_genre) < MIN_SAMPLE_SIZE:
        st.warning(f"Only {len(df_genre)} movies found for genre '{genre.replace('genre_', '')}'. Results may not be reliable.")
        return

    # Handle potential NaNs in the metric column for this subset
    df_genre = df_genre.dropna(subset=[metric])
    if len(df_genre) == 0:
         st.error(f"No valid data points found for metric '{metric}' within this genre after dropping NaNs.")
         return
    st.write(f"Analyzing **{len(df_genre)}** movies for this genre (after handling NaNs in '{metric}').")

    # --- Tabbed Layout for Organization ---
    tab1, tab2, tab3, tab4 = st.tabs(["🕒 Timing", "🎭 Co-Genres", "💰 Budget", "🏢 Entities & Top Movies"])

    with tab1:
        # --- 2. Time Period Analysis ---
        st.subheader("Performance Over Time")
        col1, col2, col3 = st.columns(3)
        # ... (Year, Season, Month analysis code remains the same as previous version) ...
        # Year
        with col1:
            year_perf = df_genre.groupby('release_year')[metric].agg(['mean', 'count'])
            year_perf = year_perf[year_perf['count'] >= MIN_SAMPLE_SIZE / 4] # Min count per year
            if not year_perf.empty:
                best_year = year_perf['mean'].idxmax()
                st.metric(label="Best Year (Avg)", value=f"{best_year}", delta=f"{year_perf.loc[best_year, 'mean']:.2f}")
                year_perf_reset = year_perf.reset_index()
                chart = alt.Chart(year_perf_reset).mark_line(point=True).encode(
                    x=alt.X('release_year', axis=alt.Axis(title='Year', format='d')),
                    y=alt.Y('mean', axis=alt.Axis(title=f'Avg {metric}')),
                    tooltip=[alt.Tooltip('release_year', title='Year'), alt.Tooltip('mean', title=f'Avg. {metric}', format='.2f'), alt.Tooltip('count', title='Count')]
                ).properties(title=f'Avg {metric} by Year').interactive()
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
                season_perf_reset = season_perf.reset_index()
                season_order = ['Spring', 'Summer', 'Fall', 'Winter']
                chart = alt.Chart(season_perf_reset).mark_bar().encode(
                    x=alt.X('release_season', axis=alt.Axis(title='Season'), sort=season_order),
                    y=alt.Y('mean', axis=alt.Axis(title=f'Avg {metric}')),
                    tooltip=[alt.Tooltip('release_season', title='Season'), alt.Tooltip('mean', title=f'Avg. {metric}', format='.2f'), alt.Tooltip('count', title='Count')]
                ).properties(title=f'Avg {metric} by Season')
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
                month_perf_reset = month_perf.reset_index()
                month_perf_reset['month_name'] = month_perf_reset['release_month'].apply(lambda x: calendar.month_abbr[x])
                month_order = [calendar.month_abbr[i] for i in range(1, 13)]
                chart = alt.Chart(month_perf_reset).mark_bar().encode(
                    x=alt.X('month_name', axis=alt.Axis(title='Month'), sort=month_order),
                    y=alt.Y('mean', axis=alt.Axis(title=f'Avg {metric}')),
                    tooltip=[alt.Tooltip('month_name', title='Month'), alt.Tooltip('mean', title=f'Avg. {metric}', format='.2f'), alt.Tooltip('count', title='Count')]
                ).properties(title=f'Avg {metric} by Month')
                st.altair_chart(chart, use_container_width=True)
            else:
                st.info("Not enough data per month.")

    with tab2:
        # --- 3. Co-occurring Genre Analysis ---
        st.subheader("Co-occurring Genre Performance")
        other_genres = [g for g in available_genres if g != genre]
        co_genre_perf = {}
        overall_mean_cogenre = df_genre[metric].mean() # Recalculate mean for clarity

        for other_g in other_genres:
            df_co_genre = df_genre[df_genre[other_g] == 1]
            count = len(df_co_genre)
            if count >= MIN_SAMPLE_SIZE / 2:
                mean_metric_val = df_co_genre[metric].mean()
                co_genre_perf[other_g.replace('genre_', '')] = {'mean': mean_metric_val, 'count': count}

        if co_genre_perf:
            co_genre_df = pd.DataFrame(co_genre_perf).T
            co_genre_df['diff_from_overall'] = co_genre_df['mean'] - overall_mean_cogenre
            co_genre_df = co_genre_df.sort_values('mean', ascending=False)

            st.metric(label=f"Overall Avg {metric} for {genre.replace('genre_', '')}", value=f"{overall_mean_cogenre:.2f}")
            top_n_display_co = 10
            st.dataframe(co_genre_df.head(top_n_display_co))

            # Altair Horizontal Bar Chart for Co-Genres
            plot_df_reset_co = co_genre_df.head(top_n_display_co).reset_index().rename(columns={'index': 'co_genre'})
            bars_co = alt.Chart(plot_df_reset_co).mark_bar().encode(
                x=alt.X('mean', axis=alt.Axis(title=f'Average {metric}')),
                y=alt.Y('co_genre', axis=alt.Axis(title='Co-Genre'), sort='-x'),
                tooltip=[alt.Tooltip('co_genre', title='Co-Genre'), alt.Tooltip('mean', title=f'Avg. {metric}', format='.2f'), alt.Tooltip('count', title='Count')]
            ).properties(title=f'Performance Impact of Top {top_n_display_co} Co-Genres')
            rule_co = alt.Chart(pd.DataFrame({'overall_mean': [overall_mean_cogenre]})).mark_rule(color='red', strokeDash=[5,5], size=2).encode(x='overall_mean')
            chart_co = alt.layer(bars_co, rule_co)
            st.altair_chart(chart_co, use_container_width=True)
            st.caption(f"<span style='color:red;'>----</span> Overall Avg ({overall_mean_cogenre:.2f}) for {genre.replace('genre_', '')}", unsafe_allow_html=True)
        else:
            st.info("Not enough data with co-occurring genres to provide reliable analysis.")

    with tab3:
        # --- 4. Budget Range Analysis ---
        st.subheader(f"Budget ({budget_col}) Analysis")
        if budget_col not in df_genre.columns:
             st.error(f"Budget column '{budget_col}' not found.")
        else:
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
                    lower_bound_raw = np.expm1(lower_bound_log)
                    upper_bound_raw = np.expm1(upper_bound_log)

                    st.metric(label=f"Best Budget Bin (Avg {metric})", value=f"Bin {best_budget_bin_idx}", delta=f"{budget_perf.loc[best_budget_bin_idx, 'mean']:.2f}")
                    st.write(f"  > Approx. Raw Budget Range: **${lower_bound_raw:,.0f} - ${upper_bound_raw:,.0f}**")

                    # Altair Bar chart for Budget Bins
                    budget_perf_reset = budget_perf.reset_index()
                    bin_map = {}
                    bin_map_tooltip = {}
                    for i in range(num_bins):
                         lower = np.expm1(bin_edges[i])
                         upper = np.expm1(bin_edges[i+1])
                         bin_map[i] = f'Bin {i}'
                         bin_map_tooltip[i] = f"${lower:,.0f} - ${upper:,.0f}"

                    budget_perf_reset['bin_label'] = budget_perf_reset['budget_bin'].map(bin_map)
                    budget_perf_reset['budget_range_tooltip'] = budget_perf_reset['budget_bin'].map(bin_map_tooltip)

                    chart_budget = alt.Chart(budget_perf_reset).mark_bar().encode(
                        x=alt.X('bin_label', axis=alt.Axis(title=f'Budget Quantile Bin ({raw_budget_col})'), sort=alt.SortField('budget_bin')),
                        y=alt.Y('mean', axis=alt.Axis(title=f'Avg {metric}')),
                        tooltip=[alt.Tooltip('bin_label', title='Bin'), alt.Tooltip('budget_range_tooltip', title='Approx. Budget Range'), alt.Tooltip('mean', title=f'Avg. {metric}', format='.2f'), alt.Tooltip('count', title='Count')]
                    ).properties(title=f'Avg {metric} by Budget Quantile Bin')
                    st.altair_chart(chart_budget, use_container_width=True)

                else:
                     st.info("Not enough data per budget bin for reliable analysis.")

                # Also show budget stats for top performing movies
                top_perf_quantile = 0.75
                top_movies_budget = df_genre[df_genre[metric] >= df_genre[metric].quantile(top_perf_quantile)]
                if not top_movies_budget.empty and raw_budget_col in top_movies_budget.columns:
                    median_budget_top = top_movies_budget[raw_budget_col].median()
                    iqr_budget_top = top_movies_budget[raw_budget_col].quantile([0.25, 0.75])
                    st.markdown(f"**Budget for Top {(1-top_perf_quantile)*100:.0f}% Performing Movies ({metric}):**")
                    st.write(f"  - Median Budget: **${median_budget_top:,.0f}**")
                    st.write(f"  - 25th-75th Percentile Budget: **${iqr_budget_top.iloc[0]:,.0f} - ${iqr_budget_top.iloc[1]:,.0f}**")

            except Exception as e:
                st.error(f"Error during budget analysis: {e}")
                st.exception(e)

    with tab4:
        # --- 5. Entity Analysis ---
        st.markdown("<br>", unsafe_allow_html=True) # Add some space
        analyze_entity(df_genre, "production_companies_", metric, MIN_SAMPLE_SIZE_ENTITY, "Production Company Performance")
        st.markdown("---")
        analyze_entity(df_genre, "production_countries_", metric, MIN_SAMPLE_SIZE_ENTITY, "Production Country Performance")
        st.markdown("---")

        # --- 6. Top Performing Movies ---
        st.subheader(f"Top Performing Movies (by {metric})")
        top_n_movies = 10
        df_top_movies = df_genre.sort_values(metric, ascending=False).head(top_n_movies)

        # Select columns to display - include raw metric if log version was used
        display_cols = ['title', 'release_year']
        if metric.endswith('_log'):
            raw_metric_col = metric.replace('_log', '')
            if raw_metric_col in df_top_movies.columns:
                display_cols.extend([metric, raw_metric_col])
            else:
                display_cols.append(metric) # Only show log if raw isn't available
        else:
            display_cols.append(metric) # Show the metric itself if not log

        # Add raw budget for context
        if raw_budget_col in df_top_movies.columns:
             display_cols.append(raw_budget_col)
        elif budget_col in df_top_movies.columns: # Fallback to log budget
             display_cols.append(budget_col)

        # Ensure all display columns exist
        display_cols = [col for col in display_cols if col in df_top_movies.columns]

        # Format numbers in the displayed dataframe
        df_display = df_top_movies[display_cols].copy()
        format_dict = {}
        if metric in df_display.columns: format_dict[metric] = "{:.2f}"
        if raw_metric_col in df_display.columns:
            if raw_metric_col in ['revenue', 'budget']:
                 format_dict[raw_metric_col] = "${:,.0f}"
            elif raw_metric_col in ['roi', 'popularity', 'vote_count']:
                 format_dict[raw_metric_col] = "{:,.2f}" # Adjust format as needed
            else: # Default for other raw metrics
                format_dict[raw_metric_col] = "{:.2f}"

        if raw_budget_col in df_display.columns: format_dict[raw_budget_col] = "${:,.0f}"
        if budget_col in df_display.columns and raw_budget_col not in df_display.columns: format_dict[budget_col] = "{:.2f}"


        st.dataframe(df_display.style.format(format_dict))


    st.markdown("---")
    st.caption("Analysis Complete. Findings based on historical correlations. Use alongside market knowledge.")
    st.caption(f"Note: Company/Country analysis only includes top entities present in the dataset based on preprocessing (Min Samples: {MIN_SAMPLE_SIZE_ENTITY}).")


# --- Main App ---
st.title("🎬 Movie Genre Performance Analyzer")

st.markdown("""
Select a primary genre and a success metric to explore historical performance trends.
Navigate the tabs below for insights on timing, co-genres, budget, production entities, and top movies.
""")

# Load Data
df_processed = load_data()

# --- Extract available genres/metrics/etc. AFTER loading ---
if df_processed is not None:
    available_genres = sorted([col for col in df_processed.columns if col.startswith('genre_')])
    available_metrics = sorted([
        'revenue_log', 'roi_log', 'popularity_log', 'vote_count_log',
        'vote_average'
    ])
    budget_col = 'budget_log'
    raw_budget_col = 'budget' # Used for display

    # --- User Inputs ---
    col1, col2 = st.columns(2)
    with col1:
        selected_genre = st.selectbox(
            "Select Primary Genre:",
            options=available_genres,
            format_func=lambda x: x.replace('genre_', '')
        )
    with col2:
        selected_metric = st.selectbox(
            "Select Success Metric:",
            options=available_metrics
        )

    # --- Run Analysis ---
    if selected_genre and selected_metric and df_processed is not None: # Check df again
        run_analysis(df_processed, selected_genre, selected_metric)
    elif df_processed is not None: # Only show this if data loaded but selections not made
        st.info("Please select a genre and a metric to start the analysis.")
    # Error message handled in load_data if df_processed is None

else:
    # This part is now less likely needed due to st.stop() in load_data failure
    st.error("Failed to load data. Cannot proceed.")