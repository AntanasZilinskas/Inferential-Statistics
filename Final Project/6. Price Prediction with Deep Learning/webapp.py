import streamlit as st
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from scipy.stats import percentileofscore
import altair as alt
import folium
import streamlit_folium as st_folium
import math

# main.py provides the following methods:
#   clean_dataframe(df)
#   load_model_for_inference() -> (model, cat_label2idx)
#   single_inference(model, cat_label2idx, example_dict) -> predicted_price
#   NUMERIC_COLS, CAT_COLS
from main import (
    clean_dataframe,
    load_model_for_inference,
    single_inference,
    NUMERIC_COLS, CAT_COLS
)

# Optional: If you want to load the same CSV for the distribution
CSV_FILE = os.path.join(os.path.dirname(__file__), "..", "listings_with_goodness.csv")

# A helper function to compute approximate distance between two lat/long pairs (in km).
# This uses a simple Haversine formula:
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius in km
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (math.sin(d_lat / 2)**2 
         + math.cos(math.radians(lat1)) 
         * math.cos(math.radians(lat2)) 
         * math.sin(d_lon / 2)**2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# ------------------------------------------------
# 1) Page config
# ------------------------------------------------
st.set_page_config(
    page_title="AirbnBling Price Predictor",
    layout="wide",  # wide layout for side-by-side columns
    initial_sidebar_state="auto",
)

# ------------------------------------------------
# 2) Minimal custom CSS
# ------------------------------------------------
st.markdown(
    """
    <style>
    /* Simple light-gray header container */
    .airbnb-header {
        background: #f7f7f7;
        padding: 1rem 1.5rem;
        border-radius: 0.4rem;
        margin-bottom: 1rem;
        text-align: center;
    }
    .airbnb-header h1 {
        color: #FF5A5F; /* Single accent color */
        font-size: 2rem;
        margin: 0;
        font-weight: 600;
    }
    .airbnb-header p {
        color: #666666;
        margin: 0.5rem 0 0;
        font-size: 1rem;
    }

    /* Moves the main content container closer to the top */
    .block-container {
        padding-top: 1rem !important;
        background-color: #ffffff; /* White background for content */
    }

    /* Section headings inside columns */
    .section-heading {
        font-size: 1.25rem;
        color: #333333;
        margin-bottom: 0.75rem;
        font-weight: 600;
    }

    /* Minimal horizontal divider color */
    hr[data-testid="stDivider"] {
        border-top: 1px solid #dddddd;
        margin: 1rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------------
# 3) Visually minimal header with brand accent
# ------------------------------------------------
st.markdown(
    """
    <div class="airbnb-header">
        <h1>AirbnBling Price Predictor</h1>
        <p>Make Your Airbnb Property Shine – Get Quick ROI Estimates!</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------------
# 4) Load model for inference
# ------------------------------------------------
model_infer, cat_label2idx = load_model_for_inference()

# ------------------------------------------------------
# (A) Read CSV once at the start to compute median values
# ------------------------------------------------------
median_values = {}
if os.path.exists(CSV_FILE):
    df_all_raw = pd.read_csv(CSV_FILE, low_memory=True)
    df_all_clean = clean_dataframe(df_all_raw)

    # If you stored log_price, revert back so numeric columns are consistent
    if "log_price" in df_all_clean.columns:
        df_all_clean["price"] = np.expm1(df_all_clean["log_price"])

    # Compute median of each numeric column (including price?)
    for col in NUMERIC_COLS:
        if col in df_all_clean.columns:
            median_values[col] = df_all_clean[col].median()
        else:
            # fallback or skip if the column is missing
            median_values[col] = 0.0

    # Optionally compute a median for "price" if you want
    # But typically the model is predicting log_price, so it's optional
else:
    st.warning("CSV file not found. Will use naive defaults for skipped fields.")
    # fallback naive defaults:
    for col in NUMERIC_COLS:
        median_values[col] = 0.0

# We'll define a reference lat/lon for our "city center" or reference point:
REF_LAT = 51.5074
REF_LON = -0.1278

# ------------------------------------------------
# 5) Side-by-side layout in a single form
# ------------------------------------------------
st.markdown('<div class="section-heading">Property Details</div>', unsafe_allow_html=True)

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    # Create a form so we can have the submission button under the map
    with st.form("prediction_form"):
        # -------------------------
        # Numeric row 1
        # -------------------------
        c1_1, c1_2, c1_3 = st.columns(3)
        with c1_1:
            bedrooms = st.number_input("Bedrooms", 0, 20, 2)
        with c1_2:
            bathrooms = st.number_input("Bathrooms", 0, 20, 1, step=1)
        with c1_3:
            accommodates = st.number_input("Accommodates", 1, 20, 4)

        # -------------------------
        # Numeric row 2
        # -------------------------
        c2_1, c2_2 = st.columns(2)
        with c2_1:
            host_listings_count = st.number_input(
                "Host Count (Number of properties hosted by the host)", 
                1, 999, 3
            )
        with c2_2:
            beds = st.number_input("Beds", 1, 20, 2)

        # -------------------------
        # Categorical row
        # -------------------------
        cat1, cat2, cat3 = st.columns(3)
        with cat1:
            room_type = st.selectbox("Room Type", 
                                     ["Entire home/apt", "Private room", "Shared room", "Hotel room"])
        with cat2:
            # Changed from st.selectbox("Superhost?", ["t", "f"])
            superhost_label = st.selectbox("Superhost?", ["True", "False"], index=1)
            host_is_superhost_map = {"True": "t", "False": "f"}
            host_is_superhost = host_is_superhost_map[superhost_label]
        with cat3:
            property_type = st.selectbox("Property Type", 
                                         ["Apartment", "House", "Bed & Breakfast", "Loft", "Other"])

        # -------------------------
        # Folium map (smaller) below
        # -------------------------
        st.markdown("Select a Location on the Map (click once, if multiple clicks, selects the last one):")
        default_lat = 51.5074
        default_lon = -0.1278
        m = folium.Map(location=[default_lat, default_lon], zoom_start=10)
        folium.TileLayer("cartodbpositron").add_to(m)
        folium.ClickForMarker(popup="Listing Location").add_to(m)

        map_data = st_folium.st_folium(m, width=350, height=300)

        # Submission button at the bottom (left side)
        submitted = st.form_submit_button("Estimate Nightly Rate")

with col_right:
    # We'll show results here after submission
    if submitted:
        # if user clicked, we take that lat/lon; else default
        if "last_clicked" in map_data and map_data["last_clicked"] is not None:
            user_lat = map_data["last_clicked"]["lat"]
            user_lon = map_data["last_clicked"]["lng"]
        else:
            user_lat = median_values.get("latitude", 51.5074)
            user_lon = median_values.get("longitude", -0.1278)

        # Now compute distance from reference lat/lon
        distance_km = haversine_distance(REF_LAT, REF_LON, user_lat, user_lon)

        example_numeric = {
            "distance_km": distance_km,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "accommodates": accommodates,
            "host_listings_count": host_listings_count,
            "beds": beds,
            # We'll use a constant/median rating since there's no UI slider:
            "review_scores_rating": median_values.get("review_scores_rating", 80.0),
            "latitude": user_lat,
            "longitude": user_lon,
            # "number_of_reviews" -> default from median
            "number_of_reviews": median_values.get("number_of_reviews", 0.0)
        }

        example_cat = {
            "room_type": room_type,
            "host_is_superhost": host_is_superhost,
            "property_type": property_type,
        }
        example_dict = {**example_numeric, **example_cat}

        predicted_price = single_inference(model_infer, cat_label2idx, example_dict)
        st.success(f"Estimated Nightly Price: ${predicted_price:,.2f}")

        # Optional: Distribution chart
        if os.path.exists(CSV_FILE):
            df_all = pd.read_csv(CSV_FILE, low_memory=True)
            df_all_clean = clean_dataframe(df_all)
            if "log_price" in df_all_clean.columns:
                df_all_clean["price"] = np.expm1(df_all_clean["log_price"])
            pctl = percentileofscore(df_all_clean["price"], predicted_price)
            st.write(f"That places your listing around the {pctl:.0f}th percentile of local listings.")

            base = alt.Chart(df_all_clean).mark_bar().encode(
                alt.X("price:Q", bin=alt.Bin(maxbins=50), title="Nightly Price"),
                alt.Y("count()", title="Count of listings"),
            ).properties(width=600, height=400, title="Distribution of Airbnb Prices")

            rule_df = pd.DataFrame({"predicted_price": [predicted_price]})
            rule = alt.Chart(rule_df).mark_rule(color="red", strokeWidth=3).encode(
                x="predicted_price:Q"
            )

            final_chart = (base + rule).interactive()
            st.altair_chart(final_chart, use_container_width=True)
        else:
            st.error("No CSV found for distribution analysis.")
    else:
        st.info("Fill out the details and pick a location on the map, then click 'Estimate Nightly Rate'.")

# ------------------------------------------------
# 7) Footer
# ------------------------------------------------
st.markdown("---")
st.markdown(
    '<p style="text-align:center; color:#999999;">'
    '© 2025 AirbnBling Price Predictor. All rights reserved.</p>',
    unsafe_allow_html=True
) 