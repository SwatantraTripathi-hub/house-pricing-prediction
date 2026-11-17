from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="House Price Intelligence",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "house_model.pkl"
DATA_PATH = BASE_DIR / "house_data_1000.csv"
FEATURES = [
    "area",
    "bedrooms",
    "bathrooms",
    "floors",
    "age",
    "parking",
    "location_score",
    "nearby_schools",
    "nearby_hospitals",
    "furnished",
]


@st.cache_resource
def load_artifact():
    """Load the fitted scaler and regressor from the saved training pipeline."""
    artifact = joblib.load(MODEL_PATH)

    if hasattr(artifact, "named_steps"):
        return artifact.named_steps["scaler"], artifact.named_steps["regressor"]

    if isinstance(artifact, dict):
        return artifact["scaler"], artifact.get("model", artifact["regressor"])

    raise TypeError(
        "house_model.pkl must contain a fitted model pipeline or model dictionary.")


@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)


def format_currency(value):
    return f"₹{value:,.0f}"


def inject_styles():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

        :root {
            --ink: #17211b;
            --muted: #657269;
            --green: #18794e;
            --mint: #e8f4ec;
            --cream: #fbfaf6;
            --line: #dce5de;
        }

        .stApp {
            background: linear-gradient(135deg, #fbfaf6 0%, #f1f7f1 52%, #e8f2ed 100%);
            color: var(--ink);
            font-family: 'DM Sans', sans-serif;
        }

        h1, h2, h3 {
            font-family: 'Space Grotesk', sans-serif;
            letter-spacing: 0;
            color: var(--ink);
        }

        h1 { font-size: clamp(2.1rem, 4vw, 3.8rem) !important; line-height: 1.05 !important; }
        .hero-kicker { color: var(--green); font-size: 0.78rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; }
        .hero-copy { color: var(--muted); font-size: 1.05rem; max-width: 720px; margin-bottom: 1.5rem; }
        .result-card {
            background: linear-gradient(120deg, #115a3a, #18794e);
            border-radius: 14px;
            color: white;
            padding: 1.6rem 1.8rem;
            box-shadow: 0 16px 35px rgba(24, 121, 78, 0.19);
        }
        .result-label { color: #bce7ca; font-size: 0.82rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; }
        .result-value { color: white; font-family: 'Space Grotesk', sans-serif; font-size: clamp(2rem, 4vw, 3.2rem); font-weight: 700; line-height: 1.15; margin: 0.25rem 0; }
        .result-note { color: #d7f0dd; font-size: 0.9rem; }
        [data-testid='stMetric'] { background: rgba(255, 255, 255, 0.72); border: 1px solid var(--line); border-radius: 12px; padding: 0.8rem 1rem; }
        [data-testid='stMetricLabel'] { color: var(--muted); }
        div.stButton > button { background: var(--green); border: 0; border-radius: 8px; color: white; font-weight: 700; min-height: 3rem; }
        div.stButton > button:hover { background: #115a3a; color: white; }
        section[data-testid='stSidebar'] {
            background: #10251b;
            border-right: 1px solid #294b39;
        }
        section[data-testid='stSidebar'] h2,
        section[data-testid='stSidebar'] p,
        section[data-testid='stSidebar'] label,
        section[data-testid='stSidebar'] [data-testid='stWidgetLabel'],
        section[data-testid='stSidebar'] [data-testid='stMarkdownContainer'] {
            color: #f1f7f2 !important;
        }
        section[data-testid='stSidebar'] [data-testid='stCaptionContainer'] p {
            color: #b7cbbd !important;
        }
        section[data-testid='stSidebar'] [data-baseweb='input'] {
            background: #1c3829;
            border-color: #527461;
        }
        section[data-testid='stSidebar'] [data-baseweb='input'] input {
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
        }
        section[data-testid='stSidebar'] [data-baseweb='select'] > div {
            background: #1c3829;
            border-color: #527461;
            color: #ffffff;
        }
        section[data-testid='stSidebar'] [data-baseweb='select'] span {
            color: #ffffff !important;
        }
        section[data-testid='stSidebar'] [data-testid='stSlider'] [role='slider'] {
            background: #8bd3a5;
            border-color: #d5f3df;
        }
        section[data-testid='stSidebar'] [data-testid='stSlider'] [data-baseweb='slider'] > div > div {
            background: #5bbd82;
        }
        section[data-testid='stSidebar'] div.stButton > button {
            background: #8bd3a5;
            color: #10251b;
        }
        section[data-testid='stSidebar'] div.stButton > button:hover {
            background: #b4e8c5;
            color: #10251b;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_styles()

try:
    scaler, model = load_artifact()
    dataset = load_dataset()
except Exception as error:
    st.error(f"Unable to load the prediction assets: {error}")
    st.stop()

with st.sidebar:
    st.markdown("## Property inputs")
    st.caption("Adjust the details below to generate a market estimate.")

    with st.form("prediction_form"):
        st.markdown("**Core property**")
        area = st.slider("Area (sq ft)", 300, 3000, 1000, 25)
        bedrooms = st.number_input(
            "Bedrooms", min_value=1, max_value=6, value=3, step=1)
        bathrooms = st.number_input(
            "Bathrooms", min_value=1, max_value=6, value=2, step=1)
        age = st.slider("Property age (years)", 0, 40, 8, 1)
        floors = st.slider("Floors", 1, 4, 2, 1)

        st.markdown("**Amenities and location**")
        parking = st.slider("Parking spaces", 0, 3, 1, 1)
        location_score = st.slider("Location factor", 1, 10, 6, 1)
        nearby_schools = st.slider("Nearby schools", 0, 8, 2, 1)
        nearby_hospitals = st.slider("Nearby hospitals", 0, 6, 2, 1)
        furnished = st.selectbox(
            "Furnished", [0, 1], format_func=lambda value: "Yes" if value else "No")

        submitted = st.form_submit_button(
            "Predict Price", use_container_width=True)

st.markdown('<div class="hero-kicker">House price intelligence</div>',
            unsafe_allow_html=True)
st.title("Make your next property decision with confidence.")
st.markdown(
    '<p class="hero-copy">A machine-learning estimate built from property characteristics, amenities, and location signals. Tune the profile in the sidebar to explore the value of your home.</p>',
    unsafe_allow_html=True,
)

if submitted:
    input_data = pd.DataFrame(
        [[
            area,
            bedrooms,
            bathrooms,
            floors,
            age,
            parking,
            location_score,
            nearby_schools,
            nearby_hospitals,
            furnished,
        ]],
        columns=FEATURES,
    )
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)[0]
    st.session_state["prediction"] = prediction

if "prediction" in st.session_state:
    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">Estimated market price</div>
            <div class="result-value">{format_currency(st.session_state["prediction"])}</div>
            <div class="result-note">Based on the property profile submitted above</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.info("Set your property profile in the sidebar and select Predict Price to generate an estimate.")

st.markdown("### Market snapshot")
metric_one, metric_two, metric_three, metric_four = st.columns(4)
metric_one.metric("Listings analyzed", f"{len(dataset):,}")
metric_two.metric("Median price", format_currency(dataset["price"].median()))
metric_three.metric("Average area", f"{dataset['area'].mean():,.0f} sq ft")
metric_four.metric("Typical location",
                   f"{dataset['location_score'].median():.0f}/10")

chart_left, chart_right = st.columns(2)
with chart_left:
    st.markdown("### Area and price relationship")
    st.caption(
        "Larger homes tend to command higher prices, with location adding variation.")
    st.scatter_chart(dataset[["area", "price"]],
                     x="area", y="price", height=300)

with chart_right:
    st.markdown("### Median price by bedrooms")
    st.caption(
        "A quick view of the price premium associated with additional bedrooms.")
    bedroom_prices = dataset.groupby(
        "bedrooms")["price"].median().rename("Median price")
    st.bar_chart(bedroom_prices, height=300, color="#18794e")

st.caption(
    "Model: Random Forest Regressor · 100 estimators · StandardScaler preprocessing")
