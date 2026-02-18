"""
app.py — Nigerian SME Monthly Sales Prediction
===============================================
Beautiful blue-themed Streamlit web application
"""

import os
import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="🇳🇬 Nigerian SME Sales Predictor",
    page_icon="💼",
    layout="wide",
)

# ─────────────────────────────────────────────
# CUSTOM CSS — Beautiful Blue Theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background gradient */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Content container */
    .main .block-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem 3rem;
        margin-top: 2rem;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #1e3a8a !important;
        font-weight: 700 !important;
    }
    
    /* Remove default padding */
    .element-container {
        margin-bottom: 0.5rem;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.6);
    }
    
    /* Input widgets */
    .stSelectbox, .stSlider, .stNumberInput {
        background: white;
        border-radius: 10px;
    }
    
    /* Metrics */
    div[data-testid="stMetricValue"] {
        color: #1e40af;
        font-size: 1.8rem;
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 15px;
        border-left: 5px solid #3b82f6;
    }
    
    /* Dataframe */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
MODEL_PATH = "model/best_model"

FEATURE_META = {
    "business_type": {
        "type": "select",
        "options": ["Retail_Shop", "Restaurant", "Salon", "Electronics",
                    "Pharmacy", "Supermarket", "Fashion_Store"],
        "label": "🏪 Business Type",
    },
    "business_age_months": {
        "type": "slider",
        "min": 6, "max": 120, "default": 24, "step": 1,
        "label": "📅 Business Age (months)",
    },
    "location_type": {
        "type": "select",
        "options": ["Market", "Shopping_Mall", "Street_Shop", "Estate", "Online"],
        "label": "📍 Location Type",
    },
    "state": {
        "type": "select",
        "options": ["Lagos", "Abuja", "Kano", "Port_Harcourt", "Ibadan", "Enugu"],
        "label": "🗺️ State",
    },
    "num_employees": {
        "type": "slider",
        "min": 1, "max": 20, "default": 3, "step": 1,
        "label": "👥 Number of Employees",
    },
    "store_size_sqm": {
        "type": "slider",
        "min": 10, "max": 200, "default": 50, "step": 5,
        "label": "📐 Store Size (sqm)",
    },
    "foot_traffic_daily": {
        "type": "slider",
        "min": 20, "max": 500, "default": 100, "step": 10,
        "label": "🚶 Daily Foot Traffic",
    },
    "has_online_presence": {
        "type": "checkbox",
        "default": False,
        "label": "🌐 Has Online Presence",
    },
    "uses_pos": {
        "type": "checkbox",
        "default": True,
        "label": "💳 Uses POS System",
    },
    "marketing_spend_naira": {
        "type": "number",
        "min": 5000, "max": 200000, "default": 30000, "step": 1000,
        "label": "📢 Monthly Marketing Spend (₦)",
    },
    "inventory_value_naira": {
        "type": "number",
        "min": 100000, "max": 5000000, "default": 500000, "step": 10000,
        "label": "📦 Inventory Value (₦)",
    },
    "num_products": {
        "type": "slider",
        "min": 20, "max": 500, "default": 100, "step": 5,
        "label": "🛍️ Number of Products",
    },
    "average_product_price_naira": {
        "type": "number",
        "min": 500, "max": 50000, "default": 5000, "step": 100,
        "label": "💰 Average Product Price (₦)",
    },
    "month": {
        "type": "select",
        "options": ["January","February","March","April","May","June",
                    "July","August","September","October","November","December"],
        "label": "📆 Month",
    },
    "customer_retention_rate": {
        "type": "slider",
        "min": 30, "max": 95, "default": 60, "step": 1,
        "label": "🔄 Customer Retention Rate (%)",
    },
    "has_loyalty_program": {
        "type": "checkbox",
        "default": False,
        "label": "⭐ Has Loyalty Program",
    },
    "accepts_credit_cards": {
        "type": "checkbox",
        "default": True,
        "label": "💳 Accepts Credit/Debit Cards",
    },
    "opening_hours_per_day": {
        "type": "slider",
        "min": 8, "max": 16, "default": 12, "step": 1,
        "label": "⏰ Opening Hours Per Day",
    },
    "competition_nearby": {
        "type": "slider",
        "min": 0, "max": 10, "default": 3, "step": 1,
        "label": "🏢 Nearby Competitors",
    },
    "has_parking": {
        "type": "checkbox",
        "default": False,
        "label": "🅿️ Has Dedicated Parking",
    },
}

# ─────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="🔄 Loading AI model...")
def load_pycaret_model(path: str):
    """Load a saved PyCaret model pipeline."""
    try:
        from pycaret.regression import load_model
        model = load_model(path)
        return model, "regression"
    except Exception:
        pass
    try:
        from pycaret.classification import load_model
        model = load_model(path)
        return model, "classification"
    except Exception as e:
        st.error(f"❌ Could not load model from `{path}.pkl`")
        st.info("💡 Make sure you have run `python train.py` first.")
        st.stop()

# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────
def predict(model, task: str, input_df: pd.DataFrame):
    """Run PyCaret prediction."""
    if task == "regression":
        from pycaret.regression import predict_model
        result = predict_model(model, data=input_df)
        return result["prediction_label"].iloc[0], None
    else:
        from pycaret.classification import predict_model
        result = predict_model(model, data=input_df)
        pred = result["prediction_label"].iloc[0]
        score = result.get("prediction_score", pd.Series([None])).iloc[0]
        return pred, score

# ─────────────────────────────────────────────
# RENDER WIDGETS
# ─────────────────────────────────────────────
def render_widget(container, key: str):
    """Render a single widget."""
    meta = FEATURE_META[key]
    wtype = meta["type"]
    label = meta["label"]

    if wtype == "select":
        return container.selectbox(label, meta["options"], key=key)
    elif wtype == "slider":
        return container.slider(
            label,
            min_value=meta["min"],
            max_value=meta["max"],
            value=meta["default"],
            step=meta["step"],
            key=key,
        )
    elif wtype == "number":
        return container.number_input(
            label,
            min_value=meta["min"],
            max_value=meta["max"],
            value=meta["default"],
            step=meta["step"],
            key=key,
        )
    elif wtype == "checkbox":
        val = container.checkbox(label, value=meta["default"], key=key)
        return 1 if val else 0
    else:
        return container.text_input(label, key=key)

# ─────────────────────────────────────────────
# FORMATTING
# ─────────────────────────────────────────────
def format_naira(value_thousands: float) -> str:
    """Format prediction to readable Naira string."""
    naira = value_thousands * 1_000
    if naira >= 1_000_000:
        return f"₦{naira/1_000_000:.2f}M"
    elif naira >= 1_000:
        return f"₦{naira/1_000:,.1f}K"
    return f"₦{naira:,.0f}"

def sales_tier(value_thousands: float) -> tuple[str, str, str]:
    """Return label, color, and emoji."""
    if value_thousands >= 5_000:
        return "Very High Revenue", "#10b981", "🔥"
    elif value_thousands >= 1_000:
        return "High Revenue", "#3b82f6", "📈"
    elif value_thousands >= 300:
        return "Moderate Revenue", "#f59e0b", "📊"
    else:
        return "Low Revenue", "#ef4444", "📉"

# ─────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────
def main():
    # ── Beautiful Header with Nigerian Flag ───────
    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        st.image("https://upload.wikimedia.org/wikipedia/commons/7/79/Flag_of_Nigeria.svg", width=150)
    with col2:
        st.markdown("""
            <div style='text-align: center; padding: 1rem 0;'>
                <h1 style='font-size: 3.5rem; margin: 0; background: linear-gradient(135deg, #1e40af 0%, #7c3aed 100%); 
                           -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 900;'>
                    Nigerian SME Sales Predictor
                </h1>
                <p style='font-size: 1.3rem; color: #475569; margin-top: 0.5rem; font-weight: 500;'>
                    Powered by AI · Predict Your Monthly Revenue in Seconds
                </p>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.image("https://upload.wikimedia.org/wikipedia/commons/7/79/Flag_of_Nigeria.svg", width=150)

    # ── Check model exists ────────────────────
    if not os.path.exists(f"{MODEL_PATH}.pkl"):
        st.error("🚫 Model file not found. Please run `python train.py` first.")
        st.code("python train.py", language="bash")
        st.stop()

    model, task = load_pycaret_model(MODEL_PATH)

    # ── Input Section with Beautiful Cards ────
    st.markdown("""
        <div style='background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); 
                    padding: 1rem 2rem; border-radius: 15px; margin: 1rem 0; border-left: 5px solid #3b82f6;'>
            <h3 style='color: #1e40af; margin: 0;'>📝 Enter Your Business Details</h3>
        </div>
    """, unsafe_allow_html=True)

    user_inputs = {}

    # ── Row 1: Business Profile ──────────────
    st.markdown("""
        <div style='background: #eff6ff; padding: 0.8rem 1.5rem; border-radius: 10px; margin: 1.5rem 0 1rem 0; border-left: 4px solid #3b82f6;'>
            <h4 style='color: #1e40af; margin: 0;'>🏪 Business Profile</h4>
        </div>
    """, unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        user_inputs["business_type"] = render_widget(col1, "business_type")
    with col2:
        user_inputs["business_age_months"] = render_widget(col2, "business_age_months")
    with col3:
        user_inputs["location_type"] = render_widget(col3, "location_type")
    with col4:
        user_inputs["state"] = render_widget(col4, "state")

    # ── Row 2: Store Operations ──────────────
    st.markdown("""
        <div style='background: #eff6ff; padding: 0.8rem 1.5rem; border-radius: 10px; margin: 1.5rem 0 1rem 0; border-left: 4px solid #3b82f6;'>
            <h4 style='color: #1e40af; margin: 0;'>⚙️ Store Operations</h4>
        </div>
    """, unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        user_inputs["num_employees"] = render_widget(col1, "num_employees")
    with col2:
        user_inputs["store_size_sqm"] = render_widget(col2, "store_size_sqm")
    with col3:
        user_inputs["opening_hours_per_day"] = render_widget(col3, "opening_hours_per_day")
    with col4:
        user_inputs["foot_traffic_daily"] = render_widget(col4, "foot_traffic_daily")

    # ── Row 3: Inventory & Products ──────────
    st.markdown("""
        <div style='background: #eff6ff; padding: 0.8rem 1.5rem; border-radius: 10px; margin: 1.5rem 0 1rem 0; border-left: 4px solid #3b82f6;'>
            <h4 style='color: #1e40af; margin: 0;'>📦 Inventory & Products</h4>
        </div>
    """, unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        user_inputs["num_products"] = render_widget(col1, "num_products")
    with col2:
        user_inputs["inventory_value_naira"] = render_widget(col2, "inventory_value_naira")
    with col3:
        user_inputs["average_product_price_naira"] = render_widget(col3, "average_product_price_naira")
    with col4:
        user_inputs["competition_nearby"] = render_widget(col4, "competition_nearby")

    # ── Row 4: Marketing & Customer ──────────
    st.markdown("""
        <div style='background: #eff6ff; padding: 0.8rem 1.5rem; border-radius: 10px; margin: 1.5rem 0 1rem 0; border-left: 4px solid #3b82f6;'>
            <h4 style='color: #1e40af; margin: 0;'>📣 Marketing & Customer Engagement</h4>
        </div>
    """, unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        user_inputs["marketing_spend_naira"] = render_widget(col1, "marketing_spend_naira")
    with col2:
        user_inputs["customer_retention_rate"] = render_widget(col2, "customer_retention_rate")
    with col3:
        user_inputs["month"] = render_widget(col3, "month")

    # ── Row 5: Technology & Features ─────────
    st.markdown("""
        <div style='background: #eff6ff; padding: 0.8rem 1.5rem; border-radius: 10px; margin: 1.5rem 0 1rem 0; border-left: 4px solid #3b82f6;'>
            <h4 style='color: #1e40af; margin: 0;'>💳 Technology & Payment Options</h4>
        </div>
    """, unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        user_inputs["has_online_presence"] = render_widget(col1, "has_online_presence")
    with col2:
        user_inputs["uses_pos"] = render_widget(col2, "uses_pos")
    with col3:
        user_inputs["accepts_credit_cards"] = render_widget(col3, "accepts_credit_cards")
    with col4:
        user_inputs["has_loyalty_program"] = render_widget(col4, "has_loyalty_program")
    with col5:
        user_inputs["has_parking"] = render_widget(col5, "has_parking")

    # ── Predict Button (Centered & Large) ────
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        predict_btn = st.button("🔮 PREDICT MY MONTHLY SALES", use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Results Section ───────────────────────
    if predict_btn:
        with st.spinner("🤖 AI is analyzing your business..."):
            try:
                input_df = pd.DataFrame([user_inputs])
                prediction, confidence = predict(model, task, input_df)

                if task == "regression":
                    tier_label, tier_color, tier_emoji = sales_tier(float(prediction))
                    formatted = format_naira(float(prediction))

                    # Beautiful result card
                    st.markdown(f"""
                        <div style='
                            background: linear-gradient(135deg, #1e3a8a 0%, #7c3aed 100%);
                            border-radius: 20px;
                            padding: 3rem;
                            text-align: center;
                            color: white;
                            margin: 2rem 0;
                            box-shadow: 0 20px 60px rgba(30, 58, 138, 0.4);
                        '>
                            <div style='font-size: 1.2rem; opacity: 0.9; text-transform: uppercase; letter-spacing: 2px;'>
                                Your Predicted Monthly Revenue
                            </div>
                            <div style='font-size: 4.5rem; font-weight: 900; margin: 1rem 0; text-shadow: 0 4px 8px rgba(0,0,0,0.3);'>
                                {formatted}
                            </div>
                            <div style='font-size: 1.1rem; opacity: 0.85;'>
                                (₦{float(prediction):,.1f}K · {float(prediction)*1000:,.0f} Naira)
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

                    # Tier badge
                    st.markdown(f"""
                        <div style='
                            background: {tier_color}20;
                            border: 3px solid {tier_color};
                            border-radius: 15px;
                            padding: 1.5rem;
                            text-align: center;
                            font-size: 1.8rem;
                            font-weight: 700;
                            color: {tier_color};
                            margin-bottom: 2rem;
                        '>
                            {tier_emoji} {tier_label}
                        </div>
                    """, unsafe_allow_html=True)

                    # Insights in beautiful cards
                    st.markdown("### 💡 Business Insights")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    inv = user_inputs["inventory_value_naira"]
                    mkt = user_inputs["marketing_spend_naira"]
                    ret = user_inputs["customer_retention_rate"]
                    revenue_naira = float(prediction) * 1000
                    roi = (revenue_naira - inv) / inv * 100 if inv else 0
                    mkt_pct = mkt / revenue_naira * 100 if revenue_naira else 0

                    with col1:
                        st.markdown(f"""
                            <div style='background: linear-gradient(135deg, #10b981 0%, #059669 100%); 
                                        padding: 2rem; border-radius: 15px; text-align: center; color: white;
                                        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);'>
                                <div style='font-size: 0.9rem; opacity: 0.9;'>Inventory ROI</div>
                                <div style='font-size: 2.5rem; font-weight: 800; margin: 0.5rem 0;'>{roi:.1f}%</div>
                                <div style='font-size: 0.85rem; opacity: 0.85;'>Return on Investment</div>
                            </div>
                        """, unsafe_allow_html=True)

                    with col2:
                        st.markdown(f"""
                            <div style='background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); 
                                        padding: 2rem; border-radius: 15px; text-align: center; color: white;
                                        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);'>
                                <div style='font-size: 0.9rem; opacity: 0.9;'>Marketing Efficiency</div>
                                <div style='font-size: 2.5rem; font-weight: 800; margin: 0.5rem 0;'>{mkt_pct:.1f}%</div>
                                <div style='font-size: 0.85rem; opacity: 0.85;'>of Revenue Spent</div>
                            </div>
                        """, unsafe_allow_html=True)

                    with col3:
                        st.markdown(f"""
                            <div style='background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); 
                                        padding: 2rem; border-radius: 15px; text-align: center; color: white;
                                        box-shadow: 0 4px 15px rgba(245, 158, 11, 0.3);'>
                                <div style='font-size: 0.9rem; opacity: 0.9;'>Customer Loyalty</div>
                                <div style='font-size: 2.5rem; font-weight: 800; margin: 0.5rem 0;'>{ret}%</div>
                                <div style='font-size: 0.85rem; opacity: 0.85;'>Retention Rate</div>
                            </div>
                        """, unsafe_allow_html=True)

                else:
                    # Classification result
                    st.success("✅ Prediction complete!")
                    st.metric("Predicted Class", str(prediction))
                    if confidence is not None:
                        st.metric("Confidence", f"{float(confidence):.1%}")

            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
                st.exception(e)

    else:
        # Initial state - show welcome info
        st.info("👆 Fill in your business details above and click **PREDICT MY MONTHLY SALES** to get your AI-powered revenue forecast!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
                <div style='background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); 
                            padding: 2rem; border-radius: 15px; border: 2px solid #3b82f6;'>
                    <h3 style='color: #1e40af; margin-top: 0;'>📊 How It Works</h3>
                    <ol style='color: #1e3a8a; font-size: 1.05rem; line-height: 1.8;'>
                        <li>Enter your business parameters above</li>
                        <li>Click the predict button</li>
                        <li>Get instant AI-powered revenue forecast</li>
                        <li>Review personalized business insights</li>
                    </ol>
                </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
                <div style='background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); 
                            padding: 2rem; border-radius: 15px; border: 2px solid #3b82f6;'>
                    <h3 style='color: #1e40af; margin-top: 0;'> About the Model</h3>
                    <ul style='color: #1e3a8a; font-size: 1.05rem; line-height: 1.8;'>
                        <li><strong>4,000+</strong> Nigerian SME records</li>
                        <li><strong>PyCaret AutoML</strong> technology</li>
                        <li><strong>7</strong> business types covered</li>
                        <li><strong>6</strong> major Nigerian states</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)

    # ── Footer ────────────────────────────────
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
        <p style='text-align: center; color: #64748b; font-size: 0.95rem; font-weight: 500;'>
            Built with ❤️ for Nigerian SMEs · Empowering Business Owners with AI
            <br>
            <a href='https://smedan.gov.ng' target='_blank' style='color: #3b82f6; text-decoration: none;'>
                🇳🇬 SMEDAN Partner
            </a>
        </p>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()