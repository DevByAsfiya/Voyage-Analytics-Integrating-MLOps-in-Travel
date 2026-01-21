import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

# Configure page
st.set_page_config(page_title="Hotel Recommender", layout="wide")
st.title("🏨 Hotel Recommendation System")
st.markdown("---")

@st.cache_data
def load_artifacts():
    """Load all trained artifacts"""
    base_path = "artifacts/hotel_recommendation"
    
    # Load model and preprocessor
    model = joblib.load(os.path.join(base_path, "Model_Data", "model.pkl"))
    preprocessor = joblib.load(os.path.join(base_path, "preprocessor.pkl"))
    
    # Load data tables
    hotel_features = pd.read_csv(os.path.join(base_path, "hotel_features.csv"))
    user_profile = pd.read_csv(os.path.join(base_path, "user_profile.csv"))
    
    st.success("✅ All models & data loaded!")
    return model, preprocessor, hotel_features, user_profile

# Load everything once
model, preprocessor, hotel_features, user_profile = load_artifacts()

# Define feature columns (must match training)
cat_cols = ['city', 'state', 'season', 'fav_city', 'fav_state', 'most_common_season']
num_cols = ['total_per_day', 'days', 'avg_price', 'avg_days', 'popularity']

class HotelRecommender:
    def __init__(self, model, preprocessor, hotel_features):
        self.model = model
        self.preprocessor = preprocessor
        self.hotel_features = hotel_features
    
    def score_user_hotels(self, user_prefs, top_n=10):
        """Generate recommendations for user preferences"""
        df = self.hotel_features.copy()
        
        # Apply user preferences to all hotels
        df['fav_city'] = user_prefs['fav_city']
        df['fav_state'] = user_prefs['fav_state']
        df['total_per_day'] = df['avg_price']
        df['days'] = user_prefs['days']
        df['season'] = df['most_common_season']
        
        # Prepare feature matrix
        feature_cols = cat_cols + num_cols
        X = df[feature_cols]
        
        # Predict scores
        X_transformed = self.preprocessor.transform(X)
        scores = self.model.predict_proba(X_transformed)[:, 1]
        df['score'] = scores
        
        # Return top recommendations
        return df.sort_values('score', ascending=False).head(top_n)

# Initialize recommender
recommender = HotelRecommender(model, preprocessor, hotel_features)

# Sidebar: User inputs
st.sidebar.header("👤 User Preferences")
mode = st.sidebar.radio("Recommendation Mode:", ["New User", "Existing User"])

if mode == "New User":
    st.sidebar.markdown("### New User (No History)")
    fav_city = st.sidebar.selectbox("Favorite City:", hotel_features['city'].unique())
    fav_state = st.sidebar.selectbox("Favorite State:", hotel_features['state'].unique())
    days = st.sidebar.slider("Nights:", 1, 7, 3)
    
    user_prefs = {
        'fav_city': fav_city,
        'fav_state': fav_state,
        'days': days
    }
    
else:  # Existing User
    st.sidebar.markdown("### Existing User")
    user_code = st.sidebar.selectbox("Select User:", user_profile['userCode'].unique())
    selected_user = user_profile[user_profile['userCode'] == user_code].iloc[0]
    
    user_prefs = {
        'fav_city': selected_user['fav_city'],
        'fav_state': selected_user['fav_state'],
        'days': int(selected_user['mean_days'])
    }
    st.sidebar.info(f"User {user_code}: {user_prefs['fav_city']}, {user_prefs['days']} nights")

# Generate recommendations
if st.button("🔮 Get Recommendations", type="primary"):
    with st.spinner("Generating personalized recommendations..."):
        recommendations = recommender.score_user_hotels(user_prefs, top_n=10)
    
    # Display results
    st.markdown("## 🎯 Your Top Recommendations")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Top Score", f"{recommendations.iloc[0]['score']:.1%}")
    with col2:
        st.metric("City Match", recommendations['city'].eq(user_prefs['fav_city']).sum())
    with col3:
        st.metric("Avg Price", f"${recommendations['avg_price'].mean():.0f}")
    
    # Top 5 hotels table
    top5 = recommendations.head(5)[['hotel_name', 'city', 'avg_price', 'score', 'popularity']]
    st.dataframe(top5.style.format({
        'avg_price': '${:.0f}',
        'score': '{:.1%}',
        'popularity': '{:.0f}'
    }), use_container_width=True)
    
    # Recommendation explanation
    top_hotel = recommendations.iloc[0]
    st.markdown(f"""
    ### 🤖 Why {top_hotel['hotel_name']}?
    - **Score**: {top_hotel['score']:.1%} preference probability
    - **City**: {top_hotel['city']} ({'✅ Matches preference' if top_hotel['city'] == user_prefs['fav_city'] else '🔄 Different city'})
    - **Price**: ${top_hotel['avg_price']:.0f} (reasonable for {user_prefs['days']} nights)
    - **Popularity**: {top_hotel['popularity']:.0f} bookings
    """)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using your MLOps pipeline | ROC-AUC: 0.608")
