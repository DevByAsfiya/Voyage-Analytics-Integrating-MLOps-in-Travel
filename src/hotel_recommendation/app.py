import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Configure page
st.set_page_config(page_title="Hotel Recommender", layout="wide", page_icon="🏨")
st.title("🏨 Content-Based Hotel Recommender")
st.markdown("---")

@st.cache_resource
def load_content_recommender():
    """Load content-based recommender artifacts"""
    base_path = "artifacts/hotel_recommendation/Content_Recommender"
    
    # Load COMPLETE recommender (your optimized pipeline)
    recommender_data = joblib.load(os.path.join(base_path, "content_recommender.pkl"))
    
    # Extract components
    preprocessor = recommender_data['preprocessor']
    hotel_matrix = recommender_data['hotel_matrix']
    hotel_features = recommender_data['hotel_features']
    
    # Test functions work
    assert callable(recommender_data['recommend_new_user'])
    
    st.success(f"✅ Content recommender loaded! {len(hotel_features):,} hotels")
    return recommender_data, hotel_features

# Load everything ONCE
recommender_data, hotel_features = load_content_recommender()
recommender = recommender_data['recommend_new_user']  # Direct function access
build_intent = recommender_data['build_intent_vector']


# Sidebar: User Preferences
st.sidebar.header("👤 Trip Preferences")
city = st.sidebar.selectbox(
    "Destination City:", 
    options=['All Cities'] + sorted(hotel_features['city'].unique().tolist())
)

season_options = ["winter", "summer", "autumn", "spring"]
season = st.sidebar.selectbox("Travel Season:", season_options)

budget = st.sidebar.slider("Budget ($/night)", 50, 500, 250)
nights = st.sidebar.slider("Nights:", 1, 7, 3)

# Advanced options
st.sidebar.markdown("---")
show_all_cities = st.sidebar.checkbox("Show hotels from other cities", value=False)

# Main recommendation button
if st.button("🚀 Find My Hotels", type="primary", use_container_width=True):
    with st.spinner("Computing personalized recommendations..."):
        # Generate recommendations (EXACTLY your Colab logic)
        recs = recommender(city, season, budget, nights, top_n=10)
        
        # Display metrics
        st.markdown("## 🎯 Your Perfect Hotels")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🏆 Top Score", f"{recs.iloc[0]['score']:.1%}")
        with col2:
            st.metric("💰 Avg Price", f"${recs['avg_price'].mean():.0f}")
        with col3:
            st.metric("📈 Most Popular", f"{recs['bookings'].max():.0f} bookings")
        with col4:
            n_city_matches = (recs['city'] == city).sum()
            st.metric("🎯 City Matches", n_city_matches)
        
        # Top recommendations table
        display_cols = ['hotel_name', 'city', 'avg_price', 'avg_days', 'most_common_season', 'bookings', 'score']
        styled_rec = recs[display_cols].style.format({
            'avg_price': '${:,.0f}',
            'score': '{:.1%}',
            'bookings': '{:.0f}'
        }).background_gradient(subset=['score'], cmap='RdYlGn')
        
        st.dataframe(styled_rec, use_container_width=True, hide_index=True)
        
        # Top hotel explanation
        top_hotel = recs.iloc[0]
        city_match = "✅ Perfect city match!" if top_hotel['city'] == city else "🔄 Best alternative"
        
        st.markdown(f"""
        ### 🌟 **Why {top_hotel['hotel_name']}?**
        - **Similarity**: {top_hotel['score']:.1%} (cosine similarity match)
        - **Location**: {top_hotel['city']} {city_match}
        - **Price**: ${top_hotel['avg_price']:,.0f} (fits ${budget} budget)
        - **Typical Stay**: {top_hotel['avg_days']:.1f} nights
        - **Popularity**: {top_hotel['bookings']:.0f} previous bookings
        """)
        
        # Price distribution chart
        st.markdown("### 💰 Price Comparison")
        fig_col1, fig_col2 = st.columns(2)
        with fig_col1:
            st.bar_chart(recs.head(5).set_index('hotel_name')['avg_price'])
        with fig_col2:
            st.bar_chart(recs.head(5).set_index('hotel_name')['score'])


# Stats sidebar
with st.sidebar.expander("📊 System Stats"):
    st.info(f"""
    **Hotels**: {len(hotel_features):,}
    **Cities**: {hotel_features['city'].nunique()}
    **Avg Price Range**: ${hotel_features['avg_price'].min():.0f} - ${hotel_features['avg_price'].max():.0f}
    **Pipeline**: Content-Based Similarity
    **Method**: Cosine Similarity on 6 features
    """)

# Instructions
with st.sidebar.expander("ℹ️ How it Works"):
    st.markdown("""
    1. **Content Features**: city, season, price_bucket, avg_price, avg_days, bookings
    2. **User Intent**: Converts your preferences to feature vector
    3. **Similarity**: Cosine distance to all hotels
    4. **City Priority**: Exact city matches first
    """)

# Footer
st.markdown("---")
st.markdown("""
*Built with your MLOps pipeline | Pure Content-Based Filtering | No training needed!*
**Deployed**: `joblib.load("content_recommender.pkl")`
""")

# Easter egg: Debug info
if st.checkbox("🔧 Debug - Raw Intent Vector"):
    price_bucket = "low" if budget < 150 else "medium" if budget < 300 else "high"
    intent_vec = build_intent(city, season, budget, nights, price_bucket)
    st.write("Intent Vector Shape:", intent_vec.shape)
    st.write("Raw vector:", intent_vec.flatten()[:5], "...")



####################################################

# import streamlit as st
# import pandas as pd
# import joblib
# import os
# import numpy as np

# # Configure page
# st.set_page_config(page_title="Hotel Recommender", layout="wide")
# st.title("🏨 Hotel Recommendation System")
# st.markdown("---")

# @st.cache_data
# def load_artifacts():
#     """Load all trained artifacts"""
#     base_path = "artifacts/hotel_recommendation"
    
#     # Load model and preprocessor
#     model = joblib.load(os.path.join(base_path, "Model_Data", "model.pkl"))
#     preprocessor = joblib.load(os.path.join(base_path, "preprocessor.pkl"))
    
#     # Load data tables
#     hotel_features = pd.read_csv(os.path.join(base_path, "hotel_features.csv"))
#     user_profile = pd.read_csv(os.path.join(base_path, "user_profile.csv"))
    
#     st.success("✅ All models & data loaded!")
#     return model, preprocessor, hotel_features, user_profile

# # Load everything once
# model, preprocessor, hotel_features, user_profile = load_artifacts()

# # Define feature columns (must match training)
# cat_cols = ['city', 'season', 'fav_city', 'most_common_season']
# num_cols = ['total_per_day', 'days', 'avg_price', 'avg_days', 'popularity']

# class HotelRecommender:
#     def __init__(self, model, preprocessor, hotel_features):
#         self.model = model
#         self.preprocessor = preprocessor
#         self.hotel_features = hotel_features
    
#     def score_user_hotels(self, user_prefs, top_n=10):
#         """Generate recommendations for user preferences"""
#         df = self.hotel_features.copy()
        
#         # Apply user preferences to all hotels
#         df['fav_city'] = user_prefs['fav_city']
#         # df['fav_state'] = user_prefs['fav_state']
#         df['total_per_day'] = df['avg_price']
#         df['days'] = user_prefs['days']
#         df['season'] = df['most_common_season']
        
#         # Prepare feature matrix
#         feature_cols = cat_cols + num_cols
#         X = df[feature_cols]
        
#         # Predict scores
#         X_transformed = self.preprocessor.transform(X)
#         scores = self.model.predict_proba(X_transformed)[:, 1]
#         df['score'] = scores
        
#         # Return top recommendations
#         return df.sort_values('score', ascending=False).head(top_n)

# # Initialize recommender
# recommender = HotelRecommender(model, preprocessor, hotel_features)

# # Sidebar: User inputs
# st.sidebar.header("👤 User Preferences")
# mode = st.sidebar.radio("Recommendation Mode:", ["New User", "Existing User"])

# if mode == "New User":
#     st.sidebar.markdown("### New User (No History)")
#     fav_city = st.sidebar.selectbox("Favorite City:", hotel_features['city'].unique())
#     # fav_state = st.sidebar.selectbox("Favorite State:", hotel_features['state'].unique())
#     days = st.sidebar.slider("Nights:", 1, 7, 3)
    
#     user_prefs = {
#         'fav_city': fav_city,
#         # 'fav_state': fav_state,
#         'days': days
#     }
    
# else:  # Existing User
#     st.sidebar.markdown("### Existing User")
#     user_code = st.sidebar.selectbox("Select User:", user_profile['userCode'].unique())
#     selected_user = user_profile[user_profile['userCode'] == user_code].iloc[0]
    
#     user_prefs = {
#         'fav_city': selected_user['fav_city'],
#         # 'fav_state': selected_user['fav_state'],
#         'days': int(selected_user['mean_days'])
#     }
#     st.sidebar.info(f"User {user_code}: {user_prefs['fav_city']}, {user_prefs['days']} nights")

# # Generate recommendations
# if st.button("🔮 Get Recommendations", type="primary"):
#     with st.spinner("Generating personalized recommendations..."):
#         recommendations = recommender.score_user_hotels(user_prefs, top_n=10)
    
#     # Display results
#     st.markdown("## 🎯 Your Top Recommendations")
    
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         st.metric("Top Score", f"{recommendations.iloc[0]['score']:.1%}")
#     with col2:
#         st.metric("City Match", recommendations['city'].eq(user_prefs['fav_city']).sum())
#     with col3:
#         st.metric("Avg Price", f"${recommendations['avg_price'].mean():.0f}")
    
#     # Top 5 hotels table
#     top5 = recommendations.head(5)[['hotel_name', 'city', 'avg_price', 'score', 'popularity']]
#     st.dataframe(top5.style.format({
#         'avg_price': '${:.0f}',
#         'score': '{:.1%}',
#         'popularity': '{:.0f}'
#     }), use_container_width=True)
    
#     # Recommendation explanation
#     top_hotel = recommendations.iloc[0]
#     st.markdown(f"""
#     ### 🤖 Why {top_hotel['hotel_name']}?
#     - **Score**: {top_hotel['score']:.1%} preference probability
#     - **City**: {top_hotel['city']} ({'✅ Matches preference' if top_hotel['city'] == user_prefs['fav_city'] else '🔄 Different city'})
#     - **Price**: ${top_hotel['avg_price']:.0f} (reasonable for {user_prefs['days']} nights)
#     - **Popularity**: {top_hotel['popularity']:.0f} bookings
#     """)

# # Footer
# st.markdown("---")
# st.markdown("Built with ❤️ using your MLOps pipeline | ROC-AUC: 0.608")
